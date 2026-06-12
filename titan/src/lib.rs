//! Titan `TradingVenue` adapter for the Quay program.
//!
//! Mirrors the contract in
//! `Titan-Pathfinder/integration-template/src/trading_venue/mod.rs`.
//! Lives in its own crate so Titan's dependency cone (litesvm, dashmap,
//! `solana-* 2.2` plus the long async-trait chain) doesn't infect
//! `quay-sdk`'s build or the on-chain BPF target.
//!
//! Each `QuayVenue` instance represents **one Strategy** — one pricing curve
//! on one `(base_mint, quote_mint)` pair. Titan's router holds many of these
//! (one per active strategy) and aggregates quote distribution across them.
//!
//! # Account-tracking model
//!
//! `get_required_pubkeys_for_update` returns nine pubkeys:
//! 1. the **Strategy** itself (bytecode + userspace + frozen flags + fee bps),
//! 2. the **MarketMaker** the strategy is anchored to (asset table + halt flags),
//! 3. the strategy's bound **Quotes** account,
//! 4. **GlobalConfig** (halt flags),
//! 5. + 6. the base and quote **mints** (decimals + Token-2022 detection),
//! 7. + 8. the base and quote **vault** token accounts (`LoadVault*` inputs),
//! 9. the **`Clock` sysvar** — `update_state` decodes `slot` + `unix_timestamp`
//!    here and threads them into `simulate_swap` so curves using
//!    `LoadNowSlot` / `LoadNowUnixSec` / `LoadArenaTimestampSec` see the
//!    same numbers a real swap would. The sysvar is a well-known account
//!    so Titan's cache dedups it across all Quay venues — one fetch per
//!    slot, not per venue.
//!
//! Heavy decoding lives in `update_state`; `quote` only re-runs the VM.
//! Mirrors the Jupiter sibling crate's "decode in update, simulate in quote"
//! split.
//!
//! # `PoolProtocol` branding
//!
//! `titan-integration-template`'s `PoolProtocol` enum is closed at the
//! template repo, so non-template venues cannot add their own variant
//! without forking. The adapter takes the upstream-supported route:
//! `protocol()` returns the generic `PoolProtocol::YourPoolProtocol`
//! discriminant and `label()` returns `"Quay"`, which is the brand
//! string Titan's UI and routing logs read.
//!
//! # Program id
//!
//! [`QUAY_PROGRAM_ID`] holds Quay's mainnet program id
//! (`QUayE6nexQWYNZAEqfN8FxoNwQDSu3CAzT2qq9J1ArG`). Routers constructing
//! `QuayVenue` via `FromAccount::from_account` read the actual program id
//! directly off `account.owner`, so the constant only matters for callers
//! that need `program_id()` *before* loading the Strategy (e.g. to filter
//! `getProgramAccounts` queries to Quay strategies).

use async_trait::async_trait;
use solana_account::Account;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;
use spl_associated_token_account::get_associated_token_address_with_program_id;

use quay_sdk::consts::{SIDE_BUY_BASE, SIDE_SELL_BASE, SWAP_LOADED_ACCOUNTS_DATA_SIZE_LIMIT};
use quay_sdk::ix;
use quay_sdk::pda;
use quay_sdk::simulate::{simulate_swap, SwapSimulationInputs};
use quay_sdk::state::{GlobalConfig, MarketMakerHeader, StrategyHeader};

use titan_integration_template::account_caching::AccountsCache;
use titan_integration_template::trading_venue::{
    error::TradingVenueError,
    protocol::PoolProtocol,
    token_info::TokenInfo,
    FromAccount, QuoteRequest, QuoteResult, SwapType, TradingVenue,
};

// ──────────────────────────────────────────────────────────────────────────
// Public constants
// ──────────────────────────────────────────────────────────────────────────

/// Quay's mainnet program id.
///
/// Sibling adapters (e.g. `quay-aggregator-jupiter`) re-export the same
/// constant. Routers constructing `QuayVenue` via
/// `FromAccount::from_account` see the program id directly off
/// `account.owner`; this constant exists for callers that need to
/// reference the program id without first loading a Strategy account
/// (e.g. building a `getProgramAccounts` filter).
pub const QUAY_PROGRAM_ID: Pubkey = solana_pubkey::pubkey!("QUayE6nexQWYNZAEqfN8FxoNwQDSu3CAzT2qq9J1ArG");

/// Recommended `set_loaded_accounts_data_size_limit` value for every swap
/// tx Titan composes against Quay. Mirrors
/// `aggregators/jupiter/src/lib.rs::QuayAmm::RECOMMENDED_LOADED_ACCOUNTS_DATA_SIZE_LIMIT`.
pub const RECOMMENDED_LOADED_ACCOUNTS_DATA_SIZE_LIMIT: u32 = SWAP_LOADED_ACCOUNTS_DATA_SIZE_LIMIT;

/// Account count the on-chain `swap` ix expects: eleven program-read
/// positional accounts (cfg + strategy + mm + quotes + 2 vaults + 2 taker
/// ATAs + taker + 2 mints), the Instructions sysvar at index 11, and one
/// trailing token-program slot the program skips but the runtime must
/// load for the transfer CPIs. A mixed token-program market appends a
/// second program — see the SDK's `push_token_programs` dedup.
pub const SWAP_ACCOUNTS_LEN: usize = 13;

/// Solana's `Clock` sysvar address. Fetched by `update_state` so curves see
/// the live slot + unix-time off the same `AccountsCache` batch Titan uses
/// for everything else — no upstream-trait change needed. Re-exported from
/// `solana-sdk-ids` so we don't carry a hard-coded pubkey literal.
pub const SYSVAR_CLOCK_ID: Pubkey = solana_sdk_ids::sysvar::clock::ID;

/// `Clock` sysvar layout: `slot` (u64 LE, 0..8) + `epoch_start_timestamp`
/// (i64 LE, 8..16) + `epoch` (u64 LE, 16..24) + `leader_schedule_epoch`
/// (u64 LE, 24..32) + `unix_timestamp` (i64 LE, 32..40). We only read slot
/// and unix_timestamp.
const CLOCK_SLOT_OFFSET: usize = 0;
const CLOCK_UNIX_TS_OFFSET: usize = 32;
const CLOCK_MIN_LEN: usize = CLOCK_UNIX_TS_OFFSET + 8;

/// Decode `(slot, unix_timestamp)` from a `Clock` sysvar account. Returns
/// `None` if the blob is shorter than the layout demands — `update_state`
/// surfaces that as a `DeserializationFailed`.
fn decode_clock(data: &[u8]) -> Option<(u64, i64)> {
    if data.len() < CLOCK_MIN_LEN {
        return None;
    }
    let mut slot_buf = [0u8; 8];
    slot_buf.copy_from_slice(&data[CLOCK_SLOT_OFFSET..CLOCK_SLOT_OFFSET + 8]);
    let mut ts_buf = [0u8; 8];
    ts_buf.copy_from_slice(&data[CLOCK_UNIX_TS_OFFSET..CLOCK_UNIX_TS_OFFSET + 8]);
    Some((u64::from_le_bytes(slot_buf), i64::from_le_bytes(ts_buf)))
}

// ──────────────────────────────────────────────────────────────────────────
// QuayVenue
// ──────────────────────────────────────────────────────────────────────────

/// A single Quay Strategy exposed as a Titan trading venue.
///
/// Constructed via `FromAccount::from_account` (passing the Strategy
/// `Account`) and then refreshed via `update_state` every slot. The four
/// raw account blobs (`strategy_data` / `mm_data` / `quotes_data` /
/// `global_config_data`) are kept in-struct because
/// `quay_sdk::simulate::simulate_swap` reads them as opaque `&[u8]`.
#[derive(Clone)]
pub struct QuayVenue {
    /// Quay program id. Pulled off `Strategy.account.owner` at construction
    /// time so the venue is robust against devnet/testnet deploys where
    /// the program lives under a different key than [`QUAY_PROGRAM_ID`].
    program_id: Pubkey,

    /// Strategy account pubkey. Returned by `market_id()`.
    strategy_key: Pubkey,
    strategy_data: Vec<u8>,

    /// MarketMaker the strategy is anchored to (PDA of `strategy.owner`).
    mm_key: Pubkey,
    mm_data: Vec<u8>,

    /// Strategy's bound quotes account (`strategy.quotes_account`).
    quotes_key: Pubkey,
    quotes_data: Vec<u8>,

    /// `GlobalConfig` PDA — cached at construction.
    global_config_key: Pubkey,
    global_config_data: Vec<u8>,

    base_mint: Pubkey,
    quote_mint: Pubkey,

    /// Vault PDAs (base / quote sides) + cached account data so
    /// `simulate_swap` can supply `vault_(base|quote)_atoms` to curves that
    /// read them.
    vault_base_key: Pubkey,
    vault_quote_key: Pubkey,
    vault_base_data: Vec<u8>,
    vault_quote_data: Vec<u8>,

    /// `[base, quote]` `TokenInfo` (mints + decimals + Token-2022 program
    /// detection). Populated by `update_state`; empty before the first call.
    tokens: Vec<TokenInfo>,

    /// Cached halt / freeze bytes — same set the on-chain `execute_swap`
    /// enforces, mirroring `aggregators/jupiter`'s [`QuayAmm`]. Every flag
    /// must read 0 for `initialized()` to return true. Bytes are sourced
    /// from `GlobalConfig` / `MarketMakerHeader` / `StrategyHeader` headers
    /// at the end of each `update_state`.
    ///
    /// Default to `1` (active-halt) before the first `update_state` so the
    /// router refuses to quote against half-decoded state during warmup —
    /// same convention Jupiter uses.
    cfg_swap_halted: u8,
    cfg_protocol_halted: u8,
    strategy_frozen: u8,
    strategy_frozen_admin: u8,
    mm_frozen: u8,
    mm_frozen_admin: u8,
    mm_halted_admin: u8,

    /// Wall clock the venue threads into `simulate_swap`. Production source
    /// is the `Clock` sysvar, fetched alongside the strategy / mm / vault
    /// blobs in `update_state` (see [`SYSVAR_CLOCK_ID`]). Callers running
    /// outside the Titan pipeline — replay tests, off-line backtests —
    /// can override post-update via [`QuayVenue::with_clock`] or
    /// [`QuayVenue::set_clock`]. Defaults to `0` so a `QuayVenue` queried
    /// before its first `update_state` quotes as if the clock were unset.
    current_slot: u64,
    current_unix_sec: i64,
}

impl QuayVenue {
    /// Account list the on-chain `swap` ix expects, in the order locked by
    /// `instructions/swap.rs`. Delegated to `quay_sdk::ix::swap` so the
    /// trailing token-program dedup matches the live builder.
    ///
    /// Exposed `pub` so callers building swap ixs outside the `TradingVenue`
    /// trait can reuse our cached PDAs without re-deriving.
    pub fn swap_account_metas(
        &self,
        taker: &Pubkey,
        taker_ata_base: &Pubkey,
        taker_ata_quote: &Pubkey,
        side: u8,
        base_token_program: &Pubkey,
        quote_token_program: &Pubkey,
    ) -> Vec<AccountMeta> {
        ix::swap(
            &self.program_id,
            &self.strategy_key,
            &self.mm_key,
            &self.quotes_key,
            &self.base_mint,
            &self.quote_mint,
            taker,
            taker_ata_base,
            taker_ata_quote,
            base_token_program,
            quote_token_program,
            0, // amount_in — callers needing only metas pass 0.
            0, // min_amount_out
            side,
        )
        .accounts
    }

    /// Did `update_state` populate the dependent account blobs? Mirrors the
    /// Jupiter adapter's "lazy" model where construction caches the Strategy
    /// bytes but defers loading the rest to the first update.
    fn has_all_state(&self) -> bool {
        !self.mm_data.is_empty()
            && !self.quotes_data.is_empty()
            && !self.global_config_data.is_empty()
    }

    /// Halt-gate mirror — `true` only when every flag the on-chain
    /// `execute_swap` checks is clear. Used by both `initialized()` and the
    /// short-circuit at the top of `quote()`.
    fn halts_clear(&self) -> bool {
        self.cfg_swap_halted == 0
            && self.cfg_protocol_halted == 0
            && self.strategy_frozen == 0
            && self.strategy_frozen_admin == 0
            && self.mm_frozen == 0
            && self.mm_frozen_admin == 0
            && self.mm_halted_admin == 0
    }

    /// Did `update_state` find a `TransferFeeConfig` extension on either
    /// mint? The on-chain `swap` doesn't subtract the fee from `amount_in`
    /// before pricing, so a transfer-fee'd mint mis-quotes — refuse the
    /// venue until the program is taught to net the fee.
    fn any_transfer_fee(&self) -> bool {
        self.tokens.iter().any(|t| t.transfer_fee.is_some())
    }

    /// Look up the token-program owner for one of the strategy's mints.
    fn token_program_for(&self, mint: &Pubkey) -> Option<Pubkey> {
        self.tokens
            .iter()
            .find(|t| t.pubkey == *mint)
            .map(TokenInfo::get_token_program)
    }

    /// Override the wall clock the venue feeds into `simulate_swap`.
    /// Production routers don't need this — `update_state` fetches the
    /// `Clock` sysvar through Titan's `AccountsCache` and updates both
    /// fields every slot. Useful for replay tests / off-line backtests
    /// that want to pin the venue to a historical clock domain.
    #[must_use]
    pub fn with_clock(mut self, current_slot: u64, current_unix_sec: i64) -> Self {
        self.current_slot = current_slot;
        self.current_unix_sec = current_unix_sec;
        self
    }

    /// In-place variant of [`Self::with_clock`]. Note that the next
    /// `update_state` call will overwrite both fields from the sysvar — use
    /// this only when the next caller is `quote()` and you intend to bypass
    /// the live clock.
    pub fn set_clock(&mut self, current_slot: u64, current_unix_sec: i64) {
        self.current_slot = current_slot;
        self.current_unix_sec = current_unix_sec;
    }
}

/// Read an SPL token account's `amount` (u64 LE at bytes 64..72). Caller
/// must verify `data.len() >= 72`; replaces an earlier `try_into().unwrap()`
/// against the zero-warnings policy.
fn read_vault_atoms(data: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[64..72]);
    u64::from_le_bytes(buf)
}

// ──────────────────────────────────────────────────────────────────────────
// FromAccount
// ──────────────────────────────────────────────────────────────────────────

impl FromAccount for QuayVenue {
    fn from_account(pubkey: &Pubkey, account: &Account) -> Result<Self, TradingVenueError> {
        let strategy = StrategyHeader::try_from_account(&account.data).map_err(|e| {
            TradingVenueError::DeserializationFailed(format!("StrategyHeader: {e}").into())
        })?;

        // `account.owner` is the on-chain program id — read it directly
        // off the loaded Strategy so the venue works against any deploy
        // (mainnet, devnet, local validator). Same convention as the
        // sibling Jupiter adapter.
        let program_id = account.owner;
        let base_mint = Pubkey::new_from_array(strategy.base_mint);
        let quote_mint = Pubkey::new_from_array(strategy.quote_mint);
        let strategy_owner = Pubkey::new_from_array(strategy.owner);
        let quotes_key = Pubkey::new_from_array(strategy.quotes_account);

        let (mm_key, _) = pda::market_maker_pda(&program_id, &strategy_owner);
        let (global_config_key, _) = pda::global_config_pda(&program_id);
        let (vault_base_key, _) = pda::vault_pda(&program_id, &mm_key, &base_mint);
        let (vault_quote_key, _) = pda::vault_pda(&program_id, &mm_key, &quote_mint);

        Ok(Self {
            program_id,
            strategy_key: *pubkey,
            strategy_data: account.data.clone(),
            mm_key,
            mm_data: Vec::new(),
            quotes_key,
            quotes_data: Vec::new(),
            global_config_key,
            global_config_data: Vec::new(),
            base_mint,
            quote_mint,
            vault_base_key,
            vault_quote_key,
            vault_base_data: Vec::new(),
            vault_quote_data: Vec::new(),
            tokens: Vec::new(),
            // Default to 1 (active-halt) so `initialized()` returns false
            // until the first `update_state` decodes real flag bytes.
            cfg_swap_halted: 1,
            cfg_protocol_halted: 1,
            // Strategy flags can be read off the bytes we already have,
            // but keep symmetry with the MM / cfg defaults: warm up halted.
            strategy_frozen: 1,
            strategy_frozen_admin: 1,
            mm_frozen: 1,
            mm_frozen_admin: 1,
            mm_halted_admin: 1,
            current_slot: 0,
            current_unix_sec: 0,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────
// TradingVenue
// ──────────────────────────────────────────────────────────────────────────

#[async_trait]
impl TradingVenue for QuayVenue {
    fn initialized(&self) -> bool {
        // Three gates Titan's route planner uses to skip the venue:
        //   1. all account blobs populated (post-first-update),
        //   2. on-chain halt / freeze set clear (the same flags the swap
        //      handler checks — see `onchain/program/src/instructions/swap.rs`),
        //   3. neither mint carries a Token-2022 `TransferFeeConfig`
        //      extension (the on-chain swap prices `amount_in` gross and
        //      would short-fill against the curve's quoted output).
        self.has_all_state() && self.halts_clear() && !self.any_transfer_fee()
    }

    fn program_id(&self) -> Pubkey {
        self.program_id
    }

    fn program_dependencies(&self) -> Vec<Pubkey> {
        // Declare both token programs so Titan's address-lookup-table
        // machinery sees them as venue prerequisites. The on-chain `swap`
        // dispatches the right one at runtime from each mint's `owner`. The
        // system program is included because the on-chain handlers CPI into
        // it for vault-rent payments during admin ixs.
        vec![
            pda::SPL_TOKEN_PROGRAM_ID,
            pda::TOKEN_2022_PROGRAM_ID,
            // System program (`11111111111111111111111111111111`).
            solana_sdk_ids::system_program::ID,
        ]
    }

    fn market_id(&self) -> Pubkey {
        self.strategy_key
    }

    fn get_token_info(&self) -> &[TokenInfo] {
        &self.tokens
    }

    fn protocol(&self) -> PoolProtocol {
        // The upstream `PoolProtocol` enum is closed at the template
        // repo, so non-template venues use the generic discriminant
        // and carry the brand string through `label()`. See the
        // module-level "PoolProtocol branding" note.
        PoolProtocol::YourPoolProtocol
    }

    fn label(&self) -> String {
        "Quay".to_string()
    }

    fn get_required_pubkeys_for_update(&self) -> Result<Vec<Pubkey>, TradingVenueError> {
        // Trailing `SYSVAR_CLOCK_ID` is a well-known sysvar, so Titan's
        // batch fetcher dedups it across every Quay venue in the route
        // graph — one fetch per slot, not per venue.
        Ok(vec![
            self.strategy_key,
            self.mm_key,
            self.quotes_key,
            self.global_config_key,
            self.base_mint,
            self.quote_mint,
            self.vault_base_key,
            self.vault_quote_key,
            SYSVAR_CLOCK_ID,
        ])
    }

    async fn update_state(&mut self, cache: &dyn AccountsCache) -> Result<(), TradingVenueError> {
        let needed = self.get_required_pubkeys_for_update()?;
        let accounts = cache.get_accounts(&needed).await?;
        if accounts.len() != needed.len() {
            return Err(TradingVenueError::FailedToFetchMultipleAccountData);
        }

        // Slot 0 — Strategy. Re-decode so the frozen flags reflect the
        // latest state we'd see at swap time. Cache the two strategy halt
        // bytes so `initialized()` short-circuits without re-decoding.
        let strategy_account = accounts[0]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.strategy_key.into()))?;
        self.strategy_data = strategy_account.data.clone();
        let strategy = StrategyHeader::try_from_account(&self.strategy_data).map_err(|e| {
            TradingVenueError::DeserializationFailed(format!("StrategyHeader: {e}").into())
        })?;
        self.strategy_frozen = strategy.frozen;
        self.strategy_frozen_admin = strategy.frozen_admin;

        // Slot 1 — MarketMaker (asset table + admin halts).
        let mm_account = accounts[1]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.mm_key.into()))?;
        self.mm_data = mm_account.data.clone();
        let mm = MarketMakerHeader::try_from_account(&self.mm_data).map_err(|e| {
            TradingVenueError::DeserializationFailed(format!("MarketMakerHeader: {e}").into())
        })?;
        self.mm_frozen = mm.frozen;
        self.mm_frozen_admin = mm.frozen_admin;
        self.mm_halted_admin = mm.halted_admin;

        // Slot 2 — Quotes.
        let quotes_account = accounts[2]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.quotes_key.into()))?;
        self.quotes_data = quotes_account.data.clone();

        // Slot 3 — GlobalConfig. Cache the two cfg halt bytes alongside the
        // raw blob.
        let cfg_account = accounts[3]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.global_config_key.into()))?;
        self.global_config_data = cfg_account.data.clone();
        let cfg = GlobalConfig::try_from_account(&self.global_config_data).map_err(|e| {
            TradingVenueError::DeserializationFailed(format!("GlobalConfig: {e}").into())
        })?;
        self.cfg_swap_halted = cfg.swap_halted;
        self.cfg_protocol_halted = cfg.protocol_halted;

        // Slots 4 + 5 — base and quote mints. `TokenInfo::new` handles both
        // SPL Token and Token-2022 layouts (Token-2022 is a strict superset,
        // and `StateWithExtensions::unpack` accepts both). The epoch argument
        // is for epoch-indexed Token-2022 fees; we don't have a clock here
        // so we pass 0 — fine for routing, the on-chain swap picks up the
        // live value when it executes.
        let base_mint_account = accounts[4]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.base_mint.into()))?;
        let quote_mint_account = accounts[5]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.quote_mint.into()))?;
        let base_info = TokenInfo::new(&self.base_mint, base_mint_account, 0)?;
        let quote_info = TokenInfo::new(&self.quote_mint, quote_mint_account, 0)?;
        self.tokens = vec![base_info, quote_info];

        // Slots 6 + 7 — base and quote vaults. Needed for `LoadVaultBase` /
        // `LoadVaultQuote`.
        let vault_base_account = accounts[6]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.vault_base_key.into()))?;
        let vault_quote_account = accounts[7]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(self.vault_quote_key.into()))?;
        self.vault_base_data = vault_base_account.data.clone();
        self.vault_quote_data = vault_quote_account.data.clone();

        // Slot 8 — `Clock` sysvar. Replaces the `with_clock` / `set_clock`
        // path as the production source for `current_slot` /
        // `current_unix_sec`. Freshness budget is one batch (≈1 slot, the
        // same as every other account here). Callers that want a different
        // clock domain (replay tests, off-line backtests) can still
        // override post-update via `set_clock`.
        let clock_account = accounts[8]
            .as_ref()
            .ok_or_else(|| TradingVenueError::NoAccountFound(SYSVAR_CLOCK_ID.into()))?;
        let (slot, unix_sec) = decode_clock(&clock_account.data).ok_or_else(|| {
            TradingVenueError::DeserializationFailed("Clock sysvar too short".into())
        })?;
        self.current_slot = slot;
        self.current_unix_sec = unix_sec;

        Ok(())
    }

    fn quote(&self, request: QuoteRequest) -> Result<QuoteResult, TradingVenueError> {
        // ExactOut: Quay's swap ix is exact-in only and the DSL is a forward
        // pricer (no inverse). Reject cleanly so Titan's router degrades
        // gracefully.
        if matches!(request.swap_type, SwapType::ExactOut) {
            return Err(TradingVenueError::ExactOutNotSupported);
        }

        // Zero-amount quotes: trait contract says we must not panic. Return
        // an empty quote rather than calling into the simulator (which
        // rejects `out_to_taker == 0` with `ZeroOutputSwap`).
        if request.amount == 0 {
            return Ok(QuoteResult {
                input_mint: request.input_mint,
                output_mint: request.output_mint,
                amount: 0,
                expected_output: 0,
                not_enough_liquidity: false,
            });
        }

        // Map mints to Quay's `side` byte. 0 = sell base, 1 = buy base.
        let side = if request.input_mint == self.base_mint
            && request.output_mint == self.quote_mint
        {
            SIDE_SELL_BASE
        } else if request.input_mint == self.quote_mint && request.output_mint == self.base_mint {
            SIDE_BUY_BASE
        } else {
            return Err(TradingVenueError::InvalidMint(request.input_mint.into()));
        };

        // Mirror `initialized()`: state populated AND halts clear AND no
        // transfer-fee mints. The simulator would also reject on the halt
        // bytes (`client/sdk/src/simulate.rs`), but failing here gives the
        // router a single canonical "not initialized" surface to skip.
        if !self.has_all_state() || !self.halts_clear() || self.any_transfer_fee() {
            return Err(TradingVenueError::NotInitialized(self.strategy_key.into()));
        }

        // `TokenInfo.decimals` is `i32` in the upstream template; SPL mints
        // always fit in `u8` (0..=18), so a narrowing cast is safe.
        let base_decimals = self
            .tokens
            .iter()
            .find(|t| t.pubkey == self.base_mint)
            .map(|t| t.decimals as u8)
            .ok_or_else(|| TradingVenueError::MissingState("base TokenInfo".into()))?;
        let quote_decimals = self
            .tokens
            .iter()
            .find(|t| t.pubkey == self.quote_mint)
            .map(|t| t.decimals as u8)
            .ok_or_else(|| TradingVenueError::MissingState("quote TokenInfo".into()))?;

        // Vault depths → `LoadVaultBase` / `LoadVaultQuote`. `amount` is
        // u64 LE at offset 64..72 of every SPL Token / Token-2022 account.
        if self.vault_base_data.len() < 72 || self.vault_quote_data.len() < 72 {
            return Err(TradingVenueError::AmmMethodError(
                "vault account data too short for LoadVault*".into(),
            ));
        }
        let vault_base_atoms = read_vault_atoms(&self.vault_base_data);
        let vault_quote_atoms = read_vault_atoms(&self.vault_quote_data);

        // Clock: live values from the `Clock` sysvar Titan's `AccountsCache`
        // fetched in `update_state`, or a `with_clock` / `set_clock`
        // override. `0` means "no update_state has run and no override was
        // attached" — curves fall back to `strategy.last_update_slot`.
        let sim = simulate_swap(SwapSimulationInputs {
            strategy_data: &self.strategy_data,
            market_maker_data: &self.mm_data,
            quotes_data: &self.quotes_data,
            global_config_data: &self.global_config_data,
            current_slot: self.current_slot,
            current_unix_sec: self.current_unix_sec,
            side,
            amount_in: request.amount,
            min_amount_out: 0,
            vault_base_atoms,
            vault_quote_atoms,
            base_decimals,
            quote_decimals,
        })
        .map_err(|e| TradingVenueError::AmmMethodError(format!("simulate_swap: {e}").into()))?;

        Ok(QuoteResult {
            input_mint: request.input_mint,
            output_mint: request.output_mint,
            amount: request.amount,
            expected_output: sim.out_to_taker,
            // Quay's curves cannot short-fill — the simulator either prices
            // the full `amount_in` or returns an error. If a future curve
            // gains short-fill we surface it via `sim.out_to_taker < expected`.
            not_enough_liquidity: false,
        })
    }

    fn generate_swap_instruction(
        &self,
        request: QuoteRequest,
        user: Pubkey,
    ) -> Result<Instruction, TradingVenueError> {
        if matches!(request.swap_type, SwapType::ExactOut) {
            return Err(TradingVenueError::ExactOutNotSupported);
        }

        let side = if request.input_mint == self.base_mint
            && request.output_mint == self.quote_mint
        {
            SIDE_SELL_BASE
        } else if request.input_mint == self.quote_mint && request.output_mint == self.base_mint {
            SIDE_BUY_BASE
        } else {
            return Err(TradingVenueError::InvalidMint(request.input_mint.into()));
        };

        // Derive taker ATAs using the per-mint token program (Token-2022
        // ATAs live under a different program id than SPL Token ATAs).
        let base_program = self
            .token_program_for(&self.base_mint)
            .ok_or_else(|| TradingVenueError::MissingState("base TokenInfo".into()))?;
        let quote_program = self
            .token_program_for(&self.quote_mint)
            .ok_or_else(|| TradingVenueError::MissingState("quote TokenInfo".into()))?;

        let taker_ata_base =
            get_associated_token_address_with_program_id(&user, &self.base_mint, &base_program);
        let taker_ata_quote =
            get_associated_token_address_with_program_id(&user, &self.quote_mint, &quote_program);

        // Slippage: pass `0` for `min_amount_out`. Titan handles slippage
        // upstream (the route compiler wraps multi-hop paths in a check;
        // single-hop callers can wrap their tx in their own min-out guard).
        // Quay's on-chain `swap` ix rejects with `SlippageExceeded` only when
        // this is set non-zero.
        let ix = ix::swap(
            &self.program_id,
            &self.strategy_key,
            &self.mm_key,
            &self.quotes_key,
            &self.base_mint,
            &self.quote_mint,
            &user,
            &taker_ata_base,
            &taker_ata_quote,
            &base_program,
            &quote_program,
            request.amount,
            0,
            side,
        );
        Ok(ix)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: panic on assertion failure is the desired behavior"
)]
mod tests {
    use super::*;

    /// Build a fake `Clock` sysvar account body. Fills the full 40-byte
    /// layout — the decoder must read only `slot` (0..8) and
    /// `unix_timestamp` (32..40), ignoring `epoch_start_timestamp`,
    /// `epoch`, and `leader_schedule_epoch`. We poison those middle fields
    /// with a non-zero pattern to catch any offset drift.
    fn fake_clock(slot: u64, unix_ts: i64) -> Vec<u8> {
        let mut data = vec![0u8; CLOCK_MIN_LEN];
        data[0..8].copy_from_slice(&slot.to_le_bytes());
        // Poison middle fields — decoder must NOT pick these up.
        data[8..16].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
        data[16..24].copy_from_slice(&0xFEED_FACE_FEED_FACEu64.to_le_bytes());
        data[24..32].copy_from_slice(&0xBAAD_F00D_BAAD_F00Du64.to_le_bytes());
        data[32..40].copy_from_slice(&unix_ts.to_le_bytes());
        data
    }

    #[test]
    fn decode_clock_reads_slot_and_unix_ts() {
        let data = fake_clock(123_456_789, 1_700_000_000);
        let (slot, ts) = decode_clock(&data).expect("decode should succeed");
        assert_eq!(slot, 123_456_789);
        assert_eq!(ts, 1_700_000_000);
    }

    #[test]
    fn decode_clock_rejects_short_buffers() {
        // 39 bytes — one short of the unix_timestamp tail.
        let data = vec![0u8; CLOCK_MIN_LEN - 1];
        assert!(decode_clock(&data).is_none());
    }

    #[test]
    fn decode_clock_accepts_oversized_buffers() {
        // Real sysvar accounts are >= 40 bytes; future Solana releases
        // could grow the struct. The decoder should ignore trailing bytes.
        let mut data = fake_clock(7, 11);
        data.extend_from_slice(&[0xAB; 32]);
        let (slot, ts) = decode_clock(&data).expect("decode should succeed");
        assert_eq!(slot, 7);
        assert_eq!(ts, 11);
    }

    #[test]
    fn decode_clock_handles_negative_unix_ts() {
        // Pre-1970 timestamps are nonsensical for mainnet but the field is
        // `i64` — the decoder must preserve the sign rather than coerce.
        let data = fake_clock(0, -1);
        let (_, ts) = decode_clock(&data).expect("decode should succeed");
        assert_eq!(ts, -1);
    }

    /// Build a `QuayVenue` with every halt / freeze byte cleared and no
    /// transfer-fee mints — `halts_clear()` returns true,
    /// `any_transfer_fee()` returns false, and (once state blobs are
    /// populated) `initialized()` returns true.
    fn all_active_venue() -> QuayVenue {
        QuayVenue {
            program_id: Pubkey::new_unique(),
            strategy_key: Pubkey::new_unique(),
            // Non-empty so `has_all_state()` passes — content doesn't
            // matter for the halt-gate tests since they exercise
            // `halts_clear()` / `any_transfer_fee()` directly.
            strategy_data: vec![0u8; 1],
            mm_key: Pubkey::new_unique(),
            mm_data: vec![0u8; 1],
            quotes_key: Pubkey::new_unique(),
            quotes_data: vec![0u8; 1],
            global_config_key: Pubkey::new_unique(),
            global_config_data: vec![0u8; 1],
            base_mint: Pubkey::new_unique(),
            quote_mint: Pubkey::new_unique(),
            vault_base_key: Pubkey::new_unique(),
            vault_quote_key: Pubkey::new_unique(),
            vault_base_data: Vec::new(),
            vault_quote_data: Vec::new(),
            tokens: Vec::new(),
            cfg_swap_halted: 0,
            cfg_protocol_halted: 0,
            strategy_frozen: 0,
            strategy_frozen_admin: 0,
            mm_frozen: 0,
            mm_frozen_admin: 0,
            mm_halted_admin: 0,
            current_slot: 0,
            current_unix_sec: 0,
        }
    }

    #[test]
    fn initialized_true_when_state_loaded_and_halts_clear() {
        assert!(all_active_venue().initialized());
    }

    /// One row of the halt-gate property table — see Jupiter sibling.
    type HaltCase = (&'static str, fn(&mut QuayVenue));

    /// Property: setting any one halt / freeze byte to non-zero must flip
    /// `initialized()` to false. Mirrors the Jupiter sibling test —
    /// adding a new flag forces a code change here.
    #[test]
    fn initialized_false_when_any_single_halt_set() {
        let cases: &[HaltCase] = &[
            ("cfg_swap_halted", |v| v.cfg_swap_halted = 1),
            ("cfg_protocol_halted", |v| v.cfg_protocol_halted = 1),
            ("strategy_frozen", |v| v.strategy_frozen = 1),
            ("strategy_frozen_admin", |v| v.strategy_frozen_admin = 1),
            ("mm_frozen", |v| v.mm_frozen = 1),
            ("mm_frozen_admin", |v| v.mm_frozen_admin = 1),
            ("mm_halted_admin", |v| v.mm_halted_admin = 1),
        ];
        for (name, set) in cases {
            let mut venue = all_active_venue();
            set(&mut venue);
            assert!(!venue.initialized(), "flag {name}=1 should fail initialized()");
        }
    }

    #[test]
    fn initialized_false_when_state_missing() {
        // Pre-first-update behavior: state blobs are empty so the
        // `has_all_state()` arm of the gate fails even though the warmup
        // defaults make `halts_clear()` false too. We force halts to clear
        // here so the test exercises the state-populated branch in
        // isolation.
        let mut venue = all_active_venue();
        venue.mm_data.clear();
        assert!(!venue.initialized(), "empty mm_data should fail initialized()");
    }

    #[test]
    fn initialized_false_when_transfer_fee_present() {
        let mut venue = all_active_venue();
        venue.tokens = vec![TokenInfo {
            pubkey: venue.base_mint,
            decimals: 6,
            is_token_2022: true,
            transfer_fee: Some(25),
            maximum_fee: Some(0),
        }];
        assert!(
            !venue.initialized(),
            "transfer-fee mint in tokens should fail initialized()"
        );
    }

    #[test]
    fn required_pubkeys_include_clock_sysvar() {
        // Sanity check: the index used in `update_state` (slot 8) must
        // match the position of `SYSVAR_CLOCK_ID` in the required-keys
        // vector. If anyone reorders these the off-by-one bites loudly.
        let venue = all_active_venue();
        let keys = venue.get_required_pubkeys_for_update().unwrap();
        assert_eq!(keys.len(), 9);
        assert_eq!(keys[8], SYSVAR_CLOCK_ID);
    }
}
