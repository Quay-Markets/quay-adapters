//! Titan construction & boundary tests. Verifies the venue correctly:
//!
//! - deserializes a Strategy account (`from_account`),
//! - loads the dependent state (`update_state`),
//! - produces valid token info (`get_token_info` / `tradable_mints` /
//!   `decimals`),
//! - quotes at both `bounds()` boundaries,
//! - performs **no heap allocation** in `quote()` — the quoting hot path must
//!   be real-time (Titan's hard requirement, checked via `assert_no_alloc`),
//! - refuses **stateful** curves — the VM's aggregator-routing contract
//!   (`quay_vm::is_stateless`): a router caches the quote, so a curve that
//!   writes `userspace` mid-swap drifts the on-chain fill away from the quote.
//!
//! Driven by the same litesvm-generated fixtures as `tests/parity.rs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]

use std::collections::HashMap;
use std::str::FromStr;

use assert_no_alloc::{assert_no_alloc, AllocDisabler};
use async_trait::async_trait;
use quay_aggregator_titan::QuayVenue;
use solana_account::Account;
use solana_pubkey::Pubkey;
use titan_integration_template::account_caching::{AccountCacheError, AccountsCache};
use titan_integration_template::trading_venue::{
    FromAccount, QuoteRequest, SwapType, TradingVenue,
};

/// `assert_no_alloc` only enforces when its `AllocDisabler` is the global
/// allocator. In a guarded region it aborts the process on any heap
/// allocation; outside one it forwards to the system allocator.
#[global_allocator]
static A: AllocDisabler = AllocDisabler;

const SIDE_SELL_BASE: u8 = 0;

/// Fixed-map `AccountsCache` — the same stand-in `parity.rs` uses for the
/// router's RPC-backed cache.
struct MapCache(HashMap<Pubkey, Account>);

#[async_trait]
impl AccountsCache for MapCache {
    async fn get_account(&self, pubkey: &Pubkey) -> Result<Option<Account>, AccountCacheError> {
        Ok(self.0.get(pubkey).cloned())
    }

    async fn get_accounts(
        &self,
        pubkeys: &[Pubkey],
    ) -> Result<Vec<Option<Account>>, AccountCacheError> {
        Ok(pubkeys.iter().map(|k| self.0.get(k).cloned()).collect())
    }
}

struct Fixture {
    strategy: Pubkey,
    base_mint: Pubkey,
    quote_mint: Pubkey,
    accounts: HashMap<Pubkey, Account>,
    /// `(side, amount_in, Some(actual_out) | None-if-on-chain-failed)`
    swaps: Vec<(u8, u64, Option<u64>)>,
}

fn b64_decode(s: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .expect("valid base64 in fixture")
}

fn load_fixture(name: &str) -> Fixture {
    let path = format!("{}/../fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {path}: {e} — regenerate with the litesvm suite"));
    let v: serde_json::Value = serde_json::from_str(&raw).expect("valid fixture JSON");

    let pk = |field: &str| Pubkey::from_str(v[field].as_str().expect(field)).expect(field);
    let accounts = v["accounts"]
        .as_array()
        .expect("accounts array")
        .iter()
        .map(|a| {
            let key = Pubkey::from_str(a["pubkey"].as_str().expect("pubkey")).expect("pubkey");
            let account = Account {
                lamports: a["lamports"].as_u64().expect("lamports"),
                data: b64_decode(a["data_b64"].as_str().expect("data_b64")),
                owner: Pubkey::from_str(a["owner"].as_str().expect("owner")).expect("owner"),
                executable: false,
                rent_epoch: 0,
            };
            (key, account)
        })
        .collect();
    let swaps = v["swaps"]
        .as_array()
        .expect("swaps array")
        .iter()
        .map(|s| {
            (
                s["side"].as_u64().expect("side") as u8,
                s["amount_in"].as_u64().expect("amount_in"),
                s["out"].as_u64(),
            )
        })
        .collect();

    Fixture {
        strategy: pk("strategy"),
        base_mint: pk("base_mint"),
        quote_mint: pk("quote_mint"),
        accounts,
        swaps,
    }
}

/// Construct + `update_state` a venue from a fixture, ready to quote.
async fn build_venue(f: &Fixture) -> QuayVenue {
    let strategy_account = f
        .accounts
        .get(&f.strategy)
        .expect("fixture missing strategy account")
        .clone();
    let mut venue = QuayVenue::from_account(&f.strategy, &strategy_account).expect("from_account");
    let cache = MapCache(f.accounts.clone());
    venue.update_state(&cache).await.expect("update_state");
    venue
}

/// The smallest and largest sweep amounts the program actually filled — the
/// quote-able boundaries for this fixture.
fn filled_amount_bounds(f: &Fixture, side: u8) -> (u64, u64) {
    let mut filled: Vec<u64> = f
        .swaps
        .iter()
        .filter(|(s, _, out)| *s == side && out.is_some())
        .map(|(_, amt, _)| *amt)
        .collect();
    filled.sort_unstable();
    (
        *filled.first().expect("at least one filled swap"),
        *filled.last().expect("at least one filled swap"),
    )
}

#[tokio::test]
async fn construction_loads_state_and_token_info() {
    let f = load_fixture("flat");
    let venue = build_venue(&f).await;

    assert!(venue.initialized(), "venue uninitialized after update_state");
    assert_eq!(venue.market_id(), f.strategy);

    // Two `TokenInfo` entries — base + quote — with sane decimals.
    let tokens = venue.get_token_info();
    assert_eq!(tokens.len(), 2, "expected base + quote token info");

    let mints = venue.tradable_mints().expect("tradable_mints");
    assert!(mints.contains(&f.base_mint) && mints.contains(&f.quote_mint));

    let decimals = venue.decimals().expect("decimals");
    assert_eq!(decimals.len(), 2);
    assert!(decimals.iter().all(|d| (0..=18).contains(d)), "decimals in range");
}

#[tokio::test]
async fn quotes_at_both_boundaries() {
    let f = load_fixture("flat");
    let venue = build_venue(&f).await;

    // `bounds(i, j)` binary-searches the admissible input range for the
    // `(token_i -> token_j)` direction. get_token(0)=base, get_token(1)=quote.
    let (sell_lo, sell_hi) = venue.bounds(0, 1).expect("bounds sell-base");
    assert!(sell_lo <= sell_hi, "sell-base lower bound exceeds upper");

    let (buy_lo, buy_hi) = venue.bounds(1, 0).expect("bounds buy-base");
    assert!(buy_lo <= buy_hi, "buy-base lower bound exceeds upper");

    // Both boundaries must produce a real, non-empty quote.
    for (amount, side) in [(sell_lo, SIDE_SELL_BASE), (sell_hi, SIDE_SELL_BASE)] {
        let (input_mint, output_mint) = (f.base_mint, f.quote_mint);
        let _ = side;
        let q = venue
            .quote(QuoteRequest { input_mint, output_mint, amount, swap_type: SwapType::ExactIn })
            .expect("boundary quote should succeed");
        assert!(q.expected_output > 0, "boundary quote produced zero output");
    }
}

/// Titan's real-time requirement: `quote()` must not touch the heap on the
/// success path. We wrap a known-filled mid-range swap from each fixture —
/// including the fee-bearing ones, which run the opposite-side simulation.
#[tokio::test]
async fn quote_does_not_allocate() {
    for name in ["flat", "spread_fee", "exact_fee", "one_sided"] {
        let f = load_fixture(name);
        let venue = build_venue(&f).await;

        // First swap the program actually filled — guaranteed `Ok`, so we
        // exercise the hot path, not the (string-allocating) error path.
        let (side, amount, _) = f
            .swaps
            .iter()
            .find(|(_, _, out)| out.is_some())
            .copied()
            .expect("fixture has a filled swap");
        let (input_mint, output_mint) = if side == SIDE_SELL_BASE {
            (f.base_mint, f.quote_mint)
        } else {
            (f.quote_mint, f.base_mint)
        };
        let req = QuoteRequest { input_mint, output_mint, amount, swap_type: SwapType::ExactIn };

        // Aborts the process if `quote()` allocates.
        let q = assert_no_alloc(|| venue.quote(req))
            .unwrap_or_else(|e| panic!("fixture {name}: filled swap quote errored: {e}"));
        assert_eq!(q.amount, amount, "fixture {name}: consumed-amount mismatch");

        // Sanity: the boundaries we'd hand the no-alloc path are real.
        let (lo, hi) = filled_amount_bounds(&f, side);
        assert!(lo <= hi);
    }
}

/// The aggregator-routing contract: a stateful curve must be refused. We take
/// the `flat` fixture's (stateless) strategy and splice a `StoreI64` opcode
/// into its bytecode — the venue must then report uninitialized and refuse to
/// quote, exactly as it does for halts and transfer-fee mints.
#[tokio::test]
async fn refuses_stateful_strategy() {
    // StrategyHeader layout: `bytecode_len` is a u32 LE at offset 144,
    // `userspace_len` at 148. The `flat` fixture has userspace_len == 0, so the
    // bytecode is the account's tail — appending an opcode keeps the layout
    // self-consistent.
    const BYTECODE_LEN_OFF: usize = 144;
    const OP_STORE_I64: u8 = 0x48; // `StoreI64` — the only userspace-mutating op.

    let mut f = load_fixture("flat");
    let mut strat = f.accounts.get(&f.strategy).expect("strategy").clone();

    let old_len = u32::from_le_bytes(
        strat.data[BYTECODE_LEN_OFF..BYTECODE_LEN_OFF + 4]
            .try_into()
            .unwrap(),
    );
    // `StoreI64` is opcode + u32 immediate = 5 bytes; append it and grow the
    // declared bytecode length to match.
    strat.data.extend_from_slice(&[OP_STORE_I64, 0, 0, 0, 0]);
    strat.data[BYTECODE_LEN_OFF..BYTECODE_LEN_OFF + 4]
        .copy_from_slice(&(old_len + 5).to_le_bytes());
    f.accounts.insert(f.strategy, strat.clone());

    // Construction already sees the stateful bytecode.
    let mut venue = QuayVenue::from_account(&f.strategy, &strat).expect("from_account");
    venue
        .update_state(&MapCache(f.accounts.clone()))
        .await
        .expect("update_state");

    assert!(
        !venue.initialized(),
        "stateful strategy must report uninitialized"
    );
    let (side, amount, _) = f
        .swaps
        .iter()
        .find(|(_, _, out)| out.is_some())
        .copied()
        .expect("filled swap");
    let (input_mint, output_mint) = if side == SIDE_SELL_BASE {
        (f.base_mint, f.quote_mint)
    } else {
        (f.quote_mint, f.base_mint)
    };
    let err = venue
        .quote(QuoteRequest { input_mint, output_mint, amount, swap_type: SwapType::ExactIn })
        .expect_err("stateful strategy must refuse to quote");
    let _ = err; // surface is `NotInitialized`; we only require a refusal.
}
