//! Instruction builders.
//!
//! Each function returns a `solana_program::instruction::Instruction` whose
//! `data` matches the wire format the on-chain dispatcher expects. Account
//! orders mirror the handler doc-comments in `onchain/program/src/instructions/`.
//! Callers wrap the output in their own `Transaction`.

use solana_program::instruction::{AccountMeta, Instruction};
use solana_program::pubkey::Pubkey;
#[allow(deprecated)]
use solana_program::system_program;

use crate::consts::{
    AGG_N, DISC_AGG_SWAP, DISC_CLOSE_QUOTES, DISC_DEPOSIT_ASSET, DISC_EDIT_MARKET_MAKER_STATUS,
    DISC_EDIT_PLATFORM_CONFIG, DISC_EDIT_STRATEGY_FEES, DISC_EDIT_STRATEGY_STATUS,
    DISC_FREEZE_MARKET_MAKER, DISC_HALT_MARKET_MAKER, DISC_INIT_GLOBAL, DISC_INIT_MARKET_MAKER,
    DISC_INIT_QUOTES, DISC_INIT_STRATEGY, DISC_RESIZE_USERSPACE, DISC_SET_USERSPACE, DISC_SWAP,
    DISC_UPDATE_QUOTES, DISC_UPDATE_STRATEGY_BYTECODE, DISC_UPDATE_STRATEGY_QUOTES,
    DISC_WITHDRAW_ASSET, DISC_WITHDRAW_PROTOCOL_FEES, QUOTES_NUM_SLOTS,
};
use crate::pda;

fn meta_w(k: &Pubkey) -> AccountMeta {
    AccountMeta::new(*k, false)
}
fn meta_r(k: &Pubkey) -> AccountMeta {
    AccountMeta::new_readonly(*k, false)
}
fn meta_signer_w(k: &Pubkey) -> AccountMeta {
    AccountMeta::new(*k, true)
}
fn meta_signer_r(k: &Pubkey) -> AccountMeta {
    AccountMeta::new_readonly(*k, true)
}

// ────────────────────────────────────────────────────────────────────────────
// Global config
// ────────────────────────────────────────────────────────────────────────────

/// `init_global` (0x00). `payer` becomes the admin.
pub fn init_global(program_id: &Pubkey, payer: &Pubkey) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(&cfg), meta_signer_w(payer), meta_r(&system_program::ID)],
        data: vec![DISC_INIT_GLOBAL],
    }
}

/// `edit_platform_config` (0x01). `mask` selects which fields to write
/// (`CFG_MASK_*`). Unselected fields are ignored.
pub fn edit_platform_config(
    program_id: &Pubkey,
    admin: &Pubkey,
    mask: u8,
    swap_halted: u8,
    protocol_halted: u8,
    new_admin: &Pubkey,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    let mut data = vec![DISC_EDIT_PLATFORM_CONFIG, mask, swap_halted, protocol_halted];
    data.extend_from_slice(new_admin.as_ref());
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(&cfg), meta_signer_r(admin)],
        data,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Quotes
// ────────────────────────────────────────────────────────────────────────────

/// `init_quotes` (0x05). `quotes` is a fresh keypair — it co-signs creation.
pub fn init_quotes(program_id: &Pubkey, quotes: &Pubkey, owner: &Pubkey) -> Instruction {
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_signer_w(quotes),
            meta_signer_w(owner),
            meta_r(&system_program::ID),
        ],
        data: vec![DISC_INIT_QUOTES],
    }
}

/// `update_quotes` (0x04). `blob` must be exactly `QUOTES_NUM_SLOTS` entries.
pub fn update_quotes(
    program_id: &Pubkey,
    quotes: &Pubkey,
    owner: &Pubkey,
    batch_ts: u64,
    blob: &[i32],
) -> Instruction {
    assert_eq!(
        blob.len(),
        QUOTES_NUM_SLOTS,
        "blob must be exactly QUOTES_NUM_SLOTS entries"
    );
    let mut data = Vec::with_capacity(1 + 8 + blob.len() * 4);
    data.push(DISC_UPDATE_QUOTES);
    data.extend_from_slice(&batch_ts.to_le_bytes());
    for v in blob {
        data.extend_from_slice(&v.to_le_bytes());
    }
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(quotes), meta_signer_r(owner)],
        data,
    }
}

/// `close_quotes` (0x06). Reclaims rent to `owner`. `owner` is account[0]
/// (a 0-data wallet) so the `update_quotes` asm hot path trampolines this.
pub fn close_quotes(program_id: &Pubkey, quotes: &Pubkey, owner: &Pubkey) -> Instruction {
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_signer_w(owner), meta_w(quotes)],
        data: vec![DISC_CLOSE_QUOTES],
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MarketMaker
// ────────────────────────────────────────────────────────────────────────────

/// `init_market_maker` (0x07). Admin-only — admin signs + pays rent, `owner`
/// is a passive reference. The MM is born frozen; the owner unfreezes via
/// `edit_market_maker_status`.
pub fn init_market_maker(program_id: &Pubkey, admin: &Pubkey, owner: &Pubkey) -> Instruction {
    let (mm, _) = pda::market_maker_pda(program_id, owner);
    let (cfg, _) = pda::global_config_pda(program_id);
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_w(&mm),
            meta_r(&cfg),
            meta_r(owner),
            meta_signer_w(admin),
            meta_r(&system_program::ID),
        ],
        data: vec![DISC_INIT_MARKET_MAKER],
    }
}

/// `deposit_asset` (0x08). Owner-only. `asset_id == ASSET_ID_NEW` appends a
/// fresh slot (and the program creates the vault for `mint`).
#[allow(clippy::too_many_arguments)]
pub fn deposit_asset(
    program_id: &Pubkey,
    mm: &Pubkey,
    mint: &Pubkey,
    owner: &Pubkey,
    owner_ata: &Pubkey,
    token_program: &Pubkey,
    amount: u64,
    asset_id: u16,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    let (vault, _) = pda::vault_pda(program_id, mm, mint);
    let mut data = vec![DISC_DEPOSIT_ASSET];
    data.extend_from_slice(&amount.to_le_bytes());
    data.extend_from_slice(&asset_id.to_le_bytes());
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_r(&cfg),
            meta_w(mm),
            meta_w(&vault),
            meta_w(owner_ata),
            meta_signer_w(owner),
            meta_r(mint),
            meta_r(token_program),
            meta_r(&system_program::ID),
        ],
        data,
    }
}

/// `withdraw_asset` (0x09). Owner-only.
#[allow(clippy::too_many_arguments)]
pub fn withdraw_asset(
    program_id: &Pubkey,
    mm: &Pubkey,
    mint: &Pubkey,
    owner: &Pubkey,
    dest: &Pubkey,
    token_program: &Pubkey,
    amount: u64,
    asset_id: u16,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    let (vault, _) = pda::vault_pda(program_id, mm, mint);
    let mut data = vec![DISC_WITHDRAW_ASSET];
    data.extend_from_slice(&amount.to_le_bytes());
    data.extend_from_slice(&asset_id.to_le_bytes());
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_r(&cfg),
            meta_w(mm),
            meta_w(&vault),
            meta_w(dest),
            meta_signer_r(owner),
            meta_r(mint),
            meta_r(token_program),
        ],
        data,
    }
}

/// `edit_market_maker_status` (0x0A). Owner toggles `frozen`.
pub fn edit_market_maker_status(
    program_id: &Pubkey,
    mm: &Pubkey,
    owner: &Pubkey,
    frozen: u8,
) -> Instruction {
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(mm), meta_signer_r(owner)],
        data: vec![DISC_EDIT_MARKET_MAKER_STATUS, frozen],
    }
}

fn mm_admin_flag_ix(
    program_id: &Pubkey,
    disc: u8,
    mm: &Pubkey,
    admin: &Pubkey,
    mode: u8,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(mm), meta_r(&cfg), meta_signer_r(admin)],
        data: vec![disc, mode],
    }
}

/// `freeze_market_maker` (0x0B). Admin toggles `frozen_admin`.
pub fn freeze_market_maker(program_id: &Pubkey, mm: &Pubkey, admin: &Pubkey, mode: u8) -> Instruction {
    mm_admin_flag_ix(program_id, DISC_FREEZE_MARKET_MAKER, mm, admin, mode)
}

/// `halt_market_maker` (0x0C). Admin toggles `halted_admin`.
pub fn halt_market_maker(program_id: &Pubkey, mm: &Pubkey, admin: &Pubkey, mode: u8) -> Instruction {
    mm_admin_flag_ix(program_id, DISC_HALT_MARKET_MAKER, mm, admin, mode)
}

/// `withdraw_protocol_fees` (0x0D). Admin drains one asset slot's accrued fees.
pub fn withdraw_protocol_fees(
    program_id: &Pubkey,
    mm: &Pubkey,
    mint: &Pubkey,
    admin: &Pubkey,
    dest: &Pubkey,
    token_program: &Pubkey,
    asset_id: u16,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    let (vault, _) = pda::vault_pda(program_id, mm, mint);
    let mut data = vec![DISC_WITHDRAW_PROTOCOL_FEES];
    data.extend_from_slice(&asset_id.to_le_bytes());
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_r(&cfg),
            meta_w(mm),
            meta_w(&vault),
            meta_w(dest),
            meta_signer_r(admin),
            meta_r(mint),
            meta_r(token_program),
        ],
        data,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Strategy
// ────────────────────────────────────────────────────────────────────────────

/// Wire-prefix length for `init_strategy`: id(1) + base_id(2) + quote_id(2)
/// + protocol_fee_bps(2) + bytecode_len(4) + userspace_len(4).
const INIT_STRATEGY_PREFIX: usize = 1 + 2 + 2 + 2 + 4 + 4;

/// `init_strategy` (0x10). Admin creates the strategy for `owner`; born frozen.
/// `quotes_account` optionally binds a quotes feed at creation.
#[allow(clippy::too_many_arguments)]
pub fn init_strategy(
    program_id: &Pubkey,
    admin: &Pubkey,
    owner: &Pubkey,
    mm: &Pubkey,
    id: u8,
    base_id: u16,
    quote_id: u16,
    protocol_fee_bps: u16,
    bytecode: &[u8],
    userspace_len: u32,
    quotes_account: Option<&Pubkey>,
) -> Instruction {
    let (strategy, _) = pda::strategy_pda(program_id, owner, id);
    let (cfg, _) = pda::global_config_pda(program_id);
    let mut data = Vec::with_capacity(1 + INIT_STRATEGY_PREFIX + bytecode.len());
    data.push(DISC_INIT_STRATEGY);
    data.push(id);
    data.extend_from_slice(&base_id.to_le_bytes());
    data.extend_from_slice(&quote_id.to_le_bytes());
    data.extend_from_slice(&protocol_fee_bps.to_le_bytes());
    data.extend_from_slice(&(bytecode.len() as u32).to_le_bytes());
    data.extend_from_slice(&userspace_len.to_le_bytes());
    data.extend_from_slice(bytecode);
    let mut accounts = vec![
        meta_w(&strategy),
        meta_r(mm),
        meta_r(&cfg),
        meta_r(owner),
        meta_signer_w(admin),
        meta_r(&system_program::ID),
    ];
    if let Some(q) = quotes_account {
        accounts.push(meta_r(q));
    }
    Instruction {
        program_id: *program_id,
        accounts,
        data,
    }
}

/// `update_strategy_bytecode` (0x11). Owner-only; the account is resized as
/// the bytecode length changes.
pub fn update_strategy_bytecode(
    program_id: &Pubkey,
    strategy: &Pubkey,
    owner: &Pubkey,
    bytecode: &[u8],
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    let mut data = Vec::with_capacity(1 + 4 + bytecode.len());
    data.push(DISC_UPDATE_STRATEGY_BYTECODE);
    data.extend_from_slice(&(bytecode.len() as u32).to_le_bytes());
    data.extend_from_slice(bytecode);
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_w(strategy),
            meta_r(&cfg),
            meta_signer_w(owner),
            meta_r(&system_program::ID),
        ],
        data,
    }
}

/// `resize_userspace` (0x12). Owner-only. Grows by ≤ 10 KiB per call.
pub fn resize_userspace(
    program_id: &Pubkey,
    strategy: &Pubkey,
    owner: &Pubkey,
    new_userspace_len: u32,
) -> Instruction {
    let mut data = vec![DISC_RESIZE_USERSPACE];
    data.extend_from_slice(&new_userspace_len.to_le_bytes());
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_w(strategy),
            meta_signer_w(owner),
            meta_r(&system_program::ID),
        ],
        data,
    }
}

/// `set_userspace` (0x13). Owner-only partial overwrite at `offset`.
pub fn set_userspace(
    program_id: &Pubkey,
    strategy: &Pubkey,
    owner: &Pubkey,
    offset: u32,
    bytes: &[u8],
) -> Instruction {
    let mut data = Vec::with_capacity(1 + 4 + bytes.len());
    data.push(DISC_SET_USERSPACE);
    data.extend_from_slice(&offset.to_le_bytes());
    data.extend_from_slice(bytes);
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(strategy), meta_signer_r(owner)],
        data,
    }
}

/// `update_strategy_quotes` (0x14). Owner rebinds the quotes account.
pub fn update_strategy_quotes(
    program_id: &Pubkey,
    strategy: &Pubkey,
    quotes_account: &Pubkey,
    owner: &Pubkey,
) -> Instruction {
    Instruction {
        program_id: *program_id,
        accounts: vec![
            meta_w(strategy),
            meta_r(quotes_account),
            meta_signer_r(owner),
        ],
        data: vec![DISC_UPDATE_STRATEGY_QUOTES],
    }
}

/// `edit_strategy_status` (0x15). `signer` is the owner (toggles `frozen`) or
/// the admin (toggles `frozen_admin`).
pub fn edit_strategy_status(
    program_id: &Pubkey,
    strategy: &Pubkey,
    signer: &Pubkey,
    mode: u8,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(strategy), meta_r(&cfg), meta_signer_r(signer)],
        data: vec![DISC_EDIT_STRATEGY_STATUS, mode],
    }
}

/// `edit_strategy_fees` (0x16). Admin sets `protocol_fee_bps`. A fee
/// **increase** auto-sets `mm.frozen = 1` so the owner must re-acknowledge
/// before swaps pay the larger cut; the program writes to `mm` for this,
/// so `mm` is now a writable account in the ix.
pub fn edit_strategy_fees(
    program_id: &Pubkey,
    strategy: &Pubkey,
    mm: &Pubkey,
    admin: &Pubkey,
    protocol_fee_bps: u16,
) -> Instruction {
    let (cfg, _) = pda::global_config_pda(program_id);
    let mut data = vec![DISC_EDIT_STRATEGY_FEES];
    data.extend_from_slice(&protocol_fee_bps.to_le_bytes());
    Instruction {
        program_id: *program_id,
        accounts: vec![meta_w(strategy), meta_w(mm), meta_r(&cfg), meta_signer_r(admin)],
        data,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Swap
// ────────────────────────────────────────────────────────────────────────────

/// `Sysvar1nstructions1111111111111111111111111` — required on every
/// `swap` / `agg_swap` at the fixed position right after the positional
/// accounts, before the token program(s); the builders below place it.
pub const INSTRUCTIONS_SYSVAR_ID: Pubkey = Pubkey::new_from_array([
    0x06, 0xa7, 0xd5, 0x17, 0x18, 0x7b, 0xd1, 0x66, 0x35, 0xda, 0xd4, 0x04, 0x55, 0xfd, 0xc2,
    0xc0, 0xc1, 0x24, 0xc6, 0x8f, 0x21, 0x56, 0x75, 0xa5, 0xdb, 0xba, 0xcb, 0x5f, 0x08, 0x00,
    0x00, 0x00,
]);

/// Append `base`/`quote` token programs, deduped, to an account list.
fn push_token_programs(accounts: &mut Vec<AccountMeta>, base: &Pubkey, quote: &Pubkey) {
    accounts.push(meta_r(base));
    if quote != base {
        accounts.push(meta_r(quote));
    }
}

/// `swap` (0x20). `side`: `0 = SELL_BASE`, `1 = BUY_BASE`. Vaults are derived
/// from `mm`; the taker's token accounts are supplied explicitly.
#[allow(clippy::too_many_arguments)]
pub fn swap(
    program_id: &Pubkey,
    strategy: &Pubkey,
    mm: &Pubkey,
    quotes: &Pubkey,
    base_mint: &Pubkey,
    quote_mint: &Pubkey,
    taker: &Pubkey,
    taker_ata_base: &Pubkey,
    taker_ata_quote: &Pubkey,
    base_token_program: &Pubkey,
    quote_token_program: &Pubkey,
    amount_in: u64,
    min_amount_out: u64,
    side: u8,
) -> Instruction {
    assert!(side <= 1, "side must be 0 (sell base) or 1 (buy base)");
    let (cfg, _) = pda::global_config_pda(program_id);
    let (vault_base, _) = pda::vault_pda(program_id, mm, base_mint);
    let (vault_quote, _) = pda::vault_pda(program_id, mm, quote_mint);

    // Resolve IN/OUT by side.
    let (mint_in, mint_out, vault_in, vault_out, ata_in, ata_out) = if side == 0 {
        (base_mint, quote_mint, &vault_base, &vault_quote, taker_ata_base, taker_ata_quote)
    } else {
        (quote_mint, base_mint, &vault_quote, &vault_base, taker_ata_quote, taker_ata_base)
    };

    let mut data = Vec::with_capacity(1 + 8 + 8 + 1);
    data.push(DISC_SWAP);
    data.extend_from_slice(&amount_in.to_le_bytes());
    data.extend_from_slice(&min_amount_out.to_le_bytes());
    data.push(side);

    let mut accounts = vec![
        meta_r(&cfg),
        meta_w(strategy),
        meta_w(mm),
        meta_r(quotes),
        meta_w(vault_in),
        meta_w(vault_out),
        meta_w(ata_in),
        meta_w(ata_out),
        meta_signer_w(taker),
        meta_r(mint_in),
        meta_r(mint_out),
    ];
    accounts.push(meta_r(&INSTRUCTIONS_SYSVAR_ID));
    push_token_programs(&mut accounts, base_token_program, quote_token_program);
    Instruction {
        program_id: *program_id,
        accounts,
        data,
    }
}

/// One `agg_swap` candidate. Vaults are derived from `mm`.
#[derive(Debug, Clone, Copy)]
pub struct AggCandidate<'a> {
    pub strategy: &'a Pubkey,
    pub mm: &'a Pubkey,
    pub quotes: &'a Pubkey,
}

/// `agg_swap` (0x21). Best-of-N across `AGG_N` candidate strategies on the
/// same `(base_mint, quote_mint)` pair.
#[allow(clippy::too_many_arguments)]
pub fn agg_swap(
    program_id: &Pubkey,
    candidates: &[AggCandidate<'_>; AGG_N],
    base_mint: &Pubkey,
    quote_mint: &Pubkey,
    taker: &Pubkey,
    taker_ata_base: &Pubkey,
    taker_ata_quote: &Pubkey,
    base_token_program: &Pubkey,
    quote_token_program: &Pubkey,
    amount_in: u64,
    min_amount_out: u64,
    side: u8,
) -> Instruction {
    assert!(side <= 1, "side must be 0 (sell base) or 1 (buy base)");
    let (cfg, _) = pda::global_config_pda(program_id);

    let mut data = Vec::with_capacity(1 + 8 + 8 + 1);
    data.push(DISC_AGG_SWAP);
    data.extend_from_slice(&amount_in.to_le_bytes());
    data.extend_from_slice(&min_amount_out.to_le_bytes());
    data.push(side);

    let mut accounts = Vec::with_capacity(6 + 5 * AGG_N + 2);
    accounts.push(meta_r(&cfg));
    accounts.push(meta_signer_w(taker));
    accounts.push(meta_w(taker_ata_base));
    accounts.push(meta_w(taker_ata_quote));
    accounts.push(meta_r(base_mint));
    accounts.push(meta_r(quote_mint));
    for c in candidates {
        accounts.push(meta_w(c.strategy));
    }
    for c in candidates {
        accounts.push(meta_w(c.mm));
    }
    for c in candidates {
        accounts.push(meta_r(c.quotes));
    }
    for c in candidates {
        let (vb, _) = pda::vault_pda(program_id, c.mm, base_mint);
        accounts.push(meta_w(&vb));
    }
    for c in candidates {
        let (vq, _) = pda::vault_pda(program_id, c.mm, quote_mint);
        accounts.push(meta_w(&vq));
    }
    accounts.push(meta_r(&INSTRUCTIONS_SYSVAR_ID));
    push_token_programs(&mut accounts, base_token_program, quote_token_program);
    Instruction {
        program_id: *program_id,
        accounts,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid() -> Pubkey {
        Pubkey::new_from_array([7u8; 32])
    }

    #[test]
    fn swap_wire_layout() {
        let k = Pubkey::new_unique();
        let ix = swap(
            &pid(), &k, &k, &k, &k, &k, &k, &k, &k, &k, &k, 1_000, 950, 0,
        );
        assert_eq!(ix.data.len(), 1 + 8 + 8 + 1);
        assert_eq!(ix.data[0], DISC_SWAP);
        assert_eq!(&ix.data[1..9], &1_000u64.to_le_bytes());
        assert_eq!(&ix.data[9..17], &950u64.to_le_bytes());
        assert_eq!(ix.data[17], 0);
        // cfg, strategy, mm, quotes, vault_in, vault_out, ata_in, ata_out,
        // taker, mint_in, mint_out, instructions sysvar, token_program
        // (base==quote → deduped to 1).
        assert_eq!(ix.accounts.len(), 13);
        assert_eq!(ix.accounts[11].pubkey, INSTRUCTIONS_SYSVAR_ID);
        assert!(ix.accounts[8].is_signer);
    }

    #[test]
    fn init_strategy_wire_prefix() {
        let k = Pubkey::new_unique();
        let ix = init_strategy(&pid(), &k, &k, &k, 3, 0, 1, 250, &[0xAA, 0xBB], 64, None);
        assert_eq!(ix.data[0], DISC_INIT_STRATEGY);
        assert_eq!(ix.data[1], 3); // id
        assert_eq!(ix.data.len(), 1 + INIT_STRATEGY_PREFIX + 2);
        assert_eq!(ix.accounts.len(), 6); // no quotes account
    }

    #[test]
    fn agg_swap_account_count() {
        let k = Pubkey::new_unique();
        let c = AggCandidate { strategy: &k, mm: &k, quotes: &k };
        let ix = agg_swap(
            &pid(), &[c; AGG_N], &k, &k, &k, &k, &k, &k, &k, 1_000, 0, 1,
        );
        // 6 fixed + 5*AGG_N + instructions sysvar + 1 token program (deduped).
        assert_eq!(ix.accounts.len(), 6 + 5 * AGG_N + 2);
        assert_eq!(ix.accounts[6 + 5 * AGG_N].pubkey, INSTRUCTIONS_SYSVAR_ID);
    }
}
