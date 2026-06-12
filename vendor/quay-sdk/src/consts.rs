//! Wire constants mirrored from `onchain/program/src/state.rs` and the 1-byte
//! instruction discriminators dispatched in `onchain/program/src/lib.rs`.

// ── Account discriminators ───────────────────────────────────────────────────
pub const GLOBAL_CONFIG_DISCRIMINATOR: u8 = 0x01;
pub const QUOTES_DISCRIMINATOR: u8 = 0x02;
pub const STRATEGY_DISCRIMINATOR: u8 = 0x03;
pub const MARKET_MAKER_DISCRIMINATOR: u8 = 0x04;

// ── Instruction discriminators ───────────────────────────────────────────────
pub const DISC_INIT_GLOBAL: u8 = 0x00;
pub const DISC_EDIT_PLATFORM_CONFIG: u8 = 0x01;
pub const DISC_UPDATE_QUOTES: u8 = 0x04;
pub const DISC_INIT_QUOTES: u8 = 0x05;
pub const DISC_CLOSE_QUOTES: u8 = 0x06;
pub const DISC_INIT_MARKET_MAKER: u8 = 0x07;
pub const DISC_DEPOSIT_ASSET: u8 = 0x08;
pub const DISC_WITHDRAW_ASSET: u8 = 0x09;
pub const DISC_EDIT_MARKET_MAKER_STATUS: u8 = 0x0A;
pub const DISC_FREEZE_MARKET_MAKER: u8 = 0x0B;
pub const DISC_HALT_MARKET_MAKER: u8 = 0x0C;
pub const DISC_WITHDRAW_PROTOCOL_FEES: u8 = 0x0D;
pub const DISC_INIT_STRATEGY: u8 = 0x10;
pub const DISC_UPDATE_STRATEGY_BYTECODE: u8 = 0x11;
pub const DISC_RESIZE_USERSPACE: u8 = 0x12;
pub const DISC_SET_USERSPACE: u8 = 0x13;
pub const DISC_UPDATE_STRATEGY_QUOTES: u8 = 0x14;
pub const DISC_EDIT_STRATEGY_STATUS: u8 = 0x15;
pub const DISC_EDIT_STRATEGY_FEES: u8 = 0x16;
pub const DISC_SWAP: u8 = 0x20;
pub const DISC_AGG_SWAP: u8 = 0x21;

/// Candidate count for `agg_swap`. Mirrors `quay_program`'s `AGG_N`.
pub const AGG_N: usize = 5;

// ── Account layout sizes ─────────────────────────────────────────────────────
pub const GLOBAL_CONFIG_SIZE: usize = 128;
pub const QUOTES_HEADER_SIZE: usize = 64;
pub const STRATEGY_HEADER_SIZE: usize = 192;
pub const MARKET_MAKER_HEADER_SIZE: usize = 64;
pub const ASSET_ENTRY_SIZE: usize = 48;
pub const SPL_TOKEN_ACCOUNT_LEN: usize = 165;
pub const SPL_TOKEN_MINT_LEN: usize = 82;

// ── Quotes geometry ──────────────────────────────────────────────────────────
pub const QUOTES_NUM_SLOTS: usize = 241;
pub const QUOTES_BLOB_LEN: usize = QUOTES_NUM_SLOTS * 4;
pub const QUOTES_ACCOUNT_SIZE: usize = QUOTES_HEADER_SIZE + QUOTES_BLOB_LEN;

// ── Side enum (u8) ───────────────────────────────────────────────────────────
pub const SIDE_SELL_BASE: u8 = 0;
pub const SIDE_BUY_BASE: u8 = 1;

/// `asset_id` sentinel for `deposit_asset`'s "append a new slot" path.
pub const ASSET_ID_NEW: u16 = 0xFFFF;

// ── Fee / size math ──────────────────────────────────────────────────────────
pub const FEE_BPS_DENOM: u64 = 10_000;
/// Maximum strategy userspace region, in bytes (1 MiB).
pub const MAX_USERSPACE_LEN: u32 = 1 << 20;

// ── Swap pricing scale ───────────────────────────────────────────────────────
pub const PRICE_SCALE_BITS: u32 = 24;
pub const PRICE_SCALE: i128 = 1i128 << PRICE_SCALE_BITS;

// ── edit_platform_config field mask bits ─────────────────────────────────────
pub const CFG_MASK_SWAP_HALTED: u8 = 0b001;
pub const CFG_MASK_PROTOCOL_HALTED: u8 = 0b010;
pub const CFG_MASK_ADMIN: u8 = 0b100;

// ── Per-tx hint ──────────────────────────────────────────────────────────────
//
// Routers SHOULD set this on every swap tx. Without it the cost-tracker bills
// 16,384 CU per tx for the default 64 MiB loaded-data limit; one page (32 KiB)
// bills 8 CU.
pub const SWAP_LOADED_ACCOUNTS_DATA_SIZE_LIMIT: u32 = 32_768;
