//! Quay pricing VM
//!
//! A tiny stack machine that interprets per-MM pricing curves. See
//! `onchain/vm/SPEC.md` for the language definition.
//!
//! `no_std` so it embeds inside Solana programs. Same crate is used by
//! off-chain simulators (routers, MM dashboards) for bit-identical pricing.
//!
//! ## Userspace design (current)
//!
//! Each Strategy carries one `userspace: &mut [u8]` region — sized at
//! `init_strategy` and resizable via `resize_userspace`. Both owner (via
//! `set_userspace`) and the running bytecode (via `StoreI64`) can write
//! to it. The VM bounds-checks every access at runtime; there is no
//! install-time validator and no `unsafe` in the dispatch loop.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod cu_estimate;
pub mod opcode;
pub mod vm;
pub mod wire;

pub use cu_estimate::{cu_estimate, swap_total_cu_estimate};
pub use opcode::{is_stateless, Op, MAX_BYTECODE_LEN, MAX_STACK_DEPTH, MAX_STEPS, MAX_TABLE_ENTRIES};
pub use vm::{
    evaluate, EvalResult, Inputs, RuntimeError, TxContext, MAX_TX_SIGNERS, TX_FLAG_FLASHLOAN,
    TX_FLAG_JITO_TIP,
};

/// VM/DSL semantic version. Bumped on breaking VM changes (opcode semantic
/// changes that affect previously-valid bytecode). Additive changes (new
/// opcodes) do not bump this version.
///
/// Stamped into `StrategyHeader.dsl_version` at `init_strategy` and
/// re-stamped on every `update_strategy_bytecode`. The swap path rejects
/// with `UnsupportedDslVersion` when `dsl_version > CURRENT_DSL_VERSION`
/// (binary-rollback guard). `0` is treated as v1-equivalent.
pub const CURRENT_DSL_VERSION: u8 = 1;
