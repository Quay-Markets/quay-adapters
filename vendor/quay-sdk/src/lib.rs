//! `quay-sdk` — off-chain Rust client for the Quay program.
//!
//! Three layers:
//!
//! 1. **Wire layer** (`consts`, `pda`, `state`, `ix`) — discriminators, PDA
//!    derivation, zero-copy account decoders, and instruction builders that
//!    mirror `quay-program`'s on-chain wire format byte-for-byte.
//! 2. **DSL layer** (`dsl`) — `BytecodeBuilder` for emitting curve bytecode.
//! 3. **Simulation layer** (`simulate`) — bit-identical off-chain `swap`
//!    simulation against the pricing VM.
//!
//! Account model: a per-owner `MarketMaker`
//! holds an asset table of all balances; each `Strategy` carries pricing
//! bytecode + userspace and references two asset slots; `Quotes` accounts feed
//! the VM.

#![allow(clippy::result_large_err)]

pub mod consts;
pub mod dsl;
pub mod error;
pub mod ix;
pub mod pda;
pub mod simulate;
pub mod state;

// Re-exports for convenience.
pub use error::{ClientError, Result};
pub use quay_vm::{
    cu_estimate, is_stateless, swap_total_cu_estimate, EvalResult, Op, RuntimeError, TxContext,
    TX_FLAG_FLASHLOAN, TX_FLAG_JITO_TIP,
};
