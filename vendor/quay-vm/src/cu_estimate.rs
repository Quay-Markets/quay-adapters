//! Static CU (compute-unit) estimator for Quay VM bytecode.
//!
//! Routers call [`cu_estimate`] before submitting a swap tx to set
//! `compute_unit_limit`. The walker is conservative — it returns an
//! **upper bound** on the CU `evaluate` will consume on the given bytecode.
//! Over-estimates are fine; under-estimates are dangerous (the router
//! under-budgets and the swap ix aborts mid-execution).
//!
//! ## Conservatism
//!
//! - Branches are walked once each linearly. Real execution takes only one
//!   arm of each `IF*`, so the static walk is an upper bound.
//! - Unknown opcodes / truncated immediates return a high penalty
//!   (50_000 CU) instead of erroring.
//! - All arithmetic is saturating; the function never panics.

use crate::opcode::Op;

/// Per-opcode CU cost (excluding the per-op dispatch overhead added by the
/// walker).
fn opcode_cost(op: Op) -> u64 {
    match op {
        // Scalar loads from Inputs.
        Op::LoadInv
        | Op::LoadSize
        | Op::LoadSide
        | Op::LoadNowSlot
        | Op::LoadVaultBase
        | Op::LoadVaultQuote
        | Op::LoadInvBase
        | Op::LoadInvQuote
        | Op::LoadNowUnixSec
        | Op::LoadBaseDecimals
        | Op::LoadQuoteDecimals
        | Op::LoadArenaTimestampSec
        | Op::LoadLastUpdateSlot
        | Op::LoadIxDepth
        | Op::LoadTxFlags => 5,

        // 32-byte immediate compare against the entrypoint program id.
        Op::EntrypointIs => 12,
        // Linear scan over the host-bounded signer set (≤ MAX_TX_SIGNERS
        // entries, 32-byte compare each) — conservative worst case.
        Op::IsSignedBy => 60,

        // LoadConst reads an 8-byte LE immediate.
        Op::LoadConst => 6,

        // Arena slot load — bounds-check + indexed read.
        Op::LoadArenaQuote | Op::LoadArenaQuoteU => 7,

        // Userspace byte/word reads — bounds check + N-byte read + sign-extend.
        // Costs trend with width since LE decode is the dominant work.
        Op::LoadI8 | Op::LoadU8 => 8,
        Op::LoadI16 | Op::LoadU16 => 9,
        Op::LoadI32 => 10,
        Op::LoadI64 => 12,

        // Userspace i64 store — bounds-check + i128→i64 cast + 8-byte write.
        Op::StoreI64 => 14,

        // Stack manipulation.
        Op::Dup | Op::Swap | Op::Drop | Op::Pick => 4,

        // Arithmetic on i128.
        Op::Add | Op::Sub | Op::Mul | Op::Div => 8,
        Op::DivCeil => 10,
        Op::Mod => 8,
        Op::Muldiv => 200,
        Op::Neg | Op::Abs => 6,

        // Fixed-point / Q-format helpers.
        // MulShr / MulQ48 / MuladdQ48 do one mul_i256 (~60 CU) + arithmetic
        // right-shift (~10 CU) — significantly cheaper than Muldiv (200 CU).
        Op::MulShr | Op::MulQ48 => 80,
        Op::MuladdQ48 => 85,
        // DivQ48 uses the full i256 divide path; identical cost to Muldiv.
        Op::DivQ48 => 200,

        // Bitwise.
        Op::And | Op::Or | Op::Xor | Op::Not => 5,
        Op::Shl | Op::Shr => 7,

        // Min/Max/Clamp.
        Op::Min | Op::Max => 6,
        Op::Clamp => 9,

        // Comparisons + select.
        Op::CmpLt | Op::CmpGt | Op::CmpLe | Op::CmpGe | Op::CmpEq => 5,
        Op::Select => 7,

        // Sqrt.
        Op::Sqrt => 60,

        // Lookups — table header read (8 B) + ascending-check + walk.
        // Conservative cap based on MAX_TABLE_ENTRIES.
        Op::TierLookup => 80,
        Op::TierLookupDyn => 85,
        Op::LerpLookup => 110,
        // TierLookup2 walks 24-B rows (vs 16-B). Slightly more memory work
        // + an extra push; otherwise same shape as TierLookup.
        Op::TierLookup2 => 90,

        // Branches.
        Op::IfLt | Op::IfGt | Op::IfLe | Op::IfGe | Op::IfEq => 7,
        Op::Jmp => 4,

        // Terminators.
        Op::Halt => 5,
        Op::Reject => 5,

        // Assertion: pop + zero-compare + (conditional) early-return.
        Op::AssertNonzero => 6,
    }
}

const ENTRY_OVERHEAD: u64 = 100;
const PER_OP_DISPATCH: u64 = 3;
const INVALID_BYTECODE_PENALTY: u64 = 50_000;

/// Estimate an upper bound on the CU `evaluate` will consume executing
/// `bytecode`. Never panics. Returns a high penalty for invalid bytecode.
pub fn cu_estimate(bytecode: &[u8]) -> u64 {
    let mut total: u64 = ENTRY_OVERHEAD;
    let mut pc = 0usize;
    while pc < bytecode.len() {
        let Some(op) = Op::from_byte(bytecode[pc]) else {
            return total.saturating_add(INVALID_BYTECODE_PENALTY);
        };
        total = total
            .saturating_add(opcode_cost(op))
            .saturating_add(PER_OP_DISPATCH);
        let len = op.instr_len();
        pc = pc.saturating_add(len);
        if pc > bytecode.len() {
            return total.saturating_add(INVALID_BYTECODE_PENALTY);
        }
    }
    total
}

// --- Wire-format / total-swap estimate ---

const SWAP_OVERHEAD: u64 = 8_000;
const SPL_TRANSFER_CPI: u64 = 5_000;
const TWO_TRANSFERS: u64 = SPL_TRANSFER_CPI * 2;
const CLOCK_SYSCALL: u64 = 100;
const VAULT_AUTH_DERIVE: u64 = 1_500;
/// Instructions-sysvar parse on every swap (tx-context gather). Measured
/// in litesvm: ~1.4k CU on a single-ix tx, ~4k on a 13-ix / ~37-meta
/// router-sized tx — budget the fat case.
const TX_CONTEXT_GATHER: u64 = 4_000;

/// Estimate total CU for a full swap ix (VM + non-VM costs). Routers should
/// pass this to `ComputeBudget::set_compute_unit_limit`.
pub fn swap_total_cu_estimate(bytecode: &[u8]) -> u64 {
    cu_estimate(bytecode)
        .saturating_add(SWAP_OVERHEAD)
        .saturating_add(TWO_TRANSFERS)
        .saturating_add(CLOCK_SYSCALL)
        .saturating_add(VAULT_AUTH_DERIVE)
        .saturating_add(TX_CONTEXT_GATHER)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn cu_estimate_baseline_load_const_halt() {
        let mut code = alloc::vec::Vec::new();
        code.push(Op::LoadConst as u8);
        code.extend(42i64.to_le_bytes());
        code.push(Op::Halt as u8);
        let est = cu_estimate(&code);
        assert!(
            est <= 200,
            "baseline LoadConst+Halt should be under 200 CU, got {est}"
        );
    }

    #[test]
    fn cu_estimate_unknown_opcode_returns_high_estimate() {
        let code = [0xCC];
        let est = cu_estimate(&code);
        assert!(
            est >= 50_000,
            "unknown opcode should yield >=50_000 CU, got {est}"
        );
    }

    #[test]
    fn cu_estimate_truncated_immediate_returns_high_estimate() {
        let code = [Op::LoadConst as u8, 0x00, 0x00];
        let est = cu_estimate(&code);
        assert!(
            est >= 50_000,
            "truncated immediate should yield >=50_000 CU, got {est}"
        );
    }

    #[test]
    fn cu_estimate_empty_bytecode_returns_entry_overhead_only() {
        let est = cu_estimate(&[]);
        assert_eq!(est, ENTRY_OVERHEAD);
    }
}
