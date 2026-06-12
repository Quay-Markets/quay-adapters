//! Opcode set + decoding helpers.
//!
//! Single-byte opcodes; immediates are little-endian, byte-aligned, follow the
//! opcode immediately. Spec lives in `onchain/vm/SPEC.md`.
//!
//! **Unified-userspace model**: every owner-controlled byte the VM can read or
//! write lives in **one** `userspace` region. The historical `params` /
//! `signals` / `scratch` split is retired; curves manage their own layout
//! inside `userspace` and address bytes by `u32` offset. The dispatcher does
//! bounds checks at every read/write — there is no install-time validator
//! pre-proof. See `vm.rs` for the runtime guarantee and `SPEC.md` for the
//! safety contract.

pub const MAX_BYTECODE_LEN: usize = 4096;
pub const MAX_STACK_DEPTH: u8 = 16;
pub const MAX_STEPS: u32 = 1024;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // ── Loads from arena (writer-controlled, read-only) ────────────────────
    LoadArenaQuote = 0x01,  // imm: u8 — sign-extend i32 slot
    LoadArenaQuoteU = 0x28, // imm: u8 — zero-extend
    LoadArenaTimestampSec = 0x29,

    // ── Loads from VM inputs (swap-time context) ───────────────────────────
    LoadInv = 0x02, // side-relative inventory (legacy)
    LoadSize = 0x03,
    LoadSide = 0x04,
    LoadConst = 0x05, // imm: i64
    LoadNowSlot = 0x27,
    LoadNowUnixSec = 0x31,
    LoadVaultBase = 0x2D,
    LoadVaultQuote = 0x2E,
    LoadInvBase = 0x2F,
    LoadInvQuote = 0x30,
    LoadBaseDecimals = 0x32,
    LoadQuoteDecimals = 0x33,
    LoadLastUpdateSlot = 0x42,

    // ── Transaction-introspection loads (host-gathered TxContext) ──────────
    //
    // Values come from `Inputs.tx`. The VM never does syscalls — the host
    // gathers these before eval (on-chain: stack-height syscall + the
    // Instructions sysvar, parsed only when `uses_tx_context(bytecode)`;
    // off-chain: simulator callers supply them, default `TxContext::DIRECT`).
    // 0x4B is retired (HaltQ48) — do not reuse.
    LoadIxDepth = 0x4C,  // push invocation stack height (1 = top-level)
    LoadTxFlags = 0x4D,  // push host-computed heuristic bits (TX_FLAG_*)
    EntrypointIs = 0x4E, // imm: [u8;32] pubkey — push 1 if the top-level
    //   ix program == imm, else 0. Equals the
    //   immediate caller iff ix_depth <= 2.
    IsSignedBy = 0x4F, // imm: [u8;32] pubkey — push 1 if imm is in the
    //   tx-level signer set, else 0

    // ── Userspace loads (unified MM-controlled region) ─────────────────────
    //
    // Imm `u32 off` is a byte offset into `Inputs.userspace`. Every load
    // bounds-checks `off + width <= userspace.len()` at runtime; out-of-bounds
    // returns `RuntimeError::UserspaceOutOfBounds`. There is no install-time
    // pre-proof — owner can rewrite userspace cheaply (incl. mid-swap via
    // `StoreI64`); the dispatcher protects itself.
    LoadI8 = 0x43,  // imm: u32 — sign-extend byte
    LoadI16 = 0x44, // imm: u32 — sign-extend i16 LE
    LoadI32 = 0x45, // imm: u32 — sign-extend i32 LE
    LoadI64 = 0x46, // imm: u32 — i64 LE
    LoadU8 = 0x47,  // imm: u32 — zero-extend byte
    LoadU16 = 0x4A, // imm: u32 — zero-extend u16 LE

    // ── Userspace stores ────────────────────────────────────────────────────
    StoreI64 = 0x48, // imm: u32 — write i64 LE; bounds-checked

    // ── Stack manipulation ──────────────────────────────────────────────────
    Dup = 0x09,
    Swap = 0x0a,
    Drop = 0x0b,
    Pick = 0x06, // imm: u8 (depth from top, 0..MAX_STACK_DEPTH-1)
    //   — copy the Nth-from-top stack value to top.
    //   Pick 0 is equivalent to Dup; Pick 1 is the
    //   second-from-top, etc. Enables real CSE for
    //   shared subexpressions that aren't adjacent
    //   operands of the same op.

    // ── Arithmetic ──────────────────────────────────────────────────────────
    Add = 0x0c,
    Sub = 0x0d,
    Mul = 0x0e,
    Div = 0x0f,
    Muldiv = 0x10, // (a * b) / c, intermediate i256
    Neg = 0x11,
    Abs = 0x12,
    Sqrt = 0x24,
    DivCeil = 0x3B,
    Mod = 0x3C,

    // ── Fixed-point / Q-format helpers ─────────────────────────────────────
    //
    // Common shapes for Q24 / Q48 pricing math used INSIDE the formula —
    // not at the terminator. Settlement is always Q24 (see `Op::Halt`).
    // Each replaces a 3-5 op chain (LoadConst <shift>, Mul, Shr) with a
    // single op and saves the Muldiv overhead when the divisor is a power
    // of two.
    MulShr = 0x3E,    // imm: u8 (bits) — (a * b) >> bits, bits ∈ [0, 127]
    MulQ48 = 0x3F,    // no imm — (a * b) >> 48; sugar for MulShr 48
    DivQ48 = 0x40,    // no imm — (a << 48) / b
    MuladdQ48 = 0x41, // no imm — ((a * b) >> 48) + c; pops c, b, a

    // ── Bitwise / shifts ────────────────────────────────────────────────────
    And = 0x13,
    Or = 0x14,
    Xor = 0x15,
    Not = 0x16,
    Shl = 0x17,
    Shr = 0x18,

    // ── Min/Max/Clamp ───────────────────────────────────────────────────────
    Min = 0x19,
    Max = 0x1a,
    Clamp = 0x1b, // (val, lo, hi) → clamped

    // ── Lookup ──────────────────────────────────────────────────────────────
    //
    // Table layout (in userspace bytes starting at `off`):
    //   off..off+8   : N as i64 LE (entry count; runtime caps at MAX_TABLE_ENTRIES)
    //   off+8..      : N pairs of (threshold: i64 LE, value: i64 LE), 16 B each
    //
    // Total table footprint: `8 + N * 16` bytes. Runtime checks
    // `off + 8 + N * 16 <= userspace.len()` and `1 <= N <= MAX_TABLE_ENTRIES`.
    // Strict-ascending threshold check happens at lookup time (per call).
    // Malformed → `RuntimeError::TierTableMalformed`.
    TierLookup = 0x1c,    // imm: u32 — table offset; stack: key → value
    LerpLookup = 0x3D,    // imm: u32 — table offset; stack: key → interpolated value
    TierLookupDyn = 0x26, // imm: none; stack: key, table_off → value

    // ── Multi-value tier lookup ────────────────────────────────────────────
    //
    // Table layout: `off..off+8 = N (i64 LE)`, then N rows of 24 bytes each:
    //   (threshold i64 LE, value_a i64 LE, value_b i64 LE).
    // Footprint: `8 + N * 24` bytes. Bounds + strict-ascending threshold
    // check at every lookup. Pops `key`, pushes `value_a` THEN `value_b`
    // (so `value_b` is on top of the stack after the op).
    TierLookup2 = 0x25, // imm: u32 — table offset; stack: key → value_a, value_b

    // ── Comparisons (push 0 or 1) ───────────────────────────────────────────
    CmpLt = 0x35,
    CmpGt = 0x36,
    CmpLe = 0x37,
    CmpGe = 0x38,
    CmpEq = 0x39,

    // ── Branchless select ──────────────────────────────────────────────────
    //
    // Stack: cond, a, b → (cond != 0 ? a : b). Pop order: b, a, cond.
    Select = 0x3A,

    // ── Control flow (forward-only jumps; runtime-checked) ──────────────────
    IfLt = 0x1d, // imm: i16 — forward offset from instr end
    IfGt = 0x1e,
    IfLe = 0x1f,
    IfGe = 0x20,
    IfEq = 0x21,
    Jmp = 0x22, // imm: i16

    // ── Terminators ─────────────────────────────────────────────────────────
    Halt = 0x23,   // pop top of stack as Q24 exec_price
    Reject = 0x34, // imm: u8 (reason); maps to Custom(0xC000 | r)

    // ── Assertions ──────────────────────────────────────────────────────────
    //
    // Sugar over the common defensive-guard pattern (`compare → IfEq jump
    // around → REJECT`). Compose with the CMP* opcodes to express e.g.
    // `assert vault_quote >= 1000`:
    //   LoadVaultQuote, LoadConst 1000, CmpGe, AssertNonzero(reason).
    AssertNonzero = 0x49, // imm: u8 (reason); pop v, REJECT(r) if v == 0
}

/// Maximum number of entries a `TierLookup`/`LerpLookup` table may declare.
/// Caps a single opcode's runtime cost at table-walk time.
pub const MAX_TABLE_ENTRIES: usize = 256;

impl Op {
    /// `const fn` clone of `from_byte` retained for downstream consumers that
    /// need compile-time opcode resolution.
    pub const fn from_byte_const(b: u8) -> Option<Self> {
        Some(match b {
            0x01 => Op::LoadArenaQuote,
            0x02 => Op::LoadInv,
            0x03 => Op::LoadSize,
            0x04 => Op::LoadSide,
            0x05 => Op::LoadConst,
            0x06 => Op::Pick,
            0x09 => Op::Dup,
            0x0a => Op::Swap,
            0x0b => Op::Drop,
            0x0c => Op::Add,
            0x0d => Op::Sub,
            0x0e => Op::Mul,
            0x0f => Op::Div,
            0x10 => Op::Muldiv,
            0x11 => Op::Neg,
            0x12 => Op::Abs,
            0x13 => Op::And,
            0x14 => Op::Or,
            0x15 => Op::Xor,
            0x16 => Op::Not,
            0x17 => Op::Shl,
            0x18 => Op::Shr,
            0x19 => Op::Min,
            0x1a => Op::Max,
            0x1b => Op::Clamp,
            0x1c => Op::TierLookup,
            0x1d => Op::IfLt,
            0x1e => Op::IfGt,
            0x1f => Op::IfLe,
            0x20 => Op::IfGe,
            0x21 => Op::IfEq,
            0x22 => Op::Jmp,
            0x23 => Op::Halt,
            0x24 => Op::Sqrt,
            0x26 => Op::TierLookupDyn,
            0x27 => Op::LoadNowSlot,
            0x28 => Op::LoadArenaQuoteU,
            0x29 => Op::LoadArenaTimestampSec,
            0x2d => Op::LoadVaultBase,
            0x2e => Op::LoadVaultQuote,
            0x2f => Op::LoadInvBase,
            0x30 => Op::LoadInvQuote,
            0x31 => Op::LoadNowUnixSec,
            0x32 => Op::LoadBaseDecimals,
            0x33 => Op::LoadQuoteDecimals,
            0x34 => Op::Reject,
            0x35 => Op::CmpLt,
            0x36 => Op::CmpGt,
            0x37 => Op::CmpLe,
            0x38 => Op::CmpGe,
            0x39 => Op::CmpEq,
            0x3a => Op::Select,
            0x3b => Op::DivCeil,
            0x3c => Op::Mod,
            0x3d => Op::LerpLookup,
            0x42 => Op::LoadLastUpdateSlot,
            0x43 => Op::LoadI8,
            0x44 => Op::LoadI16,
            0x45 => Op::LoadI32,
            0x46 => Op::LoadI64,
            0x47 => Op::LoadU8,
            0x48 => Op::StoreI64,
            0x49 => Op::AssertNonzero,
            0x4a => Op::LoadU16,
            0x4c => Op::LoadIxDepth,
            0x4d => Op::LoadTxFlags,
            0x4e => Op::EntrypointIs,
            0x4f => Op::IsSignedBy,
            0x25 => Op::TierLookup2,
            0x3e => Op::MulShr,
            0x3f => Op::MulQ48,
            0x40 => Op::DivQ48,
            0x41 => Op::MuladdQ48,
            _ => return None,
        })
    }

    /// Decode a byte to an opcode.
    pub fn from_byte(b: u8) -> Option<Self> {
        Self::from_byte_const(b)
    }

    /// Total instruction length in bytes (opcode + immediate).
    pub fn instr_len(self) -> usize {
        match self {
            // u32 imm
            Op::TierLookup
            | Op::LerpLookup
            | Op::TierLookup2
            | Op::LoadI8
            | Op::LoadI16
            | Op::LoadI32
            | Op::LoadI64
            | Op::LoadU8
            | Op::LoadU16
            | Op::StoreI64 => 5,
            // u8 imm
            Op::LoadArenaQuote
            | Op::LoadArenaQuoteU
            | Op::Reject
            | Op::AssertNonzero
            | Op::MulShr
            | Op::Pick => 2,
            // i16 imm
            Op::IfLt | Op::IfGt | Op::IfLe | Op::IfGe | Op::IfEq | Op::Jmp => 3,
            // i64 imm
            Op::LoadConst => 9,
            // 32-byte pubkey imm
            Op::EntrypointIs | Op::IsSignedBy => 33,
            // no imm
            _ => 1,
        }
    }

    /// (pops, pushes) — for stack-depth tracking by external tooling.
    pub fn stack_delta(self) -> (u8, u8) {
        match self {
            // Pure loads (0 → 1)
            Op::LoadArenaQuote
            | Op::LoadArenaQuoteU
            | Op::LoadInv
            | Op::LoadSize
            | Op::LoadSide
            | Op::LoadConst
            | Op::LoadNowSlot
            | Op::LoadNowUnixSec
            | Op::LoadVaultBase
            | Op::LoadVaultQuote
            | Op::LoadInvBase
            | Op::LoadInvQuote
            | Op::LoadBaseDecimals
            | Op::LoadQuoteDecimals
            | Op::LoadArenaTimestampSec
            | Op::LoadLastUpdateSlot
            | Op::LoadI8
            | Op::LoadI16
            | Op::LoadI32
            | Op::LoadI64
            | Op::LoadU8
            | Op::LoadU16
            | Op::LoadIxDepth
            | Op::LoadTxFlags
            | Op::EntrypointIs
            | Op::IsSignedBy => (0, 1),

            // Stores (1 → 0)
            Op::StoreI64 | Op::Drop => (1, 0),

            // Stack ops
            Op::Dup => (1, 2),
            Op::Swap => (2, 2),
            // Pick reads but doesn't consume — net (0 → 1) at the runtime
            // stack level, though it requires `imm + 1` slots present to
            // be well-formed. External tooling should use `instr_len` +
            // runtime bounds checks rather than relying on stack_delta
            // alone for Pick.
            Op::Pick => (0, 1),

            // Binops (2 → 1)
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Shl
            | Op::Shr
            | Op::Min
            | Op::Max
            | Op::CmpLt
            | Op::CmpGt
            | Op::CmpLe
            | Op::CmpGe
            | Op::CmpEq
            | Op::DivCeil
            | Op::Mod
            | Op::MulShr
            | Op::MulQ48
            | Op::DivQ48 => (2, 1),

            // Ternary (3 → 1)
            Op::Muldiv | Op::Clamp | Op::Select | Op::MuladdQ48 => (3, 1),

            // Unary (1 → 1)
            Op::Neg | Op::Abs | Op::Not | Op::Sqrt => (1, 1),

            // Lookups
            Op::TierLookup | Op::LerpLookup => (1, 1),
            Op::TierLookupDyn => (2, 1),
            // Multi-value tier lookup: pops key, pushes (value_a, value_b)
            Op::TierLookup2 => (1, 2),

            // Branches: pop comparands but no net stack change beyond the pop
            Op::IfLt | Op::IfGt | Op::IfLe | Op::IfGe | Op::IfEq => (2, 0),
            Op::Jmp => (0, 0),

            // Terminators
            Op::Halt => (0, 0),
            Op::Reject => (0, 0),

            // Assertions: pop the value to check; no push.
            Op::AssertNonzero => (1, 0),
        }
    }

    /// True if this opcode writes to `userspace` during evaluation. Aggregator-
    /// routed MMs commit to bytecode that contains no such opcode — see
    /// `is_stateless` for the bytecode-level check.
    pub const fn mutates_userspace(self) -> bool {
        matches!(self, Op::StoreI64)
    }

}

/// True if `bytecode` contains no opcode that mutates `userspace` during a
/// swap. Walks the instruction stream respecting `instr_len` so immediate
/// bytes are never mistaken for opcodes.
///
/// **Aggregator-routing contract.** Routers (Jupiter, Titan, ...) cache the
/// result of `simulate_swap` between user request and on-chain fill. If the
/// curve writes to `userspace` mid-swap, on-chain state drifts from the
/// simulator's cached view → the user's actual fill diverges from the
/// quoted amount. Stateless curves carry no such drift; stateful curves
/// should serve direct integrations instead.
///
/// **Conservative on malformed input.** An unknown opcode byte or a
/// truncated immediate returns `false` — we can't prove statelessness on
/// bytes we can't decode.
#[must_use]
pub fn is_stateless(bytecode: &[u8]) -> bool {
    let mut pc: usize = 0;
    while pc < bytecode.len() {
        let Some(op) = Op::from_byte(bytecode[pc]) else {
            return false;
        };
        if op.mutates_userspace() {
            return false;
        }
        let len = op.instr_len();
        let Some(next) = pc.checked_add(len) else {
            return false;
        };
        if next > bytecode.len() {
            return false;
        }
        pc = next;
    }
    true
}

#[cfg(all(test, feature = "std"))]
mod stateless_tests {
    use super::*;

    fn op_byte(o: Op) -> u8 {
        o as u8
    }

    #[test]
    fn empty_bytecode_is_stateless() {
        assert!(is_stateless(&[]));
    }

    #[test]
    fn load_const_halt_is_stateless() {
        let mut bc = vec![op_byte(Op::LoadConst)];
        bc.extend_from_slice(&42i64.to_le_bytes());
        bc.push(op_byte(Op::Halt));
        assert!(is_stateless(&bc));
    }

    #[test]
    fn bytecode_with_store_i64_is_stateful() {
        let mut bc = vec![op_byte(Op::LoadConst)];
        bc.extend_from_slice(&7i64.to_le_bytes());
        bc.push(op_byte(Op::StoreI64));
        bc.extend_from_slice(&0u32.to_le_bytes());
        bc.push(op_byte(Op::Halt));
        assert!(!is_stateless(&bc));
    }

    #[test]
    fn store_i64_byte_inside_immediate_does_not_false_positive() {
        // `LoadConst` immediate happens to contain the 0x48 byte that would
        // be `StoreI64`. The walker must skip immediates, so this curve is
        // still stateless.
        let mut bc = vec![op_byte(Op::LoadConst)];
        bc.extend_from_slice(&(0x4848_4848_4848_4848i64).to_le_bytes());
        bc.push(op_byte(Op::Halt));
        assert!(is_stateless(&bc));
        // Sanity-check that we constructed an i64 containing the StoreI64
        // opcode byte — if this changes, the test no longer probes what
        // we think it does.
        assert!(bc.contains(&op_byte(Op::StoreI64)));
    }

    #[test]
    fn unknown_opcode_is_conservatively_stateful() {
        // 0xFE is not a defined opcode.
        let bc = vec![0xFEu8, op_byte(Op::Halt)];
        assert!(!is_stateless(&bc));
    }

    #[test]
    fn truncated_immediate_is_conservatively_stateful() {
        // LoadConst expects 8 imm bytes; supply 3.
        let bc = vec![op_byte(Op::LoadConst), 0x11, 0x22, 0x33];
        assert!(!is_stateless(&bc));
    }

    #[test]
    fn every_non_store_opcode_is_stateless_in_isolation() {
        // Bytecode of just `Halt` is stateless. Use this as a baseline and
        // confirm `mutates_userspace` agrees with `is_stateless` for every
        // opcode that takes no immediate.
        let single_op_opcodes: &[Op] = &[
            Op::Halt,
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Muldiv,
            Op::Neg,
            Op::Abs,
            Op::Sqrt,
            Op::DivCeil,
            Op::Mod,
            Op::And,
            Op::Or,
            Op::Xor,
            Op::Not,
            Op::Shl,
            Op::Shr,
            Op::Min,
            Op::Max,
            Op::Clamp,
            Op::Dup,
            Op::Swap,
            Op::Drop,
            Op::CmpLt,
            Op::CmpGt,
            Op::CmpLe,
            Op::CmpGe,
            Op::CmpEq,
            Op::Select,
            Op::LoadInv,
            Op::LoadSize,
            Op::LoadSide,
            Op::LoadNowSlot,
            Op::LoadNowUnixSec,
            Op::LoadVaultBase,
            Op::LoadVaultQuote,
            Op::LoadInvBase,
            Op::LoadInvQuote,
            Op::LoadBaseDecimals,
            Op::LoadQuoteDecimals,
            Op::LoadArenaTimestampSec,
            Op::LoadLastUpdateSlot,
            Op::TierLookupDyn,
            Op::MulQ48,
            Op::DivQ48,
            Op::MuladdQ48,
        ];
        for &o in single_op_opcodes {
            assert!(!o.mutates_userspace(), "{o:?} should not mutate userspace");
            assert!(
                is_stateless(&[op_byte(o)]),
                "{o:?} bytecode should be stateless"
            );
        }
    }
}
