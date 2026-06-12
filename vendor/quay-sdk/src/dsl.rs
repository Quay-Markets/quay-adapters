//! DSL bytecode helpers.
//!
//! Thin fluent builder over `quay_vm::Op`. Curves manage their own userspace
//! layout (byte offsets into the unified userspace region) — the builder
//! just emits opcodes + immediates. Without an install-time validator there
//! is no `build_and_validate`; the VM bounds-checks every access at swap
//! time. Use `simulate::simulate_swap` to dry-run before committing.

pub use quay_vm::{Op, MAX_BYTECODE_LEN};

use crate::error::StandardReject;

/// Fluent builder over `quay_vm::Op`. Emits raw bytecode.
///
/// ```ignore
/// let bytecode = BytecodeBuilder::new()
///     .load_arena_quote(0)
///     .load_i64(0)           // curve constant at userspace[0..8]
///     .mul()
///     .halt()
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct BytecodeBuilder {
    code: Vec<u8>,
}

/// Handle to a forward-branch immediate that hasn't been patched yet.
/// Returned by `BytecodeBuilder::start_if_eq` (et al.) / `start_jmp` and
/// consumed by `BytecodeBuilder::patch` once the landing site is known.
///
/// Each `Label` is single-use — `patch` consumes it.
#[derive(Debug)]
pub struct Label {
    imm_pos: usize,
    after: usize,
}

impl BytecodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Raw opcode emit (no immediate).
    pub fn op(mut self, op: Op) -> Self {
        self.code.push(op as u8);
        self
    }

    fn imm_u8(mut self, op: Op, b: u8) -> Self {
        self.code.push(op as u8);
        self.code.push(b);
        self
    }

    fn imm_u32(mut self, op: Op, n: u32) -> Self {
        self.code.push(op as u8);
        self.code.extend_from_slice(&n.to_le_bytes());
        self
    }

    fn imm_i16(mut self, op: Op, n: i16) -> Self {
        self.code.push(op as u8);
        self.code.extend_from_slice(&n.to_le_bytes());
        self
    }

    fn imm_i64(mut self, op: Op, n: i64) -> Self {
        self.code.push(op as u8);
        self.code.extend_from_slice(&n.to_le_bytes());
        self
    }

    // ── Arena loads ─────────────────────────────────────────────────────────
    pub fn load_arena_quote(self, idx: u8) -> Self { self.imm_u8(Op::LoadArenaQuote, idx) }
    pub fn load_arena_quote_u(self, idx: u8) -> Self { self.imm_u8(Op::LoadArenaQuoteU, idx) }
    pub fn load_arena_timestamp_sec(self) -> Self { self.op(Op::LoadArenaTimestampSec) }

    // ── Input-register loads ────────────────────────────────────────────────
    pub fn load_inv(self) -> Self { self.op(Op::LoadInv) }
    pub fn load_inv_base(self) -> Self { self.op(Op::LoadInvBase) }
    pub fn load_inv_quote(self) -> Self { self.op(Op::LoadInvQuote) }
    pub fn load_size(self) -> Self { self.op(Op::LoadSize) }
    pub fn load_side(self) -> Self { self.op(Op::LoadSide) }
    pub fn load_vault_base(self) -> Self { self.op(Op::LoadVaultBase) }
    pub fn load_vault_quote(self) -> Self { self.op(Op::LoadVaultQuote) }
    pub fn load_now_slot(self) -> Self { self.op(Op::LoadNowSlot) }
    pub fn load_now_unix_sec(self) -> Self { self.op(Op::LoadNowUnixSec) }
    pub fn load_base_decimals(self) -> Self { self.op(Op::LoadBaseDecimals) }
    pub fn load_quote_decimals(self) -> Self { self.op(Op::LoadQuoteDecimals) }
    pub fn load_last_update_slot(self) -> Self { self.op(Op::LoadLastUpdateSlot) }
    pub fn load_const(self, v: i64) -> Self { self.imm_i64(Op::LoadConst, v) }

    // ── Transaction introspection (TxContext) ───────────────────────────────
    //
    // Host-gathered context — see SPEC §5.2b for soundness boundaries.
    // Curves using any of these require the Instructions sysvar account
    // appended to the swap ix; the simulator defaults to a benign
    // `TxContext::DIRECT` so router quoting paths are unaffected.

    /// Invocation stack height (1 = top-level, 2 = router → quay, …).
    pub fn load_ix_depth(self) -> Self { self.op(Op::LoadIxDepth) }
    /// Heuristic flag bits (`TX_FLAG_JITO_TIP`, `TX_FLAG_FLASHLOAN`).
    pub fn load_tx_flags(self) -> Self { self.op(Op::LoadTxFlags) }
    /// Push 1 if the transaction entrypoint program == `key`. Equals the
    /// immediate caller only at depth ≤ 2 — see `direct_caller_is`.
    pub fn entrypoint_is(mut self, key: &[u8; 32]) -> Self {
        self.code.push(Op::EntrypointIs as u8);
        self.code.extend_from_slice(key);
        self
    }
    /// Push 1 if `key` is in the transaction-level signer set.
    pub fn is_signed_by(mut self, key: &[u8; 32]) -> Self {
        self.code.push(Op::IsSignedBy as u8);
        self.code.extend_from_slice(key);
        self
    }
    /// Push 1 iff `key` is **provably** the immediate caller: depth ≤ 2 and
    /// the entrypoint is `key`. At depth ≥ 3 the immediate caller is not
    /// introspectable on Solana, so this pushes 0 regardless of `key` —
    /// "wrapped by an unknown program" is itself the signal.
    pub fn direct_caller_is(self, key: &[u8; 32]) -> Self {
        self.load_ix_depth()
            .load_const(2)
            .cmp_le()
            .entrypoint_is(key)
            .and()
    }

    // ── Userspace loads / stores ───────────────────────────────────────────
    pub fn load_i8(self, off: u32) -> Self { self.imm_u32(Op::LoadI8, off) }
    pub fn load_i16(self, off: u32) -> Self { self.imm_u32(Op::LoadI16, off) }
    pub fn load_i32(self, off: u32) -> Self { self.imm_u32(Op::LoadI32, off) }
    pub fn load_i64(self, off: u32) -> Self { self.imm_u32(Op::LoadI64, off) }
    pub fn load_u8(self, off: u32) -> Self { self.imm_u32(Op::LoadU8, off) }
    pub fn load_u16(self, off: u32) -> Self { self.imm_u32(Op::LoadU16, off) }
    pub fn store_i64(self, off: u32) -> Self { self.imm_u32(Op::StoreI64, off) }

    // ── Stack manipulation ──────────────────────────────────────────────────
    pub fn dup(self) -> Self { self.op(Op::Dup) }
    pub fn swap_op(self) -> Self { self.op(Op::Swap) }
    pub fn drop_op(self) -> Self { self.op(Op::Drop) }
    /// `Op::Pick imm` — copy the imm-th-from-top stack value to the top.
    /// `pick(0)` is equivalent to `dup()`; `pick(1)` is the second-from-
    /// top, etc. Used by the typed DSL's CSE pass; rarely needed when
    /// hand-writing bytecode.
    pub fn pick(self, depth_from_top: u8) -> Self {
        self.imm_u8(Op::Pick, depth_from_top)
    }

    // ── Arithmetic ──────────────────────────────────────────────────────────
    pub fn add(self) -> Self { self.op(Op::Add) }
    pub fn sub(self) -> Self { self.op(Op::Sub) }
    pub fn mul(self) -> Self { self.op(Op::Mul) }
    pub fn div(self) -> Self { self.op(Op::Div) }
    pub fn muldiv(self) -> Self { self.op(Op::Muldiv) }
    pub fn div_ceil(self) -> Self { self.op(Op::DivCeil) }
    pub fn mod_op(self) -> Self { self.op(Op::Mod) }
    pub fn neg(self) -> Self { self.op(Op::Neg) }
    pub fn abs(self) -> Self { self.op(Op::Abs) }
    pub fn sqrt(self) -> Self { self.op(Op::Sqrt) }

    // ── Fixed-point / Q-format helpers ──────────────────────────────────────
    /// `(a * b) >> bits` with i256 intermediate. `bits` must be ≤ 127.
    /// Replaces `LoadConst <2^bits>; Muldiv` for power-of-two divisors —
    /// saves ~120 CU vs Muldiv (no div, just shift).
    pub fn mul_shr(self, bits: u8) -> Self { self.imm_u8(Op::MulShr, bits) }
    /// Sugar for `mul_shr(48)`. The Q48 mul half-step in Q48 pricing math.
    pub fn mul_q48(self) -> Self { self.op(Op::MulQ48) }
    /// `(a << 48) / b` with i256 intermediate. Reciprocal Q48 step.
    pub fn div_q48(self) -> Self { self.op(Op::DivQ48) }
    /// `((a * b) >> 48) + c`. Single op for the common
    /// `mid + skew * mid >> 48` pattern. Pops c (top), b, a; pushes result.
    pub fn muladd_q48(self) -> Self { self.op(Op::MuladdQ48) }

    // ── Bitwise ─────────────────────────────────────────────────────────────
    pub fn and(self) -> Self { self.op(Op::And) }
    pub fn or(self) -> Self { self.op(Op::Or) }
    pub fn xor(self) -> Self { self.op(Op::Xor) }
    pub fn not(self) -> Self { self.op(Op::Not) }
    pub fn shl(self) -> Self { self.op(Op::Shl) }
    pub fn shr(self) -> Self { self.op(Op::Shr) }

    // ── Min/Max/Clamp ───────────────────────────────────────────────────────
    pub fn min(self) -> Self { self.op(Op::Min) }
    pub fn max(self) -> Self { self.op(Op::Max) }
    pub fn clamp(self) -> Self { self.op(Op::Clamp) }

    // ── Comparisons + select ────────────────────────────────────────────────
    pub fn cmp_lt(self) -> Self { self.op(Op::CmpLt) }
    pub fn cmp_gt(self) -> Self { self.op(Op::CmpGt) }
    pub fn cmp_le(self) -> Self { self.op(Op::CmpLe) }
    pub fn cmp_ge(self) -> Self { self.op(Op::CmpGe) }
    pub fn cmp_eq(self) -> Self { self.op(Op::CmpEq) }
    pub fn select(self) -> Self { self.op(Op::Select) }

    // ── Lookup tables (offsets into userspace) ──────────────────────────────
    pub fn tier_lookup(self, table_off: u32) -> Self { self.imm_u32(Op::TierLookup, table_off) }
    pub fn lerp_lookup(self, table_off: u32) -> Self { self.imm_u32(Op::LerpLookup, table_off) }
    pub fn tier_lookup_dyn(self) -> Self { self.op(Op::TierLookupDyn) }
    /// Multi-value tier lookup. Table rows are `(threshold, value_a, value_b)`
    /// — 24 bytes each. Pops `key`, pushes `value_a` then `value_b`
    /// (`value_b` ends up on top). Use `pack_tier_lookup2_table` to build
    /// the table bytes.
    pub fn tier_lookup2(self, table_off: u32) -> Self { self.imm_u32(Op::TierLookup2, table_off) }

    // ── Control flow (forward-only) ─────────────────────────────────────────
    pub fn if_lt(self, fwd: i16) -> Self { self.imm_i16(Op::IfLt, fwd) }
    pub fn if_gt(self, fwd: i16) -> Self { self.imm_i16(Op::IfGt, fwd) }
    pub fn if_le(self, fwd: i16) -> Self { self.imm_i16(Op::IfLe, fwd) }
    pub fn if_ge(self, fwd: i16) -> Self { self.imm_i16(Op::IfGe, fwd) }
    pub fn if_eq(self, fwd: i16) -> Self { self.imm_i16(Op::IfEq, fwd) }
    pub fn jmp(self, fwd: i16) -> Self { self.imm_i16(Op::Jmp, fwd) }

    // ── Forward-jump backpatching ────────────────────────────────────────────
    //
    // The raw `if_*`/`jmp` helpers above need the forward offset known up
    // front. For real templates the target is wherever the builder happens
    // to land after some intermediate emits, so we need a backpatching API.
    //
    // Usage:
    // ```ignore
    // let (b, l_skip)   = b.start_if_lt();   // pop top 2; jump to `skip` if a < b
    //    let b         = b.emit_active_block();
    // let (b, l_done)   = b.start_jmp();     // unconditional jump past the else
    // let b             = b.patch(l_skip);   // skip lands here
    //    let b         = b.emit_inactive_block();
    // let b             = b.patch(l_done);   // done lands here
    // ```

    /// Forward-jump label. Returned by `start_*` methods and consumed by
    /// `patch` once the destination is reached.
    pub fn start_if_lt(self) -> (Self, Label) { self.start_branch(Op::IfLt) }
    pub fn start_if_gt(self) -> (Self, Label) { self.start_branch(Op::IfGt) }
    pub fn start_if_le(self) -> (Self, Label) { self.start_branch(Op::IfLe) }
    pub fn start_if_ge(self) -> (Self, Label) { self.start_branch(Op::IfGe) }
    pub fn start_if_eq(self) -> (Self, Label) { self.start_branch(Op::IfEq) }
    pub fn start_jmp(self)   -> (Self, Label) { self.start_branch(Op::Jmp)   }

    fn start_branch(mut self, op: Op) -> (Self, Label) {
        self.code.push(op as u8);
        let imm_pos = self.code.len();
        self.code.extend_from_slice(&0i16.to_le_bytes());
        let after = self.code.len();
        (self, Label { imm_pos, after })
    }

    /// Backpatch a previously-emitted forward branch so that it lands at
    /// the current end-of-bytecode.
    ///
    /// # Panics
    /// If the patch target is more than `i16::MAX` bytes from the branch.
    pub fn patch(mut self, label: Label) -> Self {
        let fwd_isize = (self.code.len() - label.after) as isize;
        let fwd: i16 = fwd_isize.try_into().expect("forward jump > i16::MAX");
        self.code[label.imm_pos..label.imm_pos + 2].copy_from_slice(&fwd.to_le_bytes());
        self
    }

    /// `if cond != 0 { a } else { b }` — pushes the three operands of `Op::Select`
    /// in the correct order, then emits `Select`.
    ///
    /// Quay's `Op::Select` pops `[cond, a, b]` top-down (top is `b`) and pushes
    /// `cond != 0 ? a : b`. Hand-emitting that is easy to invert; this helper
    /// gets the order right by construction:
    ///
    /// ```ignore
    /// // exec_q24 = if side == 0 { exec_btoa } else { exec_atob }
    /// b.select_if(
    ///     |b| b.load_i64(US_EXEC_BTOA),      // b: side == 0 branch
    ///     |b| b.load_i64(US_EXEC_ATOB),      // a: side != 0 branch
    ///     |b| b.load_side(),                 // cond
    /// )
    /// ```
    pub fn select_if<B, A, C>(self, b_branch: B, a_branch: A, cond: C) -> Self
    where
        B: FnOnce(Self) -> Self,
        A: FnOnce(Self) -> Self,
        C: FnOnce(Self) -> Self,
    {
        // Select pops [cond, a, b] top-down (top = b), so we must emit
        // cond first (bottom of triple), then a, then b (top).
        b_branch(a_branch(cond(self))).op(Op::Select)
    }

    // ── Terminators ─────────────────────────────────────────────────────────
    pub fn halt(self) -> Self { self.op(Op::Halt) }

    /// Emit `Op::Reject <reason>` with a protocol-named reason.
    pub fn reject_standard(self, reason: StandardReject) -> Self {
        self.imm_u8(Op::Reject, reason as u8)
    }

    /// Emit `Op::Reject <byte>` with an MM-custom reason in `0x80..=0xFF`.
    /// Panics if the byte is in the protocol-reserved `0x00..=0x7F` band.
    pub fn reject_custom(self, byte: u8) -> Self {
        assert!(
            byte >= 0x80,
            "reject_custom: reasons 0x00..=0x7F are protocol-reserved; use reject_standard"
        );
        self.imm_u8(Op::Reject, byte)
    }

    /// Emit `Op::Reject <byte>` without protocol-band validation. Most
    /// callers should use `reject_standard` / `reject_custom`.
    pub fn reject_raw(self, byte: u8) -> Self {
        self.imm_u8(Op::Reject, byte)
    }

    /// Emit `Op::AssertNonzero <reason>`: pop a value off the stack;
    /// REJECT with the given standard reason if the value is zero. The
    /// idiomatic guard pattern is `<compare>; assert_nonzero_standard(...)`:
    ///
    /// ```ignore
    /// // assert vault_quote >= 1000
    /// builder
    ///     .load_vault_quote()
    ///     .load_const(1000)
    ///     .cmp_ge()
    ///     .assert_nonzero_standard(StandardReject::SizeOutOfRange);
    /// ```
    pub fn assert_nonzero_standard(self, reason: StandardReject) -> Self {
        self.imm_u8(Op::AssertNonzero, reason as u8)
    }

    /// `AssertNonzero` with an MM-custom reason byte in `0x80..=0xFF`.
    /// Panics if the byte is in the protocol-reserved `0x00..=0x7F` band.
    pub fn assert_nonzero_custom(self, byte: u8) -> Self {
        assert!(
            byte >= 0x80,
            "assert_nonzero_custom: reasons 0x00..=0x7F are protocol-reserved; use assert_nonzero_standard"
        );
        self.imm_u8(Op::AssertNonzero, byte)
    }

    /// Finalise to raw bytecode.
    pub fn build(self) -> Vec<u8> {
        self.code
    }

    /// Current bytecode length in bytes.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }
}

/// Helper: pack a `TierLookup`/`LerpLookup` table into the byte layout the
/// VM expects (N as i64 LE, then N pairs of (threshold, value), each i64 LE).
/// Caller positions the resulting bytes inside the strategy's userspace via
/// `set_userspace` (partial write at an offset).
pub fn pack_lookup_table(entries: &[(i64, i64)]) -> Vec<u8> {
    let n = entries.len() as i64;
    let mut out = Vec::with_capacity(8 + entries.len() * 16);
    out.extend_from_slice(&n.to_le_bytes());
    for (t, v) in entries {
        out.extend_from_slice(&t.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Helper: pack a `TierLookup2` table — `N` as i64 LE followed by `N` rows
/// of `(threshold, value_a, value_b)`, each i64 LE (24 B/row). Use with
/// `BytecodeBuilder::tier_lookup2`.
pub fn pack_tier_lookup2_table(entries: &[(i64, i64, i64)]) -> Vec<u8> {
    let n = entries.len() as i64;
    let mut out = Vec::with_capacity(8 + entries.len() * 24);
    out.extend_from_slice(&n.to_le_bytes());
    for (t, a, b) in entries {
        out.extend_from_slice(&t.to_le_bytes());
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_emits_loadconst_halt() {
        let code = BytecodeBuilder::new().load_const(42).halt().build();
        assert_eq!(code[0], Op::LoadConst as u8);
        assert_eq!(&code[1..9], &42i64.to_le_bytes());
        assert_eq!(code[9], Op::Halt as u8);
    }

    #[test]
    fn pack_lookup_table_layout() {
        let bytes = pack_lookup_table(&[(0, 100), (10, 200)]);
        // 8 (N) + 2 * 16 (pairs) = 40
        assert_eq!(bytes.len(), 40);
        let n = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(n, 2);
        let t0 = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let v0 = i64::from_le_bytes(bytes[16..24].try_into().unwrap());
        assert_eq!(t0, 0);
        assert_eq!(v0, 100);
    }

    #[test]
    #[should_panic(expected = "reject_custom: reasons 0x00..=0x7F are protocol-reserved")]
    fn reject_custom_refuses_standard_band() {
        let _ = BytecodeBuilder::new().reject_custom(0x00);
    }

    #[test]
    fn reject_custom_accepts_mm_band() {
        let _ = BytecodeBuilder::new().reject_custom(0x80);
    }

    #[test]
    fn forward_jump_backpatch_matches_manual_emit() {
        // Manual: IfEq <12>; LoadConst 99; ... patch lands at end
        // We emit: LoadSize; LoadConst 0; <branch>; LoadConst 99; Halt
        let manual = BytecodeBuilder::new()
            .load_size()
            .load_const(0)
            .if_eq(10)              // skip the LoadConst 99 + Halt below
            .load_const(99)
            .halt()
            .build();

        let backpatched = {
            let (b, label) = BytecodeBuilder::new()
                .load_size()
                .load_const(0)
                .start_if_eq();
            b.load_const(99).halt().patch(label).build()
        };

        assert_eq!(backpatched, manual);
    }

    #[test]
    fn forward_jump_backpatch_lands_at_current_pos() {
        // The patched offset should equal (current_len - after_branch).
        let (b, l) = BytecodeBuilder::new()
            .load_const(0)
            .load_const(1)
            .start_if_lt();
        // before patch the imm bytes are still 0
        assert_eq!(&b.code[b.code.len() - 2..], &[0, 0]);
        let bytes_after_branch = b.code.len();
        // emit 5 more bytes, then patch
        let b = b.load_const(42); // 9 bytes
        let patched = b.patch(l).build();
        // The imm at the IfLt should now equal 9 (bytes emitted between branch
        // and the patch site).
        let i16_at = i16::from_le_bytes(
            patched[bytes_after_branch - 2..bytes_after_branch].try_into().unwrap(),
        );
        assert_eq!(i16_at, 9);
    }

    #[test]
    fn select_if_emits_correct_pop_order() {
        // `select_if(b_branch, a_branch, cond)` should be equivalent to:
        // cond; a_branch; b_branch; Select.
        let helper = BytecodeBuilder::new()
            .select_if(
                |b| b.load_const(1),       // b: side==0 branch
                |b| b.load_const(2),       // a: side!=0 branch
                |b| b.load_side(),         // cond
            )
            .build();
        let manual = BytecodeBuilder::new()
            .load_side()       // cond first (bottom of stack triple)
            .load_const(2)     // a next
            .load_const(1)     // b on top
            .op(Op::Select)
            .build();
        assert_eq!(helper, manual);
    }
}
