//! Stack-machine interpreter.
//!
//! Single interpreter, used both on-chain (via the swap ix) and off-chain
//! (via the SDK simulator). Identical semantics in both. The interpreter is
//! **safe by construction**: every read/write goes through Rust slice
//! indexing or an explicit bounds check that returns `RuntimeError`. There
//! is no install-time validator and no `unsafe` in dispatch.
//!
//! ## Userspace model
//!
//! The VM sees three slices:
//! - `bytecode` — executable, read-only at runtime, set by `init_strategy` /
//!   `update_strategy_bytecode`.
//! - `userspace` — the strategy-owned data region, byte-addressable, R/W.
//!   Curves choose their own layout: tables, constants, regime bits, scratch
//!   for cross-swap state — all here. Both owner (via `set_userspace`)
//!   and the running bytecode (via `StoreI64`) can write.
//! - `quotes` — the Quotes account's 241-slot `i32` blob, read-only.
//!
//! Everything else (inventory, vault balances, decimals, clock) is scalar
//! input passed by value.
//!
//! ## Safety contract
//!
//! - Every userspace access bounds-checks against `userspace.len()`.
//! - Jump targets bounds-check against `bytecode.len()` AND must be
//!   forward-only (`target > pc`).
//! - Stack accesses bounds-check against `MAX_STACK_DEPTH`.
//! - Step counter caps total work at `MAX_STEPS = 1024`.
//! - Unknown opcodes return `UnknownOpcode(byte)`.
//!
//! Any violation → `RuntimeError`, swap reverts, no state mutation.

use crate::opcode::{Op, MAX_STACK_DEPTH, MAX_STEPS, MAX_TABLE_ENTRIES};

/// Value returned at a terminator: the curve's quoted exec price (Q24).
///
/// Pop'd from the top of the stack at `Op::Halt`. Always interpreted by
/// the swap program as `quote_per_base × 2²⁴` — settled by `apply_price`.
///
/// Wraps a raw `i64` so the swap program can't accidentally route the
/// pop'd value to inventory math without going through the apply-price
/// scaling step.
///
/// Fees are derived **outside** the VM — the swap path runs the curve twice
/// (once for the user's side, once for the opposite side on a cloned
/// userspace), takes the half-spread between the two prices, and charges
/// `protocol_fee_share_bps` of that. Curves no longer declare a fee amount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalResult(pub i64);

impl EvalResult {
    /// The Q24 exec_price that landed on top of the stack at `Op::Halt`.
    pub fn exec_price(&self) -> i64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeError {
    StepLimitExceeded,
    StackOverflow,
    StackUnderflow,
    ArithmeticOverflow,
    DivByZero,
    InvalidShiftAmount,
    HaltNotReached,
    UnknownOpcode(u8),
    TruncatedImmediate,
    /// Userspace read or write would touch bytes outside `userspace.len()`.
    UserspaceOutOfBounds,
    /// Arena slot index out of `[0, ARENA_NUM_SLOTS)`.
    QuoteIndexOutOfBounds,
    /// Jump immediate would land on a target outside `bytecode.len()` or
    /// backward (`target <= pc`). Forward-only jumps keep execution bounded.
    InvalidJumpTarget,
    /// `TierLookup`/`LerpLookup` table at the given offset is malformed
    /// (N == 0, N > MAX_TABLE_ENTRIES, runs past userspace end, or thresholds
    /// not strictly ascending).
    TierTableMalformed,
    /// i128 result didn't fit into i64 (overflow at the boundary).
    ResultOverflowsI64,
    /// Curve-emitted abort via `Op::Reject <reason>`. The reason byte is
    /// passed through to the program-side `ProgramError::Custom(0xC000 | r)`
    /// mapping. `0x00..=0x7F` are protocol-defined "standard reasons" (see
    /// SDK `StandardReject` enum); `0x80..=0xFF` are MM-custom.
    Reject(u8),
}

/// A same-transaction transfer credits one of the configured Jito tip
/// accounts. Detected by scanning every top-level instruction's account
/// metas, so CPI-routed tips inside this transaction are caught too. A tip
/// in a *different* transaction of the same bundle is invisible — treat
/// as a heuristic signal, never an authorization boundary.
pub const TX_FLAG_JITO_TIP: u64 = 1 << 0;
/// A configured flashloan program appears among the transaction's
/// top-level instructions. Flashloans invoked via CPI from an unknown
/// program are invisible — heuristic signal only.
pub const TX_FLAG_FLASHLOAN: u64 = 1 << 1;

/// Upper bound on the signer set the host passes in `TxContext.signers`.
/// On-chain the swap ix truncates beyond this; curves relying on
/// `IsSignedBy` for a signer that could be crowded out by 16 others
/// should not — real transactions carry far fewer signers.
pub const MAX_TX_SIGNERS: usize = 16;

/// Host-gathered transaction-introspection context.
///
/// The VM never touches syscalls or sysvars — the host gathers these
/// before `evaluate` so on-chain and off-chain runs stay bit-identical.
/// On-chain the swap ix fills this from the stack-height syscall + the
/// Instructions sysvar (a required trailing account on every swap);
/// off-chain simulator callers supply it, defaulting to
/// [`TxContext::DIRECT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxContext<'a> {
    /// Invocation stack height of the swap ix. 1 = top-level transaction
    /// instruction, 2 = one CPI hop (router → quay), etc.
    pub ix_depth: u8,
    /// Host-computed heuristic bits (`TX_FLAG_*`).
    pub tx_flags: u64,
    /// Program id of the top-level instruction this swap executes under.
    /// This is the *transaction entrypoint* — it equals the immediate
    /// caller iff `ix_depth <= 2`. Intermediate CPI callers are not
    /// introspectable on Solana.
    pub entrypoint_program: [u8; 32],
    /// Deduped transaction-level signer set (from Instructions-sysvar
    /// account metas), at most [`MAX_TX_SIGNERS`] entries.
    pub signers: &'a [[u8; 32]],
}

impl TxContext<'_> {
    /// Benign default: direct top-level call, no flags, no signers known,
    /// zeroed entrypoint (every `EntrypointIs` pushes 0). Off-chain
    /// quoting paths use this so context-gated curves price their
    /// "direct taker" branch.
    pub const DIRECT: TxContext<'static> = TxContext {
        ix_depth: 1,
        tx_flags: 0,
        entrypoint_program: [0u8; 32],
        signers: &[],
    };
}

/// Inputs available to the VM at swap time.
///
/// Each Strategy carries its own `userspace` slice — sized at
/// `init_strategy` and resizable via `resize_userspace`. Curves manage
/// their own layout inside it.
pub struct Inputs<'a> {
    /// Quotes blob (read-only). Indexed by `LoadArenaQuote <u8>` into 241
    /// `i32` slots.
    pub quotes: &'a [i32],

    /// Strategy-owned R/W data region. Curves address bytes by `u32` offset
    /// via `LoadI{8,16,32,64}` / `LoadU8` / `StoreI64`. Owner can also
    /// rewrite ranges between swaps via `set_userspace`.
    pub userspace: &'a mut [u8],

    pub inv: i64,
    pub size: i64,
    pub side: i64,
    pub current_slot: i64,
    pub vault_base_atoms: i64,
    pub vault_quote_atoms: i64,
    pub inventory_base: u64,
    pub inventory_quote: u64,
    pub current_unix_sec: i64,
    pub base_decimals: u8,
    pub quote_decimals: u8,
    pub arena_timestamp_sec: i64,
    pub last_update_slot: u64,

    /// Transaction-introspection context. Hosts that don't gather it pass
    /// [`TxContext::DIRECT`].
    pub tx: TxContext<'a>,
}

/// Run bytecode to completion. Returns the i64 on top of the stack at HALT
/// (the curve's exec_price, Q24). Fees are derived outside the VM — see
/// `EvalResult` doc.
pub fn evaluate(bytecode: &[u8], inputs: Inputs<'_>) -> Result<EvalResult, RuntimeError> {
    let mut stack = [0i128; MAX_STACK_DEPTH as usize];
    let mut sp: usize = 0;
    let mut pc: usize = 0;
    let mut steps: u32 = 0;

    macro_rules! push {
        ($v:expr) => {{
            if sp >= stack.len() {
                return Err(RuntimeError::StackOverflow);
            }
            stack[sp] = $v as i128;
            sp += 1;
        }};
    }
    macro_rules! pop {
        () => {{
            if sp == 0 {
                return Err(RuntimeError::StackUnderflow);
            }
            sp -= 1;
            stack[sp]
        }};
    }
    macro_rules! peek {
        ($n:expr) => {{
            if sp <= $n {
                return Err(RuntimeError::StackUnderflow);
            }
            stack[sp - 1 - $n]
        }};
    }

    while pc < bytecode.len() {
        steps += 1;
        if steps > MAX_STEPS {
            return Err(RuntimeError::StepLimitExceeded);
        }

        let op_byte = bytecode[pc];
        let op = Op::from_byte(op_byte).ok_or(RuntimeError::UnknownOpcode(op_byte))?;
        let len = op.instr_len();
        if pc + len > bytecode.len() {
            return Err(RuntimeError::TruncatedImmediate);
        }

        match op {
            Op::LoadArenaQuote => {
                let i = bytecode[pc + 1] as usize;
                if i >= inputs.quotes.len() {
                    return Err(RuntimeError::QuoteIndexOutOfBounds);
                }
                push!(inputs.quotes[i] as i128);
            }
            Op::LoadArenaQuoteU => {
                let i = bytecode[pc + 1] as usize;
                if i >= inputs.quotes.len() {
                    return Err(RuntimeError::QuoteIndexOutOfBounds);
                }
                push!(inputs.quotes[i] as u32 as i128);
            }
            Op::LoadInv => push!(inputs.inv as i128),
            Op::LoadSize => push!(inputs.size as i128),
            Op::LoadSide => push!(inputs.side as i128),
            Op::LoadNowSlot => push!(inputs.current_slot as i128),
            Op::LoadVaultBase => push!(inputs.vault_base_atoms as i128),
            Op::LoadVaultQuote => push!(inputs.vault_quote_atoms as i128),
            Op::LoadInvBase => push!(inputs.inventory_base as i128),
            Op::LoadInvQuote => push!(inputs.inventory_quote as i128),
            Op::LoadNowUnixSec => push!(inputs.current_unix_sec as i128),
            Op::LoadBaseDecimals => push!(inputs.base_decimals as i128),
            Op::LoadQuoteDecimals => push!(inputs.quote_decimals as i128),
            Op::LoadArenaTimestampSec => push!(inputs.arena_timestamp_sec as i128),
            Op::LoadLastUpdateSlot => push!(inputs.last_update_slot as i128),
            Op::LoadIxDepth => push!(inputs.tx.ix_depth as i128),
            Op::LoadTxFlags => push!(inputs.tx.tx_flags as i128),
            Op::EntrypointIs => {
                // 32-byte pubkey immediate; pc+33 <= len proven above.
                let imm = &bytecode[pc + 1..pc + 33];
                push!(i128::from(imm == inputs.tx.entrypoint_program.as_slice()));
            }
            Op::IsSignedBy => {
                let imm = &bytecode[pc + 1..pc + 33];
                let mut signed = false;
                for s in inputs.tx.signers {
                    if s.as_slice() == imm {
                        signed = true;
                        break;
                    }
                }
                push!(i128::from(signed));
            }

            Op::LoadConst => {
                let v = read_i64(bytecode, pc + 1);
                push!(v as i128);
            }

            Op::LoadI8 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = read_userspace_i8(inputs.userspace, off)?;
                push!(v as i128);
            }
            Op::LoadU8 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = read_userspace_u8(inputs.userspace, off)?;
                push!(v as i128);
            }
            Op::LoadU16 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = read_userspace_u16(inputs.userspace, off)?;
                push!(v as i128);
            }
            Op::LoadI16 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = read_userspace_i16(inputs.userspace, off)?;
                push!(v as i128);
            }
            Op::LoadI32 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = read_userspace_i32(inputs.userspace, off)?;
                push!(v as i128);
            }
            Op::LoadI64 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = read_userspace_i64(inputs.userspace, off)?;
                push!(v as i128);
            }
            Op::StoreI64 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let v = pop!();
                let v64 = i128_to_i64(v)?;
                write_userspace_i64(inputs.userspace, off, v64)?;
            }

            Op::Dup => {
                let v = peek!(0);
                push!(v);
            }
            Op::Swap => {
                if sp < 2 {
                    return Err(RuntimeError::StackUnderflow);
                }
                stack.swap(sp - 1, sp - 2);
            }
            Op::Drop => {
                let _ = pop!();
            }
            Op::Pick => {
                // Copy the Nth-from-top stack value to top. N comes from the
                // u8 immediate; N=0 is equivalent to Dup. Bounds-checked
                // against current stack depth so a malformed `Pick imm`
                // beyond the live stack yields `StackUnderflow`.
                let n = bytecode[pc + 1] as usize;
                if n + 1 > sp {
                    return Err(RuntimeError::StackUnderflow);
                }
                let v = stack[sp - 1 - n];
                push!(v);
            }

            Op::Add => {
                let b = pop!();
                let a = pop!();
                push!(a.checked_add(b).ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Sub => {
                let b = pop!();
                let a = pop!();
                push!(a.checked_sub(b).ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Mul => {
                let b = pop!();
                let a = pop!();
                push!(a.checked_mul(b).ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Div => {
                let b = pop!();
                let a = pop!();
                if b == 0 {
                    return Err(RuntimeError::DivByZero);
                }
                push!(a.checked_div(b).ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::DivCeil => {
                let b = pop!();
                let a = pop!();
                if b == 0 {
                    return Err(RuntimeError::DivByZero);
                }
                let q = a.checked_div(b).ok_or(RuntimeError::ArithmeticOverflow)?;
                let r = a.checked_rem(b).ok_or(RuntimeError::ArithmeticOverflow)?;
                let bump = (r != 0) && ((a > 0) == (b > 0));
                let out = if bump {
                    q.checked_add(1).ok_or(RuntimeError::ArithmeticOverflow)?
                } else {
                    q
                };
                push!(out);
            }
            Op::Mod => {
                let b = pop!();
                let a = pop!();
                if b == 0 {
                    return Err(RuntimeError::DivByZero);
                }
                push!(a.checked_rem(b).ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Muldiv => {
                let c = pop!();
                let b = pop!();
                let a = pop!();
                if c == 0 {
                    return Err(RuntimeError::DivByZero);
                }
                let prod = mul_i256(a, b);
                let quot = div_i256_by_i128(prod, c)?;
                push!(quot);
            }
            Op::MulShr => {
                let bits = bytecode[pc + 1];
                if bits > 127 {
                    return Err(RuntimeError::InvalidShiftAmount);
                }
                let b = pop!();
                let a = pop!();
                let prod = mul_i256(a, b);
                let r = shr_i256_to_i128(prod, bits)?;
                push!(r);
            }
            Op::MulQ48 => {
                let b = pop!();
                let a = pop!();
                let prod = mul_i256(a, b);
                let r = shr_i256_to_i128(prod, 48)?;
                push!(r);
            }
            Op::DivQ48 => {
                let b = pop!();
                let a = pop!();
                if b == 0 {
                    return Err(RuntimeError::DivByZero);
                }
                // (a << 48) / b via i256 intermediate. `1 << 48` is positive
                // so sign matches `a`; mul_i256 handles sign.
                let prod = mul_i256(a, 1i128 << 48);
                let quot = div_i256_by_i128(prod, b)?;
                push!(quot);
            }
            Op::MuladdQ48 => {
                // Pop order: c (top), b, a. Compute ((a * b) >> 48) + c.
                let c = pop!();
                let b = pop!();
                let a = pop!();
                let prod = mul_i256(a, b);
                let shifted = shr_i256_to_i128(prod, 48)?;
                push!(shifted
                    .checked_add(c)
                    .ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Neg => {
                let a = pop!();
                push!(a.checked_neg().ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Abs => {
                let a = pop!();
                push!(a.checked_abs().ok_or(RuntimeError::ArithmeticOverflow)?);
            }

            Op::And => {
                let b = pop!();
                let a = pop!();
                push!(a & b);
            }
            Op::Or => {
                let b = pop!();
                let a = pop!();
                push!(a | b);
            }
            Op::Xor => {
                let b = pop!();
                let a = pop!();
                push!(a ^ b);
            }
            Op::Not => {
                let a = pop!();
                push!(!a);
            }
            Op::Shl => {
                let b = pop!();
                let a = pop!();
                if !(0..=127).contains(&b) {
                    return Err(RuntimeError::InvalidShiftAmount);
                }
                push!(a
                    .checked_shl(b as u32)
                    .ok_or(RuntimeError::ArithmeticOverflow)?);
            }
            Op::Shr => {
                let b = pop!();
                let a = pop!();
                if !(0..=127).contains(&b) {
                    return Err(RuntimeError::InvalidShiftAmount);
                }
                push!(a
                    .checked_shr(b as u32)
                    .ok_or(RuntimeError::ArithmeticOverflow)?);
            }

            Op::Min => {
                let b = pop!();
                let a = pop!();
                push!(a.min(b));
            }
            Op::Max => {
                let b = pop!();
                let a = pop!();
                push!(a.max(b));
            }

            Op::CmpLt => {
                let b = pop!();
                let a = pop!();
                push!(i128::from(a < b));
            }
            Op::CmpGt => {
                let b = pop!();
                let a = pop!();
                push!(i128::from(a > b));
            }
            Op::CmpLe => {
                let b = pop!();
                let a = pop!();
                push!(i128::from(a <= b));
            }
            Op::CmpGe => {
                let b = pop!();
                let a = pop!();
                push!(i128::from(a >= b));
            }
            Op::CmpEq => {
                let b = pop!();
                let a = pop!();
                push!(i128::from(a == b));
            }

            Op::Select => {
                let b = pop!();
                let a = pop!();
                let cond = pop!();
                push!(if cond != 0 { a } else { b });
            }
            Op::Clamp => {
                let hi = pop!();
                let lo = pop!();
                let v = pop!();
                if lo > hi {
                    return Err(RuntimeError::ArithmeticOverflow);
                }
                push!(v.min(hi).max(lo));
            }

            Op::TierLookup => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let key = pop!();
                let key64 = i128_to_i64(key)?;
                let v = tier_lookup(inputs.userspace, off, key64)?;
                push!(v as i128);
            }
            Op::LerpLookup => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let key = pop!();
                let v = lerp_lookup(inputs.userspace, off, key)?;
                push!(v);
            }
            Op::TierLookupDyn => {
                let off_signed = pop!();
                let key = pop!();
                if off_signed < 0 || off_signed > u32::MAX as i128 {
                    return Err(RuntimeError::UserspaceOutOfBounds);
                }
                let off = off_signed as usize;
                let key64 = i128_to_i64(key)?;
                let v = tier_lookup(inputs.userspace, off, key64)?;
                push!(v as i128);
            }
            Op::TierLookup2 => {
                let off = read_u32(bytecode, pc + 1) as usize;
                let key = pop!();
                let key64 = i128_to_i64(key)?;
                let (a, b) = tier_lookup2(inputs.userspace, off, key64)?;
                // Push value_a first, then value_b → value_b ends up on top.
                push!(a as i128);
                push!(b as i128);
            }

            Op::IfLt => {
                let b = pop!();
                let a = pop!();
                let imm = read_i16(bytecode, pc + 1);
                if a < b {
                    pc = jump_target(pc, len, imm, bytecode.len())?;
                    continue;
                }
            }
            Op::IfGt => {
                let b = pop!();
                let a = pop!();
                let imm = read_i16(bytecode, pc + 1);
                if a > b {
                    pc = jump_target(pc, len, imm, bytecode.len())?;
                    continue;
                }
            }
            Op::IfLe => {
                let b = pop!();
                let a = pop!();
                let imm = read_i16(bytecode, pc + 1);
                if a <= b {
                    pc = jump_target(pc, len, imm, bytecode.len())?;
                    continue;
                }
            }
            Op::IfGe => {
                let b = pop!();
                let a = pop!();
                let imm = read_i16(bytecode, pc + 1);
                if a >= b {
                    pc = jump_target(pc, len, imm, bytecode.len())?;
                    continue;
                }
            }
            Op::IfEq => {
                let b = pop!();
                let a = pop!();
                let imm = read_i16(bytecode, pc + 1);
                if a == b {
                    pc = jump_target(pc, len, imm, bytecode.len())?;
                    continue;
                }
            }
            Op::Jmp => {
                let imm = read_i16(bytecode, pc + 1);
                pc = jump_target(pc, len, imm, bytecode.len())?;
                continue;
            }
            Op::Sqrt => {
                let a = pop!();
                if a < 0 {
                    return Err(RuntimeError::ArithmeticOverflow);
                }
                push!(isqrt_i128(a));
            }
            Op::Halt => {
                if sp == 0 {
                    return Err(RuntimeError::StackUnderflow);
                }
                let v = stack[sp - 1];
                let exec_price = i128_to_i64(v)?;
                return Ok(EvalResult(exec_price));
            }
            Op::Reject => {
                let reason = bytecode[pc + 1];
                return Err(RuntimeError::Reject(reason));
            }
            Op::AssertNonzero => {
                let reason = bytecode[pc + 1];
                let v = pop!();
                if v == 0 {
                    return Err(RuntimeError::Reject(reason));
                }
            }
        }

        pc += len;
    }
    Err(RuntimeError::HaltNotReached)
}

/// Compute forward jump target and bounds-check.
#[inline]
fn jump_target(
    pc: usize,
    len: usize,
    imm: i16,
    bytecode_len: usize,
) -> Result<usize, RuntimeError> {
    // Forward-only: imm >= 0. Backward jumps would allow loops, which our
    // step counter would catch but at the cost of wasted CU per griefing
    // attempt — refuse outright.
    if imm < 0 {
        return Err(RuntimeError::InvalidJumpTarget);
    }
    let after = pc.checked_add(len).ok_or(RuntimeError::InvalidJumpTarget)?;
    let target = after
        .checked_add(imm as usize)
        .ok_or(RuntimeError::InvalidJumpTarget)?;
    if target >= bytecode_len {
        return Err(RuntimeError::InvalidJumpTarget);
    }
    Ok(target)
}

#[inline]
fn read_userspace_i8(us: &[u8], off: usize) -> Result<i8, RuntimeError> {
    us.get(off)
        .copied()
        .map(|b| b as i8)
        .ok_or(RuntimeError::UserspaceOutOfBounds)
}
#[inline]
fn read_userspace_u8(us: &[u8], off: usize) -> Result<u8, RuntimeError> {
    us.get(off)
        .copied()
        .ok_or(RuntimeError::UserspaceOutOfBounds)
}
#[inline]
fn read_userspace_i16(us: &[u8], off: usize) -> Result<i16, RuntimeError> {
    let end = off
        .checked_add(2)
        .ok_or(RuntimeError::UserspaceOutOfBounds)?;
    if end > us.len() {
        return Err(RuntimeError::UserspaceOutOfBounds);
    }
    Ok(i16::from_le_bytes(us[off..end].try_into().unwrap()))
}
#[inline]
fn read_userspace_u16(us: &[u8], off: usize) -> Result<u16, RuntimeError> {
    let end = off
        .checked_add(2)
        .ok_or(RuntimeError::UserspaceOutOfBounds)?;
    if end > us.len() {
        return Err(RuntimeError::UserspaceOutOfBounds);
    }
    Ok(u16::from_le_bytes(us[off..end].try_into().unwrap()))
}
#[inline]
fn read_userspace_i32(us: &[u8], off: usize) -> Result<i32, RuntimeError> {
    let end = off
        .checked_add(4)
        .ok_or(RuntimeError::UserspaceOutOfBounds)?;
    if end > us.len() {
        return Err(RuntimeError::UserspaceOutOfBounds);
    }
    Ok(i32::from_le_bytes(us[off..end].try_into().unwrap()))
}
#[inline]
fn read_userspace_i64(us: &[u8], off: usize) -> Result<i64, RuntimeError> {
    let end = off
        .checked_add(8)
        .ok_or(RuntimeError::UserspaceOutOfBounds)?;
    if end > us.len() {
        return Err(RuntimeError::UserspaceOutOfBounds);
    }
    Ok(i64::from_le_bytes(us[off..end].try_into().unwrap()))
}
#[inline]
fn write_userspace_i64(us: &mut [u8], off: usize, v: i64) -> Result<(), RuntimeError> {
    let end = off
        .checked_add(8)
        .ok_or(RuntimeError::UserspaceOutOfBounds)?;
    if end > us.len() {
        return Err(RuntimeError::UserspaceOutOfBounds);
    }
    us[off..end].copy_from_slice(&v.to_le_bytes());
    Ok(())
}

/// Deterministic integer floor-sqrt of a non-negative i128.
#[inline]
pub(crate) fn isqrt_i128(a: i128) -> i128 {
    debug_assert!(a >= 0);
    if a < 2 {
        return a;
    }
    let ua = a as u128;
    let bits = 128 - ua.leading_zeros();
    let mut x = 1u128 << ((bits + 1) / 2);
    loop {
        let y = (x + ua / x) / 2;
        if y >= x {
            break;
        }
        x = y;
    }
    while x.checked_mul(x).map(|sq| sq > ua).unwrap_or(true) {
        x -= 1;
    }
    x as i128
}

#[inline]
fn read_i64(b: &[u8], at: usize) -> i64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[at..at + 8]);
    i64::from_le_bytes(buf)
}

#[inline]
fn read_u32(b: &[u8], at: usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(buf)
}

#[inline]
fn read_i16(b: &[u8], at: usize) -> i16 {
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&b[at..at + 2]);
    i16::from_le_bytes(buf)
}

#[inline]
fn i128_to_i64(v: i128) -> Result<i64, RuntimeError> {
    if v > i64::MAX as i128 || v < i64::MIN as i128 {
        return Err(RuntimeError::ResultOverflowsI64);
    }
    Ok(v as i64)
}

/// Read a tier/lerp table header at userspace `off`. Returns (n, table_start)
/// where `table_start` is the byte offset of the first (threshold, value) pair.
/// Validates: 1 <= N <= MAX_TABLE_ENTRIES and the full table fits in userspace.
fn read_table_header(us: &[u8], off: usize) -> Result<(usize, usize), RuntimeError> {
    let n_i64 = read_userspace_i64(us, off).map_err(|_| RuntimeError::TierTableMalformed)?;
    if n_i64 <= 0 || n_i64 > MAX_TABLE_ENTRIES as i64 {
        return Err(RuntimeError::TierTableMalformed);
    }
    let n = n_i64 as usize;
    let table_start = off.checked_add(8).ok_or(RuntimeError::TierTableMalformed)?;
    let table_bytes = n.checked_mul(16).ok_or(RuntimeError::TierTableMalformed)?;
    let table_end = table_start
        .checked_add(table_bytes)
        .ok_or(RuntimeError::TierTableMalformed)?;
    if table_end > us.len() {
        return Err(RuntimeError::TierTableMalformed);
    }
    Ok((n, table_start))
}

#[inline]
fn read_table_pair(us: &[u8], table_start: usize, i: usize) -> (i64, i64) {
    let off = table_start + i * 16;
    // Bounds already proven by `read_table_header`; bytes are aligned-via-LE.
    let t = i64::from_le_bytes(us[off..off + 8].try_into().unwrap());
    let v = i64::from_le_bytes(us[off + 8..off + 16].try_into().unwrap());
    (t, v)
}

/// Step-function lookup over a `[N | (t0,v0), (t1,v1), ...]` table at
/// `userspace[off..]`. Each entry is 16 bytes (i64 LE threshold + i64 LE
/// value). Returns the value of the segment where `key < threshold`, or
/// the last value if `key` is at or above every threshold.
///
/// Validates strict-ascending thresholds at lookup time.
pub(crate) fn tier_lookup(us: &[u8], off: usize, key: i64) -> Result<i64, RuntimeError> {
    let (n, table_start) = read_table_header(us, off)?;
    let mut prev_t: Option<i64> = None;
    let mut last_value = read_table_pair(us, table_start, 0).1;
    for i in 0..n {
        let (t, v) = read_table_pair(us, table_start, i);
        if let Some(prev) = prev_t {
            if t <= prev {
                return Err(RuntimeError::TierTableMalformed);
            }
        }
        if key < t {
            return Ok(v);
        }
        last_value = v;
        prev_t = Some(t);
    }
    Ok(last_value)
}

/// Read the multi-value tier table header at userspace `off`. Returns
/// `(N, table_start)` where each row is 24 B `(threshold, value_a, value_b)`.
/// Validates `1 <= N <= MAX_TABLE_ENTRIES` and that the full table fits in
/// userspace.
fn read_table2_header(us: &[u8], off: usize) -> Result<(usize, usize), RuntimeError> {
    let n_i64 = read_userspace_i64(us, off).map_err(|_| RuntimeError::TierTableMalformed)?;
    if n_i64 <= 0 || n_i64 > MAX_TABLE_ENTRIES as i64 {
        return Err(RuntimeError::TierTableMalformed);
    }
    let n = n_i64 as usize;
    let table_start = off.checked_add(8).ok_or(RuntimeError::TierTableMalformed)?;
    let table_bytes = n.checked_mul(24).ok_or(RuntimeError::TierTableMalformed)?;
    let table_end = table_start
        .checked_add(table_bytes)
        .ok_or(RuntimeError::TierTableMalformed)?;
    if table_end > us.len() {
        return Err(RuntimeError::TierTableMalformed);
    }
    Ok((n, table_start))
}

#[inline]
fn read_table2_row(us: &[u8], table_start: usize, i: usize) -> (i64, i64, i64) {
    let off = table_start + i * 24;
    // Bounds already proven by `read_table2_header`.
    let t = i64::from_le_bytes(us[off..off + 8].try_into().unwrap());
    let a = i64::from_le_bytes(us[off + 8..off + 16].try_into().unwrap());
    let b = i64::from_le_bytes(us[off + 16..off + 24].try_into().unwrap());
    (t, a, b)
}

/// Two-value tier lookup. Table layout at `userspace[off..]`:
///   `off..off+8   : N (i64 LE)`
///   `off+8..      : N rows of (threshold i64 LE, value_a i64 LE, value_b i64 LE)`
///
/// Returns `(value_a, value_b)` for the segment where `key < threshold`, or
/// the last row's pair if `key` is at or above every threshold.
///
/// Validates strict-ascending thresholds at lookup time.
pub(crate) fn tier_lookup2(us: &[u8], off: usize, key: i64) -> Result<(i64, i64), RuntimeError> {
    let (n, table_start) = read_table2_header(us, off)?;
    let mut prev_t: Option<i64> = None;
    let first = read_table2_row(us, table_start, 0);
    let mut last_a = first.1;
    let mut last_b = first.2;
    for i in 0..n {
        let (t, a, b) = read_table2_row(us, table_start, i);
        if let Some(prev) = prev_t {
            if t <= prev {
                return Err(RuntimeError::TierTableMalformed);
            }
        }
        if key < t {
            return Ok((a, b));
        }
        last_a = a;
        last_b = b;
        prev_t = Some(t);
    }
    Ok((last_a, last_b))
}

/// Piecewise-linear interpolation over the same table layout as
/// `tier_lookup`. `key <= t_0` → `v_0`; `key >= t_{N-1}` → `v_{N-1}`; in
/// between, linearly interpolates `v_i + (v_{i+1} - v_i) * (key - t_i) /
/// (t_{i+1} - t_i)` using i128 throughout.
pub(crate) fn lerp_lookup(us: &[u8], off: usize, key: i128) -> Result<i128, RuntimeError> {
    let (n, table_start) = read_table_header(us, off)?;

    let (t0, v0) = read_table_pair(us, table_start, 0);
    if key <= t0 as i128 {
        return Ok(v0 as i128);
    }
    let (last_t, last_v) = read_table_pair(us, table_start, n - 1);
    if key >= last_t as i128 {
        return Ok(last_v as i128);
    }

    let mut prev_t: i64 = t0;
    let mut prev_v: i64 = v0;
    for i in 1..n {
        let (t, v) = read_table_pair(us, table_start, i);
        if t <= prev_t {
            return Err(RuntimeError::TierTableMalformed);
        }
        if key < t as i128 {
            // Interpolate between (prev_t, prev_v) and (t, v).
            let dv = (v as i128)
                .checked_sub(prev_v as i128)
                .ok_or(RuntimeError::ArithmeticOverflow)?;
            let dk = key
                .checked_sub(prev_t as i128)
                .ok_or(RuntimeError::ArithmeticOverflow)?;
            let dt = (t as i128)
                .checked_sub(prev_t as i128)
                .ok_or(RuntimeError::ArithmeticOverflow)?;
            let num = dv.checked_mul(dk).ok_or(RuntimeError::ArithmeticOverflow)?;
            let delta = num
                .checked_div(dt)
                .ok_or(RuntimeError::ArithmeticOverflow)?;
            return (prev_v as i128)
                .checked_add(delta)
                .ok_or(RuntimeError::ArithmeticOverflow);
        }
        prev_t = t;
        prev_v = v;
    }
    Ok(last_v as i128)
}

// ---- i256 helpers for MULDIV ----

pub(crate) fn mul_i256(a: i128, b: i128) -> (i128, u128) {
    let neg = (a < 0) ^ (b < 0);
    let ua = a.unsigned_abs();
    let ub = b.unsigned_abs();
    let (hi, lo) = u128_full_mul(ua, ub);
    if neg {
        let (lo_neg, carry_in_hi) = (!lo).overflowing_add(1);
        let hi_neg = (!hi).wrapping_add(if carry_in_hi { 1 } else { 0 });
        (hi_neg as i128, lo_neg)
    } else {
        (hi as i128, lo)
    }
}

fn u128_full_mul(a: u128, b: u128) -> (u128, u128) {
    let a_lo = a as u64 as u128;
    let a_hi = a >> 64;
    let b_lo = b as u64 as u128;
    let b_hi = b >> 64;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = (ll >> 64) + (lh & 0xffff_ffff_ffff_ffff) + (hl & 0xffff_ffff_ffff_ffff);
    let lo = ((mid & 0xffff_ffff_ffff_ffff) << 64) | (ll & 0xffff_ffff_ffff_ffff);
    let hi = hh + (lh >> 64) + (hl >> 64) + (mid >> 64);

    (hi, lo)
}

pub(crate) fn div_i256_by_i128(num: (i128, u128), den: i128) -> Result<i128, RuntimeError> {
    let (hi, lo) = num;
    let neg_num = hi < 0;
    let (uhi, ulo) = if neg_num {
        let (lo_neg, carry_in_hi) = (!lo).overflowing_add(1);
        let hi_neg = (!(hi as u128)).wrapping_add(if carry_in_hi { 1 } else { 0 });
        (hi_neg, lo_neg)
    } else {
        (hi as u128, lo)
    };
    let neg_den = den < 0;
    let uden = den.unsigned_abs();

    let q = u256_div_u128(uhi, ulo, uden)?;

    let neg = neg_num ^ neg_den;
    if neg {
        const I128_MIN_ABS: u128 = 1u128 << 127;
        if q > I128_MIN_ABS {
            return Err(RuntimeError::ArithmeticOverflow);
        }
        if q == I128_MIN_ABS {
            return Ok(i128::MIN);
        }
        Ok(-(q as i128))
    } else {
        if q > i128::MAX as u128 {
            return Err(RuntimeError::ArithmeticOverflow);
        }
        Ok(q as i128)
    }
}

/// Arithmetic right-shift of a signed i256 (represented as `(hi: i128, lo: u128)`)
/// by `bits` (∈ [0, 127]), returning the result as `i128` or an overflow error
/// if the shifted value doesn't fit.
///
/// For `MulShr` / `MulQ48` / `MuladdQ48`. The pair-encoding mirrors `mul_i256`:
/// `hi` carries the sign, `lo` holds the low 128 bits as unsigned.
pub(crate) fn shr_i256_to_i128(pair: (i128, u128), bits: u8) -> Result<i128, RuntimeError> {
    debug_assert!(bits <= 127);
    if bits > 127 {
        return Err(RuntimeError::InvalidShiftAmount);
    }
    let (hi, lo) = pair;
    let n = bits as u32;

    // Compose the i256 → shift right by `n` → check fits in i128.
    // For arithmetic shift on a signed 256-bit value:
    //   new_lo = (lo >> n) | (hi << (128 - n))           (n > 0)
    //   new_hi = hi >> n   (arithmetic — sign-extending)
    // Then verify new_hi is the sign-extension of new_lo viewed as i128.

    let (new_hi, new_lo) = if n == 0 {
        (hi, lo)
    } else {
        // `hi as u128` for the shift-into-lo, but the arithmetic shift of `hi`
        // itself uses signed semantics so the sign bit propagates.
        let lo_from_hi = (hi as u128).checked_shl(128 - n).unwrap_or(0);
        let new_lo = (lo >> n) | lo_from_hi;
        // Signed arithmetic right-shift of `hi` by `n` (in [1, 127]).
        let new_hi = hi >> n;
        (new_hi, new_lo)
    };

    // Result fits in i128 iff `new_hi` is the sign-extension of (new_lo as i128).
    let expected_hi = (new_lo as i128) >> 127;
    if new_hi != expected_hi {
        return Err(RuntimeError::ArithmeticOverflow);
    }
    Ok(new_lo as i128)
}

fn u256_div_u128(hi: u128, lo: u128, den: u128) -> Result<u128, RuntimeError> {
    if den == 0 {
        return Err(RuntimeError::DivByZero);
    }
    if hi == 0 {
        return Ok(lo / den);
    }
    if hi >= den {
        return Err(RuntimeError::ArithmeticOverflow);
    }
    let mut rem = hi;
    let mut quo: u128 = 0;
    let mut bit = 128;
    while bit > 0 {
        bit -= 1;
        rem = (rem << 1) | ((lo >> bit) & 1);
        quo <<= 1;
        if rem >= den {
            rem -= den;
            quo |= 1;
        }
    }
    Ok(quo)
}
