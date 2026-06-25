//! A stateful curve whose **price depends on its own state**: it tallies buys
//! and sells in `userspace` (as in `counter_curve.rs`), then quotes a *wider
//! spread on every even-numbered swap of that side*. Built with
//! `quay_sdk::dsl::BytecodeBuilder`.
//!
//! This is the case the titan adapter's "stateful drift" note is really about
//! (`titan/src/lib.rs`): the curve reads a counter it just bumped to choose the
//! price, so the fill genuinely moves as state advances. Off-chain `quote()`
//! prices a snapshot and discards the mutation, so a cached quote computed at
//! count N mis-predicts a fill that actually lands at count N+1 — bounded only
//! by the route's `min_amount_out` slippage guard. Here we drive the VM with a
//! persistent `userspace`, so we can watch the spread alternate for real.
//!
//! Userspace layout (16 bytes): sell count @0..8, buy count @8..16.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]

use quay_sdk::dsl::BytecodeBuilder;
use quay_vm::{evaluate, Inputs, TxContext};

const SIDE_SELL_BASE: u8 = 0;
const SIDE_BUY_BASE: u8 = 1;

const SELL_COUNT_OFF: u32 = 0;
const BUY_COUNT_OFF: u32 = 8;

const BPS_DENOM: i64 = 10_000;
const NARROW_SPREAD_BPS: i64 = 10; // 0.10% on odd swaps
const WIDE_SPREAD_BPS: i64 = 100; // 1.00% on even swaps

/// Build the even-widens-spread curve fluently.
///
/// ```text
///   load_side; load_const 0; if_eq -> SELL
///   ; BUY: bump buy count, leave the NEW count on the stack
///   load_i64[8]; load_const 1; add; dup; store_i64[8]; jmp -> SPREAD
/// SELL:
///   load_i64[0]; load_const 1; add; dup; store_i64[0]
/// SPREAD:                       ; stack: [count]
///   load_const 2; mod           ; [count % 2]
///   load_const 0; if_eq -> EVEN ; even count → wide spread
///   load_const NARROW; jmp -> APPLY
/// EVEN:
///   load_const WIDE
/// APPLY:                        ; stack: [spread_bps]
///   load_const 10000; swap; sub ; [10000 - spread_bps]
///   load_size; load_const 10000; muldiv   ; net * size / 10000
///   halt
/// ```
fn build_spread_curve() -> Vec<u8> {
    // `count = load[off] + 1; store[off] = count;` leaving `count` on the stack.
    fn bump(b: BytecodeBuilder, off: u32) -> BytecodeBuilder {
        b.load_i64(off).load_const(1).add().dup().store_i64(off)
    }

    let (b, to_sell) = BytecodeBuilder::new().load_side().load_const(0).start_if_eq();

    // BUY branch, then jump to the shared SPREAD section.
    let b = bump(b, BUY_COUNT_OFF);
    let (b, to_spread) = b.start_jmp();

    // SELL branch (if_eq target).
    let b = bump(b.patch(to_sell), SELL_COUNT_OFF);

    // SPREAD (jmp target): stack [count] → [count % 2], then branch on parity.
    let b = b.patch(to_spread).load_const(2).mod_op().load_const(0);
    let (b, to_even) = b.start_if_eq();

    // ODD branch: narrow spread, then skip the EVEN branch.
    let b = b.load_const(NARROW_SPREAD_BPS);
    let (b, to_apply) = b.start_jmp();

    // EVEN branch (if_eq target): wide spread.
    let b = b.patch(to_even).load_const(WIDE_SPREAD_BPS);

    // APPLY (jmp target): out = size * (10000 - spread) / 10000.
    b.patch(to_apply)
        .load_const(BPS_DENOM)
        .swap_op()
        .sub()
        .load_size()
        .load_const(BPS_DENOM)
        .muldiv()
        .halt()
        .build()
}

fn run_swap(bytecode: &[u8], userspace: &mut [u8], side: u8, size: u64) -> u64 {
    let inputs = Inputs {
        quotes: &[],
        userspace,
        size,
        side,
        current_slot: 0,
        inventory_base: u64::MAX,
        inventory_quote: u64::MAX,
        current_unix_sec: 0,
        base_decimals: 6,
        quote_decimals: 6,
        quotes_timestamp_ns: 1,
        last_update_slot: 0,
        tx: TxContext::DIRECT,
    };
    evaluate(bytecode, inputs)
        .expect("spread curve should evaluate")
        .amount_out()
}

fn read_count(userspace: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(userspace[off..off + 8].try_into().unwrap())
}

/// `out = size * (10000 - spread_bps) / 10000`.
fn expected(size: u64, spread_bps: i64) -> u64 {
    (size as i128 * (BPS_DENOM - spread_bps) as i128 / BPS_DENOM as i128) as u64
}

#[test]
fn even_swaps_get_the_wider_spread() {
    let bc = build_spread_curve();
    let mut userspace = [0u8; 16];
    let size = 1_000_000u64;

    // Buys: count 1,2,3,4 → narrow, wide, narrow, wide.
    for n in 1..=4u64 {
        let out = run_swap(&bc, &mut userspace, SIDE_BUY_BASE, size);
        let spread = if n % 2 == 0 { WIDE_SPREAD_BPS } else { NARROW_SPREAD_BPS };
        assert_eq!(out, expected(size, spread), "buy #{n}");
        assert_eq!(read_count(&userspace, BUY_COUNT_OFF as usize), n as i64);
    }

    // Concretely: even buys are worse for the taker than odd buys.
    assert_eq!(expected(size, NARROW_SPREAD_BPS), 999_000); // odd
    assert_eq!(expected(size, WIDE_SPREAD_BPS), 990_000); // even
}

#[test]
fn buy_and_sell_spreads_track_independent_counters() {
    let bc = build_spread_curve();
    let mut userspace = [0u8; 16];
    let size = 500_000u64;

    // One buy (count 1 → odd → narrow) then one sell (count 1 → odd → narrow).
    assert_eq!(run_swap(&bc, &mut userspace, SIDE_BUY_BASE, size), expected(size, NARROW_SPREAD_BPS));
    assert_eq!(run_swap(&bc, &mut userspace, SIDE_SELL_BASE, size), expected(size, NARROW_SPREAD_BPS));

    // Second sell (count 2 → even → wide) while buys is still at 1 (odd).
    assert_eq!(run_swap(&bc, &mut userspace, SIDE_SELL_BASE, size), expected(size, WIDE_SPREAD_BPS));
    // Second buy (count 2 → even → wide).
    assert_eq!(run_swap(&bc, &mut userspace, SIDE_BUY_BASE, size), expected(size, WIDE_SPREAD_BPS));

    assert_eq!(read_count(&userspace, BUY_COUNT_OFF as usize), 2);
    assert_eq!(read_count(&userspace, SELL_COUNT_OFF as usize), 2);
}

/// The drift mechanism, made concrete: a router that cached the quote for the
/// *next* buy sees one price, but if another buy lands first the real fill uses
/// the next parity and pays differently. Same starting state → same price (VM
/// is deterministic); advanced state → different price.
#[test]
fn price_moves_as_state_advances() {
    let bc = build_spread_curve();
    let size = 1_000_000u64;

    // Quote computed against a snapshot at buy-count 0 (next swap → count 1, odd).
    let mut snapshot = [0u8; 16];
    let quoted = run_swap(&bc, &mut snapshot.clone(), SIDE_BUY_BASE, size);

    // But on-chain a buy already landed (count is now 1); the *next* buy is
    // count 2 (even) and fills wider.
    run_swap(&bc, &mut snapshot, SIDE_BUY_BASE, size);
    let actual_fill = run_swap(&bc, &mut snapshot, SIDE_BUY_BASE, size);

    assert_eq!(quoted, expected(size, NARROW_SPREAD_BPS));
    assert_eq!(actual_fill, expected(size, WIDE_SPREAD_BPS));
    assert!(actual_fill < quoted, "drift: fill is worse than the stale quote");
}
