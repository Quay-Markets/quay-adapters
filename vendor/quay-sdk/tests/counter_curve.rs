//! A *genuinely* stateful curve, built with `quay_sdk::dsl::BytecodeBuilder`:
//! a flat 1:1 pricer that also tallies how many buys and sells have gone
//! through, keeping the two counts in `userspace`.
//!
//! Unlike the dead-code `StoreI64`-after-`Halt` splice in the titan
//! construction test, the writes here run on the live path and persist across
//! swaps — the one thing no parity fixture pins down: a curve whose `userspace`
//! write actually executes.
//!
//! Userspace layout (16 bytes, two `i64` counters):
//!   - bytes 0..8   → sell count  (side == SIDE_SELL_BASE == 0)
//!   - bytes 8..16  → buy  count  (side == SIDE_BUY_BASE  == 1)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]

use quay_sdk::dsl::BytecodeBuilder;
use quay_vm::{evaluate, is_stateless, Inputs, TxContext};

const SIDE_SELL_BASE: u8 = 0;
const SIDE_BUY_BASE: u8 = 1;

const SELL_COUNT_OFF: u32 = 0;
const BUY_COUNT_OFF: u32 = 8;

/// Build the counter curve fluently.
///
/// ```text
///   load_side; load_const 0; if_eq -> SELL
///   ; BUY  (side == 1): buys = buys + 1; store
///   load_i64[8]; load_const 1; add; store_i64[8]; jmp -> PRICE
/// SELL (side == 0): sells = sells + 1; store
///   load_i64[0]; load_const 1; add; store_i64[0]
/// PRICE: load_size; halt          ; amount_out = amount_in (flat 1:1)
/// ```
fn build_counter_curve() -> Vec<u8> {
    let (b, to_sell) = BytecodeBuilder::new().load_side().load_const(0).start_if_eq();

    // BUY branch, then jump over the SELL branch to PRICE.
    let b = b
        .load_i64(BUY_COUNT_OFF)
        .load_const(1)
        .add()
        .store_i64(BUY_COUNT_OFF);
    let (b, to_price) = b.start_jmp();

    // SELL branch (if_eq target).
    let b = b
        .patch(to_sell)
        .load_i64(SELL_COUNT_OFF)
        .load_const(1)
        .add()
        .store_i64(SELL_COUNT_OFF);

    // PRICE (jmp target): flat 1:1.
    b.patch(to_price).load_size().halt().build()
}

/// Run one swap against a persistent `userspace`, returning `amount_out`. The
/// buffer is borrowed mutably, so any store the curve runs mutates it in place
/// — the next call sees the updated counts, exactly as sequential on-chain
/// swaps would.
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
        .expect("counter curve should evaluate")
        .amount_out()
}

fn read_count(userspace: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(userspace[off..off + 8].try_into().unwrap())
}

#[test]
fn counter_curve_is_classified_stateful() {
    // It writes userspace on the live path, so the static classifier must flag
    // it stateful — unlike the dead-code store in the titan construction test.
    let bc = build_counter_curve();
    assert!(
        !is_stateless(&bc),
        "a curve that runs a store must be classified stateful"
    );
}

#[test]
fn counts_buys_and_sells_separately_across_swaps() {
    let bc = build_counter_curve();
    let mut userspace = [0u8; 16];

    // buy, buy, sell, buy, sell  → 3 buys, 2 sells.
    let session = [
        (SIDE_BUY_BASE, 1_000_000u64),
        (SIDE_BUY_BASE, 250_000),
        (SIDE_SELL_BASE, 999),
        (SIDE_BUY_BASE, 7),
        (SIDE_SELL_BASE, 1_000_000_000),
    ];

    for (side, size) in session {
        let out = run_swap(&bc, &mut userspace, side, size);
        assert_eq!(out, size, "flat curve must still price 1:1");
    }

    assert_eq!(read_count(&userspace, BUY_COUNT_OFF as usize), 3, "buys");
    assert_eq!(read_count(&userspace, SELL_COUNT_OFF as usize), 2, "sells");
}

#[test]
fn counter_persists_and_increments_one_per_swap() {
    let bc = build_counter_curve();
    let mut userspace = [0u8; 16];

    for n in 1..=5 {
        run_swap(&bc, &mut userspace, SIDE_SELL_BASE, 42);
        assert_eq!(read_count(&userspace, SELL_COUNT_OFF as usize), n);
        assert_eq!(read_count(&userspace, BUY_COUNT_OFF as usize), 0);
    }
    for n in 1..=3 {
        run_swap(&bc, &mut userspace, SIDE_BUY_BASE, 42);
        assert_eq!(read_count(&userspace, BUY_COUNT_OFF as usize), n);
        assert_eq!(read_count(&userspace, SELL_COUNT_OFF as usize), 5);
    }
}
