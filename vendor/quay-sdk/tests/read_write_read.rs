//! Within a single swap, userspace is one mutable buffer with no journaling:
//! a write is immediately visible to a later read in the *same* execution.
//! Built with `quay_sdk::dsl::BytecodeBuilder`.
//!
//! ```text
//!   load_u64[0]      ; read1 = initial value
//!   drop             ; (discard read1)
//!   load_const 777
//!   store_u64[0]     ; write 777
//!   load_u64[0]      ; read2 — must observe the write
//!   halt             ; amount_out = read2
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]

use quay_sdk::dsl::BytecodeBuilder;
use quay_vm::{evaluate, Inputs, TxContext};

#[test]
fn write_is_visible_to_a_later_read_in_the_same_swap() {
    let bc = BytecodeBuilder::new()
        .load_u64(0)
        .drop_op()
        .load_const(777)
        .store_u64(0)
        .load_u64(0)
        .halt()
        .build();

    // Start with a non-777 value so we know the second read isn't just the
    // initial state leaking through.
    let mut userspace = [0u8; 8];
    userspace.copy_from_slice(&123u64.to_le_bytes());

    let inputs = Inputs {
        quotes: &[],
        userspace: &mut userspace,
        size: 1,
        side: 0,
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

    let out = evaluate(&bc, inputs).expect("evaluate").amount_out();

    // The second read returned the just-written 777, not the initial 123.
    assert_eq!(out, 777, "later read must see the write made earlier this swap");
    assert_eq!(u64::from_le_bytes(userspace), 777);
}
