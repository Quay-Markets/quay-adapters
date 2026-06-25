//! Quotes-blob writer helpers.
//!
//! The quotes account is a row of `i32` boxes (`QUOTES_NUM_SLOTS`). Single-box
//! values are written directly into the `&mut [i32]` blob passed to
//! [`crate::ix::update_quotes`]. Wide values that a curve reads with the 2-box
//! loads (`LoadQuoteU64` / `LoadQuoteI64`) span two consecutive boxes and must
//! match the VM's combining rule exactly:
//!
//! ```text
//! value = (box[slot] as u32 as u64) | ((box[slot+1] as u32 as u64) << 32)
//! ```
//!
//! i.e. the low box holds the low 32 bits, the high box the high 32 bits. These
//! helpers are the producer side of that contract — without them a keeper would
//! have to hand-pack the bit layout.

use crate::consts::QUOTES_NUM_SLOTS;

/// Write a `u64` across boxes `[slot, slot+1]` for `LoadQuoteU64`.
///
/// # Panics
/// If `slot + 1 >= QUOTES_NUM_SLOTS` (no room for the high box).
pub fn write_quote_u64(blob: &mut [i32], slot: u8, value: u64) {
    let lo = slot as usize;
    let hi = lo + 1;
    assert!(
        hi < QUOTES_NUM_SLOTS && hi < blob.len(),
        "2-box quote write at slot {slot} needs boxes {lo},{hi} within {QUOTES_NUM_SLOTS}",
    );
    // Bit-reinterpret each 32-bit half as i32 (the box type); the VM masks each
    // box back to u32 on read, so the sign of the stored i32 is irrelevant.
    blob[lo] = (value as u32) as i32;
    blob[hi] = ((value >> 32) as u32) as i32;
}

/// Write an `i64` across boxes `[slot, slot+1]` for `LoadQuoteI64`. Identical
/// bit layout to [`write_quote_u64`]; the VM sign-extends from the high box.
///
/// # Panics
/// If `slot + 1 >= QUOTES_NUM_SLOTS`.
pub fn write_quote_i64(blob: &mut [i32], slot: u8, value: i64) {
    write_quote_u64(blob, slot, value as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quay_vm::{evaluate, Inputs, Op, TxContext};

    /// Round-trip the producer against the VM consumer: write a value, then run
    /// `LoadQuoteU64` / `LoadQuoteI64` and assert the VM reads it back.
    fn vm_read(blob: &[i32], op: Op, slot: u8) -> i64 {
        let mut us: [u8; 0] = [];
        let code = [op as u8, slot, Op::Halt as u8];
        let inputs = Inputs {
            quotes: blob,
            userspace: &mut us,
            size: 0,
            side: 0,
            current_slot: 0,
            inventory_base: 0,
            inventory_quote: 0,
            current_unix_sec: 0,
            base_decimals: 0,
            quote_decimals: 0,
            quotes_timestamp_ns: 0,
            last_update_slot: 0,
            tx: TxContext::DIRECT,
        };
        evaluate(&code, inputs).expect("evaluate").amount_out() as i64
    }

    #[test]
    fn u64_round_trips_through_the_vm() {
        let mut blob = vec![0i32; QUOTES_NUM_SLOTS];
        // Low half has bit 31 set (exercises mask-not-sign-extend on the low
        // box); total fits i64 so `Halt` can return it.
        let v: u64 = 0x0000_0001_FFFF_0001;
        write_quote_u64(&mut blob, 3, v);
        assert_eq!(vm_read(&blob, Op::LoadQuoteU64, 3) as u64, v);
    }

    #[test]
    fn i64_round_trips_through_the_vm() {
        let mut blob = vec![0i32; QUOTES_NUM_SLOTS];
        let v: i64 = -(1i64 << 40) - 7;
        write_quote_i64(&mut blob, 10, v);
        // `Halt` rejects a negative `amount_out`, so negate the loaded value
        // first and observe `-v` — proving the high box sign-extended to `v`.
        let mut us: [u8; 0] = [];
        let code = [Op::LoadQuoteI64 as u8, 10, Op::Neg as u8, Op::Halt as u8];
        let inputs = Inputs {
            quotes: &blob,
            userspace: &mut us,
            size: 0,
            side: 0,
            current_slot: 0,
            inventory_base: 0,
            inventory_quote: 0,
            current_unix_sec: 0,
            base_decimals: 0,
            quote_decimals: 0,
            quotes_timestamp_ns: 0,
            last_update_slot: 0,
            tx: TxContext::DIRECT,
        };
        assert_eq!(evaluate(&code, inputs).expect("evaluate").amount_out(), (-v) as u64);
    }

    #[test]
    #[should_panic]
    fn rejects_high_box_out_of_range() {
        let mut blob = vec![0i32; QUOTES_NUM_SLOTS];
        write_quote_u64(&mut blob, (QUOTES_NUM_SLOTS - 1) as u8, 1);
    }
}
