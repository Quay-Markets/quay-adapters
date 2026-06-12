//! Compact zero-copy wire format for the test `Evaluate` ix.
//!
//! Replaces borsh (which allocates Vecs and pays ~500 CU/call) with a fixed
//! header + raw byte slices that the program reads in place.
//!
//! Unified-userspace model: one `userspace_count` field replaces the
//! historical `params_count` + `signals_count` split.
//!
//! Layout (all little-endian):
//!
//! ```text
//! [0..4)    bytecode_len        : u32
//! [4..8)    userspace_count     : u32   (bytes; flat MM-controlled region)
//! [8..12)   quotes_count        : u32   (i32 entries; ≤ ARENA_NUM_SLOTS)
//! [12..16)  _reserved           : u32   (must be zero)
//! [16..24)  inv                 : i64   (IN-side inventory; LoadInv)
//! [24..32)  size                : i64
//! [32..40)  side                : i64
//! [40..48)  current_slot        : i64
//! [48..56)  vault_base_atoms    : i64
//! [56..64)  vault_quote_atoms   : i64
//! [64..72)  inventory_base      : u64
//! [72..80)  inventory_quote     : u64
//! [80..88)  current_unix_sec    : i64
//! [88..89)  base_decimals       : u8
//! [89..90)  quote_decimals      : u8
//! [90..96)  _pad                : [u8;6] (must be zero)
//! [96..104)  arena_timestamp_sec : i64
//! [104..112) last_update_slot    : u64
//! [112..112+B)              bytecode bytes
//! [112+B..+U)               userspace bytes
//! [112+B+U..+4*Q)           quotes (Q × i32 LE)
//! ```

pub const HEADER_LEN: usize = 112;

/// Borrowed view into the wire-format ix data. No allocation.
pub struct UnpackedInputs<'a> {
    pub bytecode: &'a [u8],
    /// Unified MM-controlled region (R/W to the VM).
    pub userspace_bytes: &'a [u8],
    /// Raw arena slot bytes (read-only); convert via `parse_quotes`.
    pub quotes_bytes: &'a [u8],
    pub quotes_count: usize,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    TooShort,
    InvalidLength,
    ReservedNonZero,
    ParseBufferTooSmall,
}

pub fn unpack(data: &[u8]) -> Result<UnpackedInputs<'_>, WireError> {
    if data.len() < HEADER_LEN {
        return Err(WireError::TooShort);
    }
    let bytecode_len = read_u32(data, 0) as usize;
    let userspace_count = read_u32(data, 4) as usize;
    let quotes_count = read_u32(data, 8) as usize;
    let reserved = read_u32(data, 12);
    if reserved != 0 {
        return Err(WireError::ReservedNonZero);
    }
    let inv = read_i64(data, 16);
    let size = read_i64(data, 24);
    let side = read_i64(data, 32);
    let current_slot = read_i64(data, 40);
    let vault_base_atoms = read_i64(data, 48);
    let vault_quote_atoms = read_i64(data, 56);
    let inventory_base = read_u64(data, 64);
    let inventory_quote = read_u64(data, 72);
    let current_unix_sec = read_i64(data, 80);
    let base_decimals = data[88];
    let quote_decimals = data[89];
    for b in &data[90..96] {
        if *b != 0 {
            return Err(WireError::ReservedNonZero);
        }
    }
    let arena_timestamp_sec = read_i64(data, 96);
    let last_update_slot = read_u64(data, 104);

    let bc_end = HEADER_LEN
        .checked_add(bytecode_len)
        .ok_or(WireError::InvalidLength)?;
    let us_end = bc_end
        .checked_add(userspace_count)
        .ok_or(WireError::InvalidLength)?;
    let q_bytes = quotes_count
        .checked_mul(4)
        .ok_or(WireError::InvalidLength)?;
    let q_end = us_end
        .checked_add(q_bytes)
        .ok_or(WireError::InvalidLength)?;
    if q_end != data.len() {
        return Err(WireError::InvalidLength);
    }

    Ok(UnpackedInputs {
        bytecode: &data[HEADER_LEN..bc_end],
        userspace_bytes: &data[bc_end..us_end],
        quotes_bytes: &data[us_end..q_end],
        quotes_count,
        inv,
        size,
        side,
        current_slot,
        vault_base_atoms,
        vault_quote_atoms,
        inventory_base,
        inventory_quote,
        current_unix_sec,
        base_decimals,
        quote_decimals,
        arena_timestamp_sec,
        last_update_slot,
    })
}

#[inline(always)]
fn read_u32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}
#[inline(always)]
fn read_i64(d: &[u8], at: usize) -> i64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[at..at + 8]);
    i64::from_le_bytes(b)
}
#[inline(always)]
fn read_u64(d: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[at..at + 8]);
    u64::from_le_bytes(b)
}

/// Read quotes bytes (i32 LE) into a caller-provided buffer.
pub fn parse_quotes(bytes: &[u8], buf: &mut [i32]) -> Result<usize, WireError> {
    let n = bytes.len() / 4;
    if n > buf.len() {
        return Err(WireError::ParseBufferTooSmall);
    }
    for i in 0..n {
        let mut x = [0u8; 4];
        x.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
        buf[i] = i32::from_le_bytes(x);
    }
    Ok(n)
}
