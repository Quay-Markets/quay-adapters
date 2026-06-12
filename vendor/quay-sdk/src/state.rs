//! Zero-copy account decoders.
//!
//! These structs mirror `onchain/program/src/state.rs` exactly — `repr(C,
//! packed)` and `bytemuck::Pod`, so they decode straight off an RPC
//! `account.data` slice with no allocation. `try_from_account` returns a
//! borrowed `&Self`; copy it to an owned value (`let h = *T::try_from_account(d)?;`)
//! before reading multi-byte fields.

use bytemuck::{Pod, Zeroable};

use crate::consts::{
    GLOBAL_CONFIG_DISCRIMINATOR, MARKET_MAKER_DISCRIMINATOR, QUOTES_BLOB_LEN,
    QUOTES_DISCRIMINATOR, QUOTES_NUM_SLOTS, STRATEGY_DISCRIMINATOR,
};
use crate::error::{ClientError, Result};

// ────────────────────────────────────────────────────────────────────────────
// GlobalConfig (128 B)
// ────────────────────────────────────────────────────────────────────────────

#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GlobalConfig {
    pub discriminator: u8,
    pub schema_version: u8,
    pub swap_halted: u8,
    pub protocol_halted: u8,
    pub _pad: [u8; 4],
    pub admin: [u8; 32],
    pub _reserved_a: [u8; 64],
    pub _reserved_b: [u8; 24],
}

impl GlobalConfig {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn try_from_account(data: &[u8]) -> Result<&Self> {
        check(data, Self::SIZE, GLOBAL_CONFIG_DISCRIMINATOR)?;
        Ok(bytemuck::from_bytes::<Self>(&data[..Self::SIZE]))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// QuotesHeader (64 B) — followed by `i32[QUOTES_NUM_SLOTS]`
// ────────────────────────────────────────────────────────────────────────────

#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuotesHeader {
    pub discriminator: u8,
    pub schema_version: u8,
    pub _pad: [u8; 6],
    pub owner: [u8; 32],
    pub updated_ts: u64,
    pub _reserved: [u8; 16],
}

impl QuotesHeader {
    pub const SIZE: usize = core::mem::size_of::<Self>();
    pub const ACCOUNT_SIZE: usize = Self::SIZE + QUOTES_BLOB_LEN;

    pub fn try_from_account(data: &[u8]) -> Result<&Self> {
        check(data, Self::ACCOUNT_SIZE, QUOTES_DISCRIMINATOR)?;
        Ok(bytemuck::from_bytes::<Self>(&data[..Self::SIZE]))
    }

    /// The packed quote blob (`QUOTES_BLOB_LEN` bytes).
    pub fn quote_blob(data: &[u8]) -> Result<&[u8]> {
        check(data, Self::ACCOUNT_SIZE, QUOTES_DISCRIMINATOR)?;
        Ok(&data[Self::SIZE..Self::ACCOUNT_SIZE])
    }

    /// Read all `QUOTES_NUM_SLOTS` slots as `i32`.
    pub fn read_all_slots(data: &[u8]) -> Result<[i32; QUOTES_NUM_SLOTS]> {
        let blob = Self::quote_blob(data)?;
        let mut out = [0i32; QUOTES_NUM_SLOTS];
        for (i, chunk) in blob.chunks_exact(4).enumerate() {
            out[i] = i32::from_le_bytes(chunk.try_into().unwrap());
        }
        Ok(out)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StrategyHeader (192 B) — followed by bytecode + userspace
// ────────────────────────────────────────────────────────────────────────────

#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct StrategyHeader {
    pub discriminator: u8,
    pub schema_version: u8,
    pub dsl_version: u8,
    pub bump: u8,
    pub frozen_admin: u8,
    pub frozen: u8,
    pub _pad: [u8; 2],
    pub owner: [u8; 32],
    pub base_mint: [u8; 32],
    pub quote_mint: [u8; 32],
    pub quotes_account: [u8; 32],
    pub base_id: u16,
    pub quote_id: u16,
    pub protocol_fee_bps: u16,
    pub _pad2: u16,
    pub bytecode_len: u32,
    pub userspace_len: u32,
    pub last_update_slot: u64,
    pub _reserved: [u8; 32],
}

impl StrategyHeader {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn try_from_account(data: &[u8]) -> Result<&Self> {
        check(data, Self::SIZE, STRATEGY_DISCRIMINATOR)?;
        Ok(bytemuck::from_bytes::<Self>(&data[..Self::SIZE]))
    }

    fn extent(&self) -> (usize, usize) {
        let bc_end = Self::SIZE + self.bytecode_len as usize;
        let us_end = bc_end + self.userspace_len as usize;
        (bc_end, us_end)
    }

    /// Pricing bytecode.
    pub fn bytecode<'a>(&self, data: &'a [u8]) -> Result<&'a [u8]> {
        let (bc_end, us_end) = self.extent();
        if data.len() < us_end {
            return Err(ClientError::LogicalExtentExceedsAccount {
                got: data.len(),
                need: us_end,
            });
        }
        Ok(&data[Self::SIZE..bc_end])
    }

    /// MM-controlled userspace region.
    pub fn userspace<'a>(&self, data: &'a [u8]) -> Result<&'a [u8]> {
        let (bc_end, us_end) = self.extent();
        if data.len() < us_end {
            return Err(ClientError::LogicalExtentExceedsAccount {
                got: data.len(),
                need: us_end,
            });
        }
        Ok(&data[bc_end..us_end])
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MarketMakerHeader (64 B) — followed by `AssetEntry[asset_count]`
// ────────────────────────────────────────────────────────────────────────────

#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MarketMakerHeader {
    pub discriminator: u8,
    pub bump: u8,
    pub frozen_admin: u8,
    pub halted_admin: u8,
    pub frozen: u8,
    pub _pad: [u8; 3],
    pub owner: [u8; 32],
    pub asset_count: u32,
    pub _reserved: [u8; 20],
}

impl MarketMakerHeader {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn try_from_account(data: &[u8]) -> Result<&Self> {
        check(data, Self::SIZE, MARKET_MAKER_DISCRIMINATOR)?;
        Ok(bytemuck::from_bytes::<Self>(&data[..Self::SIZE]))
    }

    /// The asset table (`asset_count` entries).
    pub fn assets<'a>(&self, data: &'a [u8]) -> Result<&'a [AssetEntry]> {
        let need = Self::SIZE + self.asset_count as usize * AssetEntry::SIZE;
        if data.len() < need {
            return Err(ClientError::LogicalExtentExceedsAccount {
                got: data.len(),
                need,
            });
        }
        Ok(bytemuck::cast_slice(&data[Self::SIZE..need]))
    }

    /// A single asset slot.
    pub fn asset<'a>(&self, data: &'a [u8], idx: u16) -> Result<&'a AssetEntry> {
        if u32::from(idx) >= self.asset_count {
            return Err(ClientError::InvalidInput("asset index >= asset_count"));
        }
        Ok(&self.assets(data)?[idx as usize])
    }
}

/// One row of a MarketMaker asset table.
#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AssetEntry {
    pub mint: [u8; 32],
    pub amount: u64,
    pub protocol_fee_amount: u64,
}

impl AssetEntry {
    pub const SIZE: usize = core::mem::size_of::<Self>();
}

// ────────────────────────────────────────────────────────────────────────────

/// Validate `data` is long enough and carries `disc` at byte 0.
fn check(data: &[u8], need: usize, disc: u8) -> Result<()> {
    if data.len() < need {
        return Err(ClientError::AccountTooSmall {
            got: data.len(),
            need,
        });
    }
    if data[0] != disc {
        return Err(ClientError::BadDiscriminator {
            got: data[0],
            expected: disc,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{
        GLOBAL_CONFIG_SIZE, MARKET_MAKER_HEADER_SIZE, QUOTES_HEADER_SIZE, STRATEGY_HEADER_SIZE,
        ASSET_ENTRY_SIZE,
    };

    #[test]
    fn header_sizes_match() {
        assert_eq!(GlobalConfig::SIZE, GLOBAL_CONFIG_SIZE);
        assert_eq!(QuotesHeader::SIZE, QUOTES_HEADER_SIZE);
        assert_eq!(StrategyHeader::SIZE, STRATEGY_HEADER_SIZE);
        assert_eq!(MarketMakerHeader::SIZE, MARKET_MAKER_HEADER_SIZE);
        assert_eq!(AssetEntry::SIZE, ASSET_ENTRY_SIZE);
    }
}
