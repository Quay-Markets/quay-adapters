//! Client-side errors for state decoding, ix building, and simulation.
//!
//! On-chain errors live in `onchain/program/src/errors.rs::QuayError`. Anything
//! that surfaces from a sent tx is a `solana_program::program_error::ProgramError`
//! the caller can match on; everything in this enum is what the client itself
//! produces before / after the tx.
//!
//! # VM error namespace
//!
//! On-chain VM errors are mapped to `ProgramError::Custom(...)` codes in a
//! stable namespace (see `onchain/program/src/instructions/swap.rs::map_vm_error`):
//!
//! - `0xC000..=0xC0FF` — curve-issued `Reject(reason)`. The lower byte is the
//!   reason: `0x00..=0x7F` are protocol-defined "standard reasons" (see
//!   [`StandardReject`]); `0x80..=0xFF` are MM-custom (opaque to routers).
//! - `0xCE00..=0xCE0F` — VM runtime user-data faults (DivByZero,
//!   ArithmeticOverflow, ResultOverflowsI64, StepLimitExceeded,
//!   InvalidShiftAmount).
//! - `0xDEAD` — should-never-happen-post-validator faults. If you see this,
//!   the validator missed something — file a bug.
//!
//! Off-chain callers decode via [`ClientError::from_program_error`].

use thiserror::Error;

/// Protocol-defined "standard" REJECT reasons. Codes `0x00..=0x7F` carry these
/// meanings by convention; MMs that want to signal these conditions SHOULD
/// emit the named code so routers and wallets can build clean UX. MMs that
/// want bespoke reasons use `0x80..=0xFF` via `BytecodeBuilder::reject_custom`.
///
/// Once published, a standard code is owned by the protocol forever — never
/// repurposed. New named reasons go into the reserved `0x09..=0x7F` band.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardReject {
    /// Bare `REJECT 0` — no reason given.
    Unspecified = 0x00,
    /// Trading venue is closed (e.g. equity-tracking xStock outside NYSE hours).
    MarketClosed = 0x01,
    /// Quote feed older than the curve's tolerance.
    StaleFeed = 0x02,
    /// Insufficient inventory on the side the swap is requesting.
    InventoryLow = 0x03,
    /// Computed price would breach the MM's spread floor.
    SpreadFloor = 0x04,
    /// Trade size outside the MM's accepted band.
    SizeOutOfRange = 0x05,
    /// Curve-side circuit breaker tripped.
    CircuitBreaker = 0x06,
    /// Current regime doesn't accept this trade direction.
    RegimeMismatch = 0x07,
    /// Curve-side self-pause (distinct from `mm.status` byte).
    PolicyPaused = 0x08,
    /// Tx-context gate rejected the flow (unknown entrypoint, heuristic
    /// flags, or non-allowlisted caller).
    UntrustedFlow = 0x09,
    // 0x0A..=0x7F reserved for future protocol additions.
}

impl StandardReject {
    /// Try to convert a raw reason byte to a named standard variant.
    /// Returns `None` for bytes in the MM-custom range (`0x80..=0xFF`) or
    /// in the reserved `0x09..=0x7F` band.
    pub fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            0x00 => Self::Unspecified,
            0x01 => Self::MarketClosed,
            0x02 => Self::StaleFeed,
            0x03 => Self::InventoryLow,
            0x04 => Self::SpreadFloor,
            0x05 => Self::SizeOutOfRange,
            0x06 => Self::CircuitBreaker,
            0x07 => Self::RegimeMismatch,
            0x08 => Self::PolicyPaused,
            0x09 => Self::UntrustedFlow,
            _ => return None,
        })
    }
}

/// Decoded REJECT reason — either a protocol-defined named variant or a raw
/// MM-custom byte. Routers can pattern-match on `Standard(...)` for clean UX
/// and fall through to `Custom(byte)` for unknown / MM-defined codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmRejectReason {
    Standard(StandardReject),
    Custom(u8),
}

impl VmRejectReason {
    pub fn from_byte(b: u8) -> Self {
        match StandardReject::from_byte(b) {
            Some(s) => Self::Standard(s),
            None => Self::Custom(b),
        }
    }

    /// Raw byte for round-trip — useful for `BytecodeBuilder::reject_custom`
    /// or logging.
    pub fn as_byte(self) -> u8 {
        match self {
            Self::Standard(s) => s as u8,
            Self::Custom(b) => b,
        }
    }
}

/// VM runtime fault variant (the `0xCE00..=0xCE0F` namespace).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmFaultKind {
    ArithmeticOverflow = 0x00,
    DivByZero = 0x01,
    ResultOverflowsI64 = 0x02,
    StepLimitExceeded = 0x03,
    InvalidShiftAmount = 0x04,
}

impl VmFaultKind {
    pub fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            0x00 => Self::ArithmeticOverflow,
            0x01 => Self::DivByZero,
            0x02 => Self::ResultOverflowsI64,
            0x03 => Self::StepLimitExceeded,
            0x04 => Self::InvalidShiftAmount,
            _ => return None,
        })
    }
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("account too small: got {got} bytes, need at least {need}")]
    AccountTooSmall { got: usize, need: usize },

    #[error("account discriminator mismatch: got 0x{got:02x}, expected 0x{expected:02x}")]
    BadDiscriminator { got: u8, expected: u8 },

    #[error("account logical extent ({need}) exceeds account data ({got})")]
    LogicalExtentExceedsAccount { got: usize, need: usize },

    #[error("swap math produced zero output (amount_in {amount_in} too small)")]
    ZeroOutputSwap { amount_in: u64 },

    #[error("strategy is frozen")]
    StrategyFrozen,

    #[error("market maker is frozen")]
    MarketMakerFrozen,

    #[error("market maker is halted by admin")]
    MarketMakerHalted,

    #[error("quotes account has never been published")]
    QuotesNotPublished,

    #[error("strategy dsl_version is newer than this client supports")]
    UnsupportedDslVersion,

    #[error("protocol halted")]
    ProtocolHalted,

    #[error("swap halted")]
    SwapHalted,

    #[error("VM error: {0:?}")]
    Vm(quay_vm::RuntimeError),

    /// Curve-emitted abort with a structured reason. Surfaces from on-chain
    /// `ProgramError::Custom(0xC000..=0xC0FF)` and from off-chain
    /// `simulate_swap` when the curve hits `Op::Reject`. Inspect `reason`
    /// to decide whether to retry (e.g. `StaleFeed` → retry; `MarketClosed`
    /// → don't).
    #[error("VM REJECT: {reason:?}")]
    VmReject { reason: VmRejectReason },

    /// Input-dependent VM runtime fault (`0xCE00..=0xCE0F`). Validator can't
    /// statically prevent these — they fire on swap-time inputs the curve
    /// didn't guard against (e.g. divide-by-zero on a zero-priced slot).
    #[error("VM runtime fault: {kind:?}")]
    VmRuntimeFault { kind: VmFaultKind },

    /// `ProgramError::Custom(0xDEAD)` — a VM error the validator should have
    /// caught at install time. File a bug; should not occur in production.
    #[error("VM protocol bug (Custom 0xDEAD) — validator missed an invariant")]
    VmProtocolBug,

    #[error("integer overflow in swap math")]
    SwapMathOverflow,

    #[error("non-positive exec_price emitted by curve: {0}")]
    InvalidExecPrice(i64),

    #[error("slippage: out_to_taker {got} < min_amount_out {min}")]
    SlippageExceeded { got: u64, min: u64 },

    #[error("agg_swap: no candidate strategy produced a usable quote")]
    NoEligibleStrategy,

    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
}

impl From<quay_vm::RuntimeError> for ClientError {
    fn from(e: quay_vm::RuntimeError) -> Self {
        // Match the on-chain namespace mapping so off-chain simulators and
        // on-chain swaps surface the same typed error for the same condition.
        // See `onchain/program/src/instructions/swap.rs::map_vm_error`.
        match e {
            quay_vm::RuntimeError::Reject(r) => Self::VmReject {
                reason: VmRejectReason::from_byte(r),
            },
            quay_vm::RuntimeError::ArithmeticOverflow => Self::VmRuntimeFault {
                kind: VmFaultKind::ArithmeticOverflow,
            },
            quay_vm::RuntimeError::DivByZero => Self::VmRuntimeFault {
                kind: VmFaultKind::DivByZero,
            },
            quay_vm::RuntimeError::ResultOverflowsI64 => Self::VmRuntimeFault {
                kind: VmFaultKind::ResultOverflowsI64,
            },
            quay_vm::RuntimeError::StepLimitExceeded => Self::VmRuntimeFault {
                kind: VmFaultKind::StepLimitExceeded,
            },
            quay_vm::RuntimeError::InvalidShiftAmount => Self::VmRuntimeFault {
                kind: VmFaultKind::InvalidShiftAmount,
            },
            // Everything else is a protocol-bug-class fault (validator should
            // have caught it). Surface raw so the caller can debug.
            other => Self::Vm(other),
        }
    }
}

impl ClientError {
    /// Decode an on-chain `Custom(c)` code into a typed `ClientError`. Returns
    /// `None` for codes outside the VM namespace so callers can fall through
    /// to whatever error handling they had before.
    ///
    /// Usage: aggregators / wallets that receive a `ProgramError` from a
    /// failed swap tx can call this to get a typed reason for retry / UX
    /// decisions without parsing the on-chain error code themselves.
    pub fn from_custom_code(code: u32) -> Option<Self> {
        let high = code >> 8;
        let low = (code & 0xFF) as u8;
        match high {
            0xC0 => Some(Self::VmReject {
                reason: VmRejectReason::from_byte(low),
            }),
            0xCE => VmFaultKind::from_byte(low).map(|kind| Self::VmRuntimeFault { kind }),
            _ if code == 0xDEAD => Some(Self::VmProtocolBug),
            _ => None,
        }
    }
}

pub type Result<T, E = ClientError> = core::result::Result<T, E>;
