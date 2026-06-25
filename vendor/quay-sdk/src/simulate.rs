//! Off-chain swap simulator.
//!
//! Bit-identical to the on-chain `swap` ix in `quay-program`. Routers and MM
//! dashboards call `simulate_swap` to predict execution price + protocol fee
//! split for a `(Strategy, MarketMaker, Quotes, GlobalConfig)` set — without
//! sending a tx.
//!
//! Stateful curves (`Op::StoreI64`) are supported: the simulator clones the
//! strategy's userspace, lets the curve mutate the clone, and returns the
//! post-mutation bytes in [`SwapSimulation::userspace_post`].

use quay_vm::{evaluate, Inputs, TxContext, CURRENT_DSL_VERSION};

use crate::consts::{FEE_BPS_DENOM, SIDE_SELL_BASE};
use crate::error::{ClientError, Result};
use crate::state::{GlobalConfig, MarketMakerHeader, QuotesHeader, StrategyHeader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapSimulation {
    /// OUT atoms the taker receives — the curve's `amount_out` for the net
    /// input (`amount_in - protocol_cut`).
    pub out_to_taker: u64,
    /// Protocol fee skimmed from the input, in IN-token atoms. `0` when the
    /// strategy's `protocol_fee_bps` is `0`. Accrues to the IN-side asset.
    pub protocol_cut: u64,
    /// Userspace bytes after the curve mutated them.
    pub userspace_post: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapSimulationInputs<'a> {
    pub strategy_data: &'a [u8],
    pub market_maker_data: &'a [u8],
    pub quotes_data: &'a [u8],
    pub global_config_data: &'a [u8],
    /// Solana slot the simulation runs against.
    pub current_slot: u64,
    /// Unix seconds at simulation time.
    pub current_unix_sec: i64,
    pub side: u8,
    pub amount_in: u64,
    pub min_amount_out: u64,
    /// Decimals of the base / quote mints (caller reads the mint accounts).
    pub base_decimals: u8,
    pub quote_decimals: u8,
}

/// Simulate a `swap` against a strategy. Mirrors `quay-program`'s on-chain
/// `swap` path step for step.
///
/// Evaluates under [`TxContext::DIRECT`] — quoting happens before the
/// transaction exists, so context-gated curves price their benign
/// "direct taker" branch; that is the quote-accuracy contract (gates
/// widen, never tighten). To evaluate a curve under a synthetic context,
/// call `quay_vm::evaluate` with raw `Inputs` directly.
pub fn simulate_swap(inputs: SwapSimulationInputs<'_>) -> Result<SwapSimulation> {
    if inputs.side > 1 {
        return Err(ClientError::InvalidInput("side must be 0 or 1"));
    }
    let cfg = *GlobalConfig::try_from_account(inputs.global_config_data)?;
    if cfg.protocol_halted != 0 {
        return Err(ClientError::ProtocolHalted);
    }
    if cfg.swap_halted != 0 {
        return Err(ClientError::SwapHalted);
    }

    let strategy = *StrategyHeader::try_from_account(inputs.strategy_data)?;
    if strategy.frozen != 0 || strategy.frozen_admin != 0 {
        return Err(ClientError::StrategyFrozen);
    }
    if strategy.dsl_version > CURRENT_DSL_VERSION {
        return Err(ClientError::UnsupportedDslVersion);
    }

    let mm = *MarketMakerHeader::try_from_account(inputs.market_maker_data)?;
    if mm.frozen != 0 || mm.frozen_admin != 0 {
        return Err(ClientError::MarketMakerFrozen);
    }
    if mm.halted_admin != 0 {
        return Err(ClientError::MarketMakerHalted);
    }

    let inventory_base = mm.asset(inputs.market_maker_data, strategy.base_id)?.amount;
    let inventory_quote = mm.asset(inputs.market_maker_data, strategy.quote_id)?.amount;

    let quotes_hdr = *QuotesHeader::try_from_account(inputs.quotes_data)?;
    if quotes_hdr.updated_ts == 0 {
        return Err(ClientError::QuotesNotPublished);
    }
    let quotes_buf = QuotesHeader::read_all_slots(inputs.quotes_data)?;

    let bytecode = strategy.bytecode(inputs.strategy_data)?;
    let userspace_src = strategy.userspace(inputs.strategy_data)?;

    let protocol_fee_bps = u16::min(strategy.protocol_fee_bps, 10_000);

    // Skim the protocol fee from the input (IN-token atoms); the curve prices
    // the net input. Mirrors `swap.rs::protocol_fee` + the fee skim.
    let protocol_cut = u64::try_from(
        (u128::from(inputs.amount_in) * u128::from(protocol_fee_bps)) / u128::from(FEE_BPS_DENOM),
    )
    .map_err(|_| ClientError::SwapMathOverflow)?;
    let net_in = inputs
        .amount_in
        .checked_sub(protocol_cut)
        .ok_or(ClientError::SwapMathOverflow)?;

    // Single run on a clone of userspace so the post-state can be returned.
    let mut userspace_clone = userspace_src.to_vec();
    let inputs_vm = Inputs {
        quotes: &quotes_buf,
        userspace: &mut userspace_clone,
        size: net_in,
        side: inputs.side,
        current_slot: inputs.current_slot,
        inventory_base,
        inventory_quote,
        current_unix_sec: inputs.current_unix_sec.max(0) as u64,
        base_decimals: inputs.base_decimals,
        quote_decimals: inputs.quote_decimals,
        quotes_timestamp_ns: quotes_hdr.updated_ts,
        last_update_slot: strategy.last_update_slot,
        tx: TxContext::DIRECT,
    };
    let out_to_taker = evaluate(bytecode, inputs_vm)?.amount_out();
    if out_to_taker == 0 {
        return Err(ClientError::ZeroOutputSwap {
            amount_in: inputs.amount_in,
        });
    }
    if out_to_taker < inputs.min_amount_out {
        return Err(ClientError::SlippageExceeded {
            got: out_to_taker,
            min: inputs.min_amount_out,
        });
    }

    // Mirror `settle_inventory`'s structural solvency gate: on-chain, the
    // OUT-side asset entry is debited `out_to_taker` with `checked_sub`, the
    // IN side credited `net_in` (and `protocol_cut` accrued as fees) — a swap
    // the inventory can't cover fails at settlement. The full `amount_in` lands
    // in the IN vault. Without this gate the simulator quotes fills the program
    // will refuse.
    let (inventory_in, inventory_out) = if inputs.side == SIDE_SELL_BASE {
        (inventory_base, inventory_quote)
    } else {
        (inventory_quote, inventory_base)
    };
    if out_to_taker > inventory_out {
        return Err(ClientError::InsufficientInventory {
            needed: out_to_taker,
            available: inventory_out,
        });
    }
    if inventory_in.checked_add(net_in).is_none() {
        return Err(ClientError::SwapMathOverflow);
    }

    Ok(SwapSimulation {
        out_to_taker,
        protocol_cut,
        userspace_post: userspace_clone,
    })
}
