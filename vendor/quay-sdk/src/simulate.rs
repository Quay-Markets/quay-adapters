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

use quay_vm::{evaluate, Inputs, RuntimeError, TxContext, CURRENT_DSL_VERSION};

use crate::consts::{FEE_BPS_DENOM, PRICE_SCALE, SIDE_BUY_BASE, SIDE_SELL_BASE};
use crate::error::{ClientError, Result};
use crate::state::{GlobalConfig, MarketMakerHeader, QuotesHeader, StrategyHeader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapSimulation {
    /// OUT atoms the taker receives at `exec_price`.
    pub out_to_taker: u64,
    /// Protocol's slice of the half-spread, in OUT atoms. `0` when the
    /// strategy's `protocol_fee_bps` is `0`.
    pub protocol_cut: u64,
    /// Curve's exec_price for the requested side (Q24).
    pub exec_price: i64,
    /// Opposite-side exec_price (Q24). `None` when no fee applies (no
    /// opposite-side sim was run).
    pub opp_exec_price: Option<i64>,
    /// Userspace bytes after the real-side curve mutated them.
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
    /// Base-vault token balance in atoms (caller reads the vault token
    /// account). Exposed to bytecode via `LoadVaultBase`.
    pub vault_base_atoms: u64,
    /// Quote-vault token balance in atoms. Exposed via `LoadVaultQuote`.
    pub vault_quote_atoms: u64,
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
    let arena_timestamp_sec = (quotes_hdr.updated_ts / 1_000_000_000) as i64;

    let bytecode = strategy.bytecode(inputs.strategy_data)?;
    let userspace_src = strategy.userspace(inputs.strategy_data)?;

    let inv_real = inv_for(inputs.side, inventory_base, inventory_quote);
    let inv_opp = inv_for(opposite(inputs.side), inventory_base, inventory_quote);
    let protocol_fee_bps = u16::min(strategy.protocol_fee_bps, 10_000);

    macro_rules! mk_inputs {
        ($us:expr, $inv:expr, $side:expr) => {
            Inputs {
                quotes: &quotes_buf,
                userspace: $us,
                inv: $inv,
                size: inputs.amount_in as i64,
                side: i64::from($side),
                current_slot: inputs.current_slot as i64,
                vault_base_atoms: i64::try_from(inputs.vault_base_atoms).unwrap_or(i64::MAX),
                vault_quote_atoms: i64::try_from(inputs.vault_quote_atoms).unwrap_or(i64::MAX),
                inventory_base,
                inventory_quote,
                current_unix_sec: inputs.current_unix_sec,
                base_decimals: inputs.base_decimals,
                quote_decimals: inputs.quote_decimals,
                arena_timestamp_sec,
                last_update_slot: strategy.last_update_slot,
                tx: TxContext::DIRECT,
            }
        };
    }

    // Opposite-side sim first (on a clone). Skipped when no fee applies.
    let opp_exec_price: Option<i64> = if protocol_fee_bps > 0 {
        let mut opp_us = userspace_src.to_vec();
        match evaluate(bytecode, mk_inputs!(&mut opp_us, inv_opp, opposite(inputs.side))) {
            Ok(result) => Some(result.exec_price()),
            // One-sided curve: no two-sided spread → zero protocol fee on
            // this swap. Real VM faults (`Err(e)` below) still abort.
            Err(RuntimeError::Reject(_)) => None,
            Err(e) => return Err(e.into()),
        }
    } else {
        None
    };

    // Real-side run on its own clone (so the post-state can be returned).
    let mut userspace_clone = userspace_src.to_vec();
    let exec_price = evaluate(bytecode, mk_inputs!(&mut userspace_clone, inv_real, inputs.side))?
        .exec_price();
    if exec_price <= 0 {
        return Err(ClientError::InvalidExecPrice(exec_price));
    }

    let out_to_taker = apply_price(inputs.amount_in, exec_price, inputs.side)?;
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

    let protocol_cut: u64 = if let Some(opp) = opp_exec_price {
        if opp <= 0 {
            return Err(ClientError::InvalidExecPrice(opp));
        }
        let opp_out = apply_price(inputs.amount_in, opp, inputs.side)?;
        let spread = (i128::from(opp_out) - i128::from(out_to_taker)).unsigned_abs();
        let half = spread / 2;
        let scaled = half
            .checked_mul(u128::from(protocol_fee_bps))
            .ok_or(ClientError::SwapMathOverflow)?;
        u64::try_from(scaled / u128::from(FEE_BPS_DENOM))
            .map_err(|_| ClientError::SwapMathOverflow)?
    } else {
        0
    };

    Ok(SwapSimulation {
        out_to_taker,
        protocol_cut,
        exec_price,
        opp_exec_price,
        userspace_post: userspace_clone,
    })
}

fn opposite(side: u8) -> u8 {
    if side == SIDE_SELL_BASE {
        SIDE_BUY_BASE
    } else {
        SIDE_SELL_BASE
    }
}

fn inv_for(side: u8, base: u64, quote: u64) -> i64 {
    let v = if side == SIDE_SELL_BASE { base } else { quote };
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// OUT atoms for `amount_in` at `exec_price` (Q24).
fn apply_price(amount_in: u64, exec_price: i64, side: u8) -> Result<u64> {
    let a = i128::from(amount_in);
    let p = i128::from(exec_price);
    let out = match side {
        SIDE_SELL_BASE => a.checked_mul(p).and_then(|v| v.checked_div(PRICE_SCALE)),
        _ => a.checked_mul(PRICE_SCALE).and_then(|v| v.checked_div(p)),
    }
    .ok_or(ClientError::SwapMathOverflow)?;
    if out < 0 || out > i128::from(u64::MAX) {
        return Err(ClientError::SwapMathOverflow);
    }
    Ok(out as u64)
}
