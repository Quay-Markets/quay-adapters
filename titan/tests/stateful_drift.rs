//! End-to-end stateful drift, through the real titan adapter.
//!
//! Splices the "even swaps widen the spread" curve (see
//! `vendor/quay-vm/tests/stateful_spread.rs`) into the `flat` fixture's
//! Strategy and drives it through `QuayVenue::quote()` — the exact path a
//! Titan router runs.
//!
//! It demonstrates two things the adapter docs assert (`titan/src/lib.rs`
//! "All curves, allocation-free"):
//!
//!   1. `quote()` prices a *snapshot*: it copies userspace into a scratch
//!      buffer, lets the curve mutate the copy, then discards it. So calling
//!      `quote()` twice with no intervening `update_state` returns the *same*
//!      price — the adapter never advances the curve's state itself.
//!   2. The on-chain fill does persist the mutation. So once real state moves
//!      (modelled here by editing the Strategy account's userspace and
//!      re-running `update_state`), the next quote drifts to a different price.
//!      That gap is what the route's `min_amount_out` slippage guard bounds.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]

use std::collections::HashMap;
use std::str::FromStr;

use async_trait::async_trait;
use quay_aggregator_titan::QuayVenue;
use quay_sdk::dsl::BytecodeBuilder;
use solana_account::Account;
use solana_pubkey::Pubkey;
use titan_integration_template::account_caching::{AccountCacheError, AccountsCache};
use titan_integration_template::trading_venue::{
    FromAccount, QuoteRequest, SwapType, TradingVenue,
};

// ── curve constants (mirror quay-sdk's stateful_spread.rs) ──────────────────
const BPS_DENOM: i64 = 10_000;
const NARROW_SPREAD_BPS: i64 = 10;
const WIDE_SPREAD_BPS: i64 = 100;
const SELL_COUNT_OFF: u32 = 0;
const BUY_COUNT_OFF: usize = 8;

// ── strategy account layout ────────────────────────────────────────────────
const HEADER_LEN: usize = 192;
const BYTECODE_LEN_OFF: usize = 144;
const USERSPACE_LEN_OFF: usize = 148;
const USERSPACE_BYTES: usize = 16;

/// The even-widens-spread curve, built with `quay_sdk::dsl::BytecodeBuilder`
/// (identical semantics to `vendor/quay-sdk/tests/stateful_spread.rs`).
fn build_spread_curve() -> Vec<u8> {
    fn bump(b: BytecodeBuilder, off: u32) -> BytecodeBuilder {
        b.load_i64(off).load_const(1).add().dup().store_i64(off)
    }

    let (b, to_sell) = BytecodeBuilder::new().load_side().load_const(0).start_if_eq();
    let b = bump(b, BUY_COUNT_OFF as u32);
    let (b, to_spread) = b.start_jmp();
    let b = bump(b.patch(to_sell), SELL_COUNT_OFF);

    let b = b.patch(to_spread).load_const(2).mod_op().load_const(0);
    let (b, to_even) = b.start_if_eq();
    let b = b.load_const(NARROW_SPREAD_BPS);
    let (b, to_apply) = b.start_jmp();
    let b = b.patch(to_even).load_const(WIDE_SPREAD_BPS);

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

/// Replace the strategy's bytecode with `build_spread_curve()` and give it a
/// 16-byte (zeroed) userspace. Returns the absolute offset where userspace
/// begins, so the test can later edit a counter in place.
fn splice_spread_curve(strat: &mut Account) -> usize {
    let bc = build_spread_curve();
    let mut data = strat.data[..HEADER_LEN].to_vec();
    data.extend_from_slice(&bc);
    let userspace_start = data.len();
    data.extend_from_slice(&[0u8; USERSPACE_BYTES]);
    data[BYTECODE_LEN_OFF..BYTECODE_LEN_OFF + 4].copy_from_slice(&(bc.len() as u32).to_le_bytes());
    data[USERSPACE_LEN_OFF..USERSPACE_LEN_OFF + 4]
        .copy_from_slice(&(USERSPACE_BYTES as u32).to_le_bytes());
    strat.data = data;
    userspace_start
}

/// `out = size * (10000 - spread_bps) / 10000` — the curve's pricing.
fn expected(size: u64, spread_bps: i64) -> u64 {
    (size as i128 * (BPS_DENOM - spread_bps) as i128 / BPS_DENOM as i128) as u64
}

// ── fixture harness (same shape as parity.rs / test_construction.rs) ────────
struct MapCache(HashMap<Pubkey, Account>);

#[async_trait]
impl AccountsCache for MapCache {
    async fn get_account(&self, pubkey: &Pubkey) -> Result<Option<Account>, AccountCacheError> {
        Ok(self.0.get(pubkey).cloned())
    }
    async fn get_accounts(
        &self,
        pubkeys: &[Pubkey],
    ) -> Result<Vec<Option<Account>>, AccountCacheError> {
        Ok(pubkeys.iter().map(|k| self.0.get(k).cloned()).collect())
    }
}

struct Fixture {
    strategy: Pubkey,
    base_mint: Pubkey,
    quote_mint: Pubkey,
    accounts: HashMap<Pubkey, Account>,
    swaps: Vec<(u8, u64, Option<u64>)>,
}

fn load_fixture(name: &str) -> Fixture {
    use base64::Engine as _;
    let path = format!("{}/../fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("valid fixture JSON");
    let pk = |f: &str| Pubkey::from_str(v[f].as_str().expect(f)).expect(f);
    let accounts = v["accounts"]
        .as_array()
        .expect("accounts")
        .iter()
        .map(|a| {
            let key = Pubkey::from_str(a["pubkey"].as_str().unwrap()).unwrap();
            let account = Account {
                lamports: a["lamports"].as_u64().unwrap(),
                data: base64::engine::general_purpose::STANDARD
                    .decode(a["data_b64"].as_str().unwrap())
                    .unwrap(),
                owner: Pubkey::from_str(a["owner"].as_str().unwrap()).unwrap(),
                executable: false,
                rent_epoch: 0,
            };
            (key, account)
        })
        .collect();
    let swaps = v["swaps"]
        .as_array()
        .expect("swaps")
        .iter()
        .map(|s| {
            (
                s["side"].as_u64().unwrap() as u8,
                s["amount_in"].as_u64().unwrap(),
                s["out"].as_u64(),
            )
        })
        .collect();
    Fixture {
        strategy: pk("strategy"),
        base_mint: pk("base_mint"),
        quote_mint: pk("quote_mint"),
        accounts,
        swaps,
    }
}

/// A buy-base request: input = quote mint, output = base mint → SIDE_BUY_BASE.
fn buy_request(f: &Fixture, amount: u64) -> QuoteRequest {
    QuoteRequest {
        input_mint: f.quote_mint,
        output_mint: f.base_mint,
        amount,
        swap_type: SwapType::ExactIn,
    }
}

#[tokio::test]
async fn quote_prices_a_snapshot_then_drifts_when_state_advances() {
    let mut f = load_fixture("flat");

    // Splice the stateful spread curve into the strategy (userspace zeroed).
    let mut strat = f.accounts.get(&f.strategy).expect("strategy").clone();
    let userspace_start = splice_spread_curve(&mut strat);
    f.accounts.insert(f.strategy, strat.clone());

    // Pick a buy amount the fixture's inventory is known to cover (an on-chain
    // 1:1 buy succeeded for it; our spread output is strictly smaller, so it is
    // covered too).
    // Largest successful buy: big enough that the spread haircut doesn't floor
    // to zero, and (since on-chain it filled 1:1) inventory covers our smaller
    // spread output too.
    let amount = f
        .swaps
        .iter()
        .filter(|(side, _, out)| *side == 1 && out.is_some())
        .map(|(_, amt, _)| *amt)
        .max()
        .expect("a successful buy row in the flat fixture");

    // Build the venue and run the full from_account → update_state → quote flow.
    let mut venue = QuayVenue::from_account(&f.strategy, &strat).expect("from_account");
    venue
        .update_state(&MapCache(f.accounts.clone()))
        .await
        .expect("update_state");
    assert!(venue.initialized(), "spliced strategy should be routable");

    // (1) First quote: buy count goes 0 → 1 (odd) → NARROW spread.
    let q1 = venue.quote(buy_request(&f, amount)).expect("quote 1").expected_output;
    assert_eq!(q1, expected(amount, NARROW_SPREAD_BPS), "first quote is the narrow price");

    // (2) Quote again with no update_state: the adapter re-prices the same
    //     snapshot (mutation was discarded), so the price is identical — it did
    //     NOT advance the counter to 2.
    let q2 = venue.quote(buy_request(&f, amount)).expect("quote 2").expected_output;
    assert_eq!(q2, q1, "repeated quote prices the same snapshot — no self-advance");

    // (3) Model an on-chain buy landing: bump the Strategy account's userspace
    //     buy counter to 1, then re-run update_state (a fresh RPC fetch). The
    //     next buy is now count 2 (even) → WIDE spread.
    let mut advanced = strat.clone();
    advanced.data[userspace_start + BUY_COUNT_OFF..userspace_start + BUY_COUNT_OFF + 8]
        .copy_from_slice(&1i64.to_le_bytes());
    f.accounts.insert(f.strategy, advanced);
    venue
        .update_state(&MapCache(f.accounts.clone()))
        .await
        .expect("update_state after advance");

    let q3 = venue.quote(buy_request(&f, amount)).expect("quote 3").expected_output;
    assert_eq!(q3, expected(amount, WIDE_SPREAD_BPS), "post-advance quote is the wide price");
    assert!(q3 < q1, "drift: once state moved, the fill is worse than the earlier quote");
}
