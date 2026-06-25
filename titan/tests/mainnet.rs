//! Live mainnet smoke test: drive `QuayVenue` against real Quay strategies
//! over JSON-RPC, exactly as a router would — `from_account` ->
//! `get_required_pubkeys_for_update` -> fetch -> `update_state` -> `quote`.
//!
//! Ignored by default (needs network + a mainnet RPC). Run with:
//!
//! ```bash
//! RPC_URL='https://...' cargo test -p quay-aggregator-titan \
//!     --test mainnet -- --ignored --nocapture
//! ```
//!
//! It also wraps one `quote()` in `assert_no_alloc`, proving the no-heap
//! guarantee holds on a *real* strategy — including the live USDT/USDC one
//! that carries a non-empty `userspace` (the curve the old stateless gate
//! would have refused).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]

use std::collections::HashMap;
use std::str::FromStr;

use assert_no_alloc::{assert_no_alloc, AllocDisabler};
use async_trait::async_trait;
use quay_aggregator_titan::{QuayVenue, QUAY_PROGRAM_ID};
use solana_account::Account;
use solana_pubkey::Pubkey;
use titan_integration_template::account_caching::{AccountCacheError, AccountsCache};
use titan_integration_template::trading_venue::{
    FromAccount, QuoteRequest, SwapType, TradingVenue,
};

#[global_allocator]
static A: AllocDisabler = AllocDisabler;

/// `getProgramAccounts` filter for Strategy accounts (discriminator `0x03` at
/// offset 0). Base58 of the single byte `0x03` is `"4"`.
const STRATEGY_DISCRIMINATOR_B58: &str = "4";

struct MapCache(HashMap<Pubkey, Account>);

#[async_trait]
impl AccountsCache for MapCache {
    async fn get_account(&self, k: &Pubkey) -> Result<Option<Account>, AccountCacheError> {
        Ok(self.0.get(k).cloned())
    }
    async fn get_accounts(
        &self,
        ks: &[Pubkey],
    ) -> Result<Vec<Option<Account>>, AccountCacheError> {
        Ok(ks.iter().map(|k| self.0.get(k).cloned()).collect())
    }
}

fn rpc_call(url: &str, body: serde_json::Value) -> serde_json::Value {
    ureq::post(url)
        .send_json(body)
        .expect("rpc request")
        .into_json()
        .expect("rpc json")
}

fn account_from_rpc(v: &serde_json::Value) -> Option<Account> {
    use base64::Engine as _;
    if v.is_null() {
        return None;
    }
    let data = base64::engine::general_purpose::STANDARD
        .decode(v["data"][0].as_str().expect("data b64"))
        .expect("b64");
    Some(Account {
        lamports: v["lamports"].as_u64().unwrap_or(0),
        data,
        owner: Pubkey::from_str(v["owner"].as_str().expect("owner")).expect("owner pk"),
        executable: false,
        rent_epoch: 0,
    })
}

/// `getProgramAccounts` → all Strategy pubkeys owned by the Quay program.
fn fetch_strategy_keys(url: &str) -> Vec<Pubkey> {
    let resp = rpc_call(
        url,
        serde_json::json!({
            "jsonrpc":"2.0","id":1,"method":"getProgramAccounts",
            "params":[QUAY_PROGRAM_ID.to_string(), {
                "encoding":"base64",
                "dataSlice":{"offset":0,"length":0},
                "filters":[{"memcmp":{"offset":0,"bytes":STRATEGY_DISCRIMINATOR_B58}}]
            }]
        }),
    );
    resp["result"]
        .as_array()
        .expect("result array")
        .iter()
        .map(|a| Pubkey::from_str(a["pubkey"].as_str().unwrap()).unwrap())
        .collect()
}

/// `getMultipleAccounts` for a key set.
fn fetch_accounts(url: &str, keys: &[Pubkey]) -> HashMap<Pubkey, Account> {
    let key_strs: Vec<String> = keys.iter().map(ToString::to_string).collect();
    let resp = rpc_call(
        url,
        serde_json::json!({
            "jsonrpc":"2.0","id":1,"method":"getMultipleAccounts",
            "params":[key_strs, {"encoding":"base64"}]
        }),
    );
    let values = resp["result"]["value"].as_array().expect("value array");
    keys.iter()
        .zip(values)
        .filter_map(|(k, v)| account_from_rpc(v).map(|a| (*k, a)))
        .collect()
}

#[tokio::test]
#[ignore = "needs network + RPC_URL"]
async fn quote_live_strategies() {
    let Ok(url) = std::env::var("RPC_URL") else {
        eprintln!("skipping: set RPC_URL to a mainnet endpoint");
        return;
    };

    let strategies = fetch_strategy_keys(&url);
    assert!(!strategies.is_empty(), "no Quay strategies found on-chain");
    println!("found {} strategy account(s)", strategies.len());

    // Sweep in 6-decimal atoms (USDC/USDT): 1, 100, 10k units.
    let amounts = [1_000_000u64, 100_000_000, 10_000_000_000];

    for strat_key in strategies {
        let strat_acc = fetch_accounts(&url, &[strat_key])
            .remove(&strat_key)
            .expect("strategy account");

        let mut venue = match QuayVenue::from_account(&strat_key, &strat_acc) {
            Ok(v) => v,
            Err(e) => {
                println!("  {strat_key}: from_account failed: {e}");
                continue;
            }
        };

        let needed = venue
            .get_required_pubkeys_for_update()
            .expect("required pubkeys");
        let cache = MapCache(fetch_accounts(&url, &needed));
        if let Err(e) = venue.update_state(&cache).await {
            println!("  {strat_key}: update_state failed: {e}");
            continue;
        }

        let tokens = venue.get_token_info();
        let base = tokens.first().map(|t| t.pubkey).unwrap_or_default();
        let quote = tokens.get(1).map(|t| t.pubkey).unwrap_or_default();
        println!(
            "  {strat_key}: initialized={} base={} quote={}",
            venue.initialized(),
            base,
            quote,
        );
        if !venue.initialized() {
            continue;
        }

        // Prove the no-heap guarantee on this live strategy (which may carry a
        // non-empty userspace) before the readable sweep below.
        let probe = QuoteRequest {
            input_mint: base,
            output_mint: quote,
            amount: amounts[0],
            swap_type: SwapType::ExactIn,
        };
        let _ = assert_no_alloc(|| venue.quote(probe.clone()));

        for amount in amounts {
            for (label, input_mint, output_mint) in
                [("SELL_BASE", base, quote), ("BUY_BASE", quote, base)]
            {
                let req = QuoteRequest {
                    input_mint,
                    output_mint,
                    amount,
                    swap_type: SwapType::ExactIn,
                };
                match venue.quote(req) {
                    Ok(q) => println!(
                        "    {label} in {amount} -> out {} (not_enough_liq={})",
                        q.expected_output, q.not_enough_liquidity
                    ),
                    Err(e) => println!("    {label} in {amount} -> refused: {e}"),
                }
            }
        }
    }
}
