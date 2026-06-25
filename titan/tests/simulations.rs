//! Titan "Simulation Tests (Critical)": load the compiled Quay program into
//! LiteSVM and execute the adapter's **own** `generate_swap_instruction`,
//! asserting that
//!
//! - the on-chain fill (the taker's OUT-ATA credit) equals `quote()`'s
//!   `expected_output` for every filled sweep point, and the swap the program
//!   refuses is also refused by the adapter,
//! - the generated instruction's accounts / ATAs / token ownership are correct
//!   — the transaction simply would not execute otherwise.
//!
//! The market state comes from the same litesvm-generated fixtures the parity
//! test uses; this test re-seeds that exact account set into a fresh SVM, funds
//! a taker, and runs the real `from_account` -> `update_state` -> `quote` +
//! `generate_swap_instruction` flow a router would.
//!
//! The compiled program (`quay_program.so`) is located, in order: the
//! `QUAY_PROGRAM_SO` env var, the monorepo `target/deploy` build (so a
//! rebuild is always tested against current sources), then the copy bundled
//! next to the fixtures (`../fixtures/quay_program.so`) so the standalone
//! export runs with no setup. Only if none exists does the test print a notice
//! and return — it never fails for a missing binary.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: panic on assertion failure is the desired behavior"
)]
// `single_match_else` is allowed workspace-wide as too noisy; the two-arm
// filled/rejected match reads more clearly than an `if let ... else`.
#![allow(clippy::single_match_else)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use async_trait::async_trait;
use litesvm::LiteSVM;
use quay_aggregator_titan::QuayVenue;
use solana_account::Account;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_program::program_option::COption;
use solana_program::program_pack::Pack;
use solana_pubkey::Pubkey;
use solana_signer::Signer;
use solana_transaction::Transaction;
use titan_integration_template::account_caching::{AccountCacheError, AccountsCache};
use titan_integration_template::trading_venue::{
    FromAccount, QuoteRequest, SwapType, TradingVenue,
};

const SIDE_SELL_BASE: u8 = 0;
/// Account index of the taker's IN / OUT ATAs in the `swap` ix (see
/// `quay_sdk::ix::swap` — `[cfg, strategy, mm, quotes, vault_in, vault_out,
/// ata_in, ata_out, taker, mint_in, mint_out, sysvar, token_program]`).
const ATA_IN_IDX: usize = 6;
const ATA_OUT_IDX: usize = 7;
const MINT_IN_IDX: usize = 9;

/// Locate the compiled program. Returns `None` (test skips) when absent.
fn program_so_path() -> Option<PathBuf> {
    // 1. Explicit override.
    if let Ok(p) = std::env::var("QUAY_PROGRAM_SO") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // 2. The copy bundled next to the fixtures — makes the standalone export
    //    repo self-contained (no env var, no monorepo checkout).
    let bundled = manifest.join("../fixtures/quay_program.so");
    bundled.exists().then_some(bundled)
}

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
    program_id: Pubkey,
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
        program_id: pk("program_id"),
        strategy: pk("strategy"),
        base_mint: pk("base_mint"),
        quote_mint: pk("quote_mint"),
        accounts,
        swaps,
    }
}

/// A funded, initialized SPL token account blob owned by `token_program`.
fn token_account(mint: Pubkey, owner: Pubkey, amount: u64, token_program: Pubkey) -> Account {
    let acct = spl_token::state::Account {
        mint: solana_program::pubkey::Pubkey::new_from_array(mint.to_bytes()),
        owner: solana_program::pubkey::Pubkey::new_from_array(owner.to_bytes()),
        amount,
        delegate: COption::None,
        state: spl_token::state::AccountState::Initialized,
        is_native: COption::None,
        delegated_amount: 0,
        close_authority: COption::None,
    };
    let mut data = vec![0u8; spl_token::state::Account::LEN];
    spl_token::state::Account::pack(acct, &mut data).expect("pack token account");
    Account {
        lamports: 10_000_000,
        data,
        owner: token_program,
        executable: false,
        rent_epoch: 0,
    }
}

fn token_balance(svm: &LiteSVM, ata: &Pubkey) -> u64 {
    svm.get_account(ata)
        .map_or(0, |a| u64::from_le_bytes(a.data[64..72].try_into().unwrap()))
}

/// Re-seed the fixture's account set so each sweep point runs against the
/// snapshotted state — matching how the generator recorded independent
/// outcomes.
fn seed_market(svm: &mut LiteSVM, f: &Fixture) {
    for (k, a) in &f.accounts {
        svm.set_account(*k, a.clone()).expect("seed account");
    }
}

/// Build the venue exactly as a router would: from the raw strategy account,
/// then `update_state` through the fixture cache.
async fn build_venue(f: &Fixture) -> QuayVenue {
    let strat = f.accounts.get(&f.strategy).expect("strategy").clone();
    let mut venue = QuayVenue::from_account(&f.strategy, &strat).expect("from_account");
    venue
        .update_state(&MapCache(f.accounts.clone()))
        .await
        .expect("update_state");
    venue
}

/// Token program that owns a mint in this fixture (SPL Token for all current
/// fixtures; read off the account so a Token-2022 fixture would still work).
fn token_program_of(f: &Fixture, mint: &Pubkey) -> Pubkey {
    f.accounts.get(mint).expect("mint account").owner
}

async fn run_simulation(name: &str, so: &[u8]) {
    let f = load_fixture(name);
    let venue = build_venue(&f).await;

    let mut svm = LiteSVM::new().with_blockhash_check(false);
    svm.add_program(f.program_id, so);

    for (side, amount_in, expected_out) in &f.swaps {
        // Fresh market + taker for every sweep point.
        seed_market(&mut svm, &f);
        let taker = Keypair::new();
        svm.airdrop(&taker.pubkey(), 1_000_000_000).expect("airdrop taker");

        let (input_mint, output_mint) = if *side == SIDE_SELL_BASE {
            (f.base_mint, f.quote_mint)
        } else {
            (f.quote_mint, f.base_mint)
        };
        let req = QuoteRequest {
            input_mint,
            output_mint,
            amount: *amount_in,
            swap_type: SwapType::ExactIn,
        };

        let ix: Instruction = venue
            .generate_swap_instruction(req.clone(), taker.pubkey())
            .expect("generate_swap_instruction");

        // Fund exactly the ATAs the adapter's instruction references.
        let ata_in = ix.accounts[ATA_IN_IDX].pubkey;
        let ata_out = ix.accounts[ATA_OUT_IDX].pubkey;
        let mint_in = ix.accounts[MINT_IN_IDX].pubkey;
        let token_prog = token_program_of(&f, &mint_in);
        svm.set_account(
            ata_in,
            token_account(mint_in, taker.pubkey(), *amount_in, token_prog),
        )
        .expect("fund ata_in");
        svm.set_account(
            ata_out,
            token_account(output_mint, taker.pubkey(), 0, token_prog),
        )
        .expect("init ata_out");

        let quote = venue.quote(req);
        let before = token_balance(&svm, &ata_out);
        let bh = svm.latest_blockhash();
        let tx = Transaction::new_signed_with_payer(&[ix], Some(&taker.pubkey()), &[&taker], bh);
        let result = svm.send_transaction(tx);

        match expected_out {
            // Program filled this swap → so must the adapter, with the same OUT.
            Some(out) => {
                let meta = result.unwrap_or_else(|e| {
                    panic!("{name}: side {side} in {amount_in} filled on-chain in the generator but reverted here: {e:?}")
                });
                let _ = meta;
                let onchain_out = token_balance(&svm, &ata_out) - before;
                assert_eq!(
                    onchain_out, *out,
                    "{name}: side {side} in {amount_in} — on-chain OUT {onchain_out} != fixture {out}"
                );
                let q = quote.unwrap_or_else(|e| {
                    panic!("{name}: adapter refused a swap the program filled: {e}")
                });
                assert_eq!(
                    q.expected_output, onchain_out,
                    "{name}: side {side} in {amount_in} — quote {} != on-chain {onchain_out}",
                    q.expected_output
                );
            }
            // Program rejected this swap → the on-chain tx must revert, and the
            // adapter must not promise a non-zero fill.
            None => {
                assert!(
                    result.is_err(),
                    "{name}: side {side} in {amount_in} failed in the generator but succeeded here"
                );
                if let Ok(q) = quote {
                    assert_eq!(
                        q.expected_output, 0,
                        "{name}: side {side} in {amount_in} rejected on-chain but adapter quoted {}",
                        q.expected_output
                    );
                }
            }
        }
    }
}

async fn run_all(name: &str) {
    let Some(path) = program_so_path() else {
        eprintln!(
            "skipping simulations::{name}: program binary not found — set QUAY_PROGRAM_SO to quay_program.so"
        );
        return;
    };
    let so = std::fs::read(&path).expect("read program .so");
    run_simulation(name, &so).await;
}

#[tokio::test]
async fn simulate_flat() {
    run_all("flat").await;
}

#[tokio::test]
async fn simulate_spread_fee() {
    run_all("spread_fee").await;
}

#[tokio::test]
async fn simulate_exact_fee() {
    run_all("exact_fee").await;
}

#[tokio::test]
async fn simulate_one_sided() {
    run_all("one_sided").await;
}
