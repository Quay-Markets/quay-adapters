# quay-aggregator-titan

Titan `TradingVenue` adapter for the Quay program, for routers using
`titan-integration-template`. Depends on the vendored `quay-sdk` in this
repo (see the root README).

```toml
quay-aggregator-titan = { git = "<this repo>" }
```

```rust
use quay_aggregator_titan::QuayVenue;
use titan_integration_template::trading_venue::{FromAccount, TradingVenue};

let mut venue = QuayVenue::from_account(&strategy_pubkey, &strategy_account)?;
// Titan's router pulls these every slot via `update_state`:
//   venue.get_required_pubkeys_for_update()? ==
//     [strategy, mm, quotes, global_config, base_mint, quote_mint,
//      vault_base, vault_quote, Clock sysvar]
venue.update_state(&cache).await?;
```

Each `QuayVenue` instance represents **one Strategy** — a pricing curve
bound to one `(base_mint, quote_mint)` pair on a MarketMaker. Titan's
router typically holds many (one per active strategy) and aggregates
quote distribution across them.

## What this crate does

- `FromAccount` impl decodes a `StrategyHeader` (`quay_sdk::state`),
  derives the MarketMaker PDA from `strategy.owner`, and caches the
  bound quotes account, both vault PDAs, the `GlobalConfig` PDA, and
  both mints. Program id comes off `account.owner` so the same venue
  type works against mainnet, devnet, or a local validator without a
  rebuild.
- `update_state` pulls 9 accounts in one batch and refreshes the four
  raw account blobs (`strategy_data` / `mm_data` / `quotes_data` /
  `global_config_data`) the simulator reads, two `TokenInfo` entries
  for decimals + Token-2022 detection (handles mixed SPL / Token-2022
  markets), the two vault token-account blobs (for `LoadVault*`
  inputs), and the live `Clock` sysvar (for `LoadNowSlot` /
  `LoadNowUnixSec` / `LoadArenaTimestampSec`).
- `quote` calls `quay_sdk::simulate::simulate_swap` — bit-identical to
  the on-chain `swap` ix.
- `generate_swap_instruction` delegates to `quay_sdk::ix::swap` with
  `min_amount_out = 0` (Titan handles slippage upstream).
- `program_dependencies` returns `[SPL Token, Token-2022, System program]`
  so Titan's address-lookup-table machinery sees them as venue
  prerequisites.

## `titan-integration-template` is pinned by git rev

We pin to a specific commit rather than `branch = "main"` so a force-push
or trait-rename upstream can't silently break our build. Bumping the rev
is a manual review step.

## Branding via `label()`

The upstream `PoolProtocol` enum in `titan-integration-template` is
closed at the template repo, so non-template venues cannot add their
own variant. `QuayVenue::protocol()` returns the generic
`PoolProtocol::YourPoolProtocol` discriminant and `QuayVenue::label()`
returns `"Quay"` — the brand string Titan's UI and routing logs read.

## `QUAY_PROGRAM_ID`

The exported constant holds Quay's mainnet program id
(`QUayE6nexQWYNZAEqfN8FxoNwQDSu3CAzT2qq9J1ArG`). Routers constructing
`QuayVenue` via `FromAccount::from_account` see the program id directly
off `account.owner`, so the same venue type works against mainnet,
devnet, or a local validator without a rebuild. The constant exists for
callers that need to reference the program id before loading a Strategy
(e.g. building a `getProgramAccounts` filter).

## Slot + unix-time at quote time

`get_required_pubkeys_for_update()` returns the `Clock` sysvar address
(`SysvarC1ock11111111111111111111111111111111`) as its trailing entry, so
`update_state` decodes `slot` + `unix_timestamp` off the same
`AccountsCache` batch Titan uses for every other account. Production
routers get a live clock for free — no upstream-trait change, no
per-router wiring. Freshness budget is one batch (≈1 slot), the same as
Jupiter's `AmmContext.clock_ref`.

The sysvar is a well-known address so Titan's cache dedups it across all
Quay venues — one fetch per slot, not per venue.

For replay tests / off-line backtests that need a pinned clock domain,
`with_clock` / `set_clock` override post-update:

```rust
let venue = QuayVenue::from_account(&pk, &acc)?.with_clock(slot, ts);
// or, mid-flight:
venue.set_clock(slot, ts);
```

Note: the next `update_state` will overwrite both fields from the live
sysvar. Use the override only when the next caller is `quote()`.

## Halt-gate mirror + transfer-fee refusal

`initialized()` and `quote()` both gate on the on-chain `execute_swap`
halt set — `!cfg.{swap,protocol}_halted` ∧
`!strategy.{frozen,frozen_admin}` ∧ `!mm.{frozen,frozen_admin,halted_admin}`
— and additionally refuse Token-2022 mints carrying a `TransferFeeConfig`
extension. The on-chain swap prices `amount_in` gross, so a transfer-fee
mint would short-fill against the curve's quoted output; the venue stays
"not initialized" until the on-chain swap is taught to net the fee.
