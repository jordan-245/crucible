# Live-execution integration map (scout, 2026-06-09)

Scouted Atlas's live-execution stack before building. Conclusion: **reuse Atlas's broker/reconcile/kill-switch
substrate; do NOT reuse its plan layer (it's long-only entry+stop, doesn't fit target-weight long-short books);
add an IB adapter + a thin target-weight executor + a track-vs-expectation gate.** Don't rebuild in hephaestus.

## Reuse surface (confirmed reusable as-is)
| layer | Atlas module | reuse |
|---|---|---|
| Broker interface + types | `brokers/base.py` (`BrokerAdapter` ABC; OrderResult/PositionInfo/AccountInfo/DealInfo) | IB adapter implements this |
| Broker selection (config-driven) | `brokers/registry.py` (`get_live_broker`, factory dict, `_KNOWN_BROKERS`) | register `"ib"` |
| Paper/live/passive routing | `brokers/routing_policy.py` (`BrokerRoutingPolicy`, `needs_paper_pass`, `for_paper`) | as-is |
| Position sync + fill reconcile | `brokers/live_portfolio.py`, `core/reconcile.py` (`reconcile_fills/positions`) | as-is |
| Kill-switch L1–L4 | `core/remediation_kill_switch.py` (env / remediation-halt file / `data/HALT` / **L4 drawdown-from-peak**) | as-is |
| Shadow reconciliation (running) | `scripts/reconcile_shadow.py` + `atlas-reconcile-shadow.timer` (broker-vs-internal state) | as-is |
| Order preflight | `brokers/preflight.py` | as-is |

## THE CRUX — model mismatch (why we do NOT reuse the plan layer)
Atlas `Signal` (`strategies/base.py`) + `generate_plan`/`execute_plan` (`proposed_entries`) is a **LONG-ONLY,
ENTRY + STOP-LOSS + TAKE-PROFIT swing-equity model**:
- `Signal.direction = "always 'long'"`; shorts are only half-scaffolded (comments at live_executor.py:826-833,
  exit logic at :1360) — not a first-class path.
- Per-position stop/take-profit orders are core (live_executor.py:1729/1896).
- `execute_plan` consumes `plan["proposed_entries"]` (discrete entries), even via `generate_regime_plan`.

The forge/BOREAS produce **TARGET-WEIGHT portfolios**: periodic rebalance to {symbol: weight}, vol-targeted,
**long-SHORT** (BOREAS carry/trend; forge market-neutral), no per-trade stops (risk = vol-target + portfolio
drawdown kill-switch), and BOREAS is **futures** (contracts/multipliers/margin, not shares/cash).

=> Forcing target-weight long-short futures into the long-only entry+stop Signal/plan model is the wrong move.
Build a thin target-weight executor on top of the reusable substrate instead.

## The precise build (3 pieces, all in ATLAS, reusing the substrate)
1. **IB micro-futures `BrokerAdapter`** — `brokers/ib/broker.py` implementing base.py's 9 abstract methods over
   IB; handles futures contract symbols, multipliers, margin, integer contracts, and BUY **and SELL** (shorts).
   Register in `registry.py` (`_KNOWN_BROKERS += ("ib",)` + `_make_ib_broker`). *The only large broker-code piece.*
2. **Target-weight executor (the productionization bridge)** — `brokers/target_executor.py` (NEW, thin): input
   {symbol: target_weight} from a frozen forge spec / BOREAS book → mark-to-market via the BrokerAdapter →
   target qty = weight × deployable_equity / (price × multiplier) → diff vs `get_positions()` → place orders via
   the adapter. Portfolio risk = vol-target + L4 drawdown kill-switch (reused), NOT per-trade stops. Bypasses
   `generate_plan`/`proposed_entries`; reuses BrokerAdapter + reconcile + routing + kill-switch.
3. **Track-vs-expectation gate** — `live/track_expectation.py` (NEW, small): live/shadow realized net returns
   vs the strategy's MODELED backtest expectation (Atlas shadow-reconcile checks broker-vs-internal; this checks
   strategy-vs-backtest). Halt if realized net expectancy < 0 over the forward window or daily PnL >3σ off model.

## Policy params (no code — config on top of the reused layers)
≤$250 first-blood canary, AUM-economics gate (~$10–15K micro / ~$25K equity), BOREAS-first (IB), human approval,
reconciliation halt >0.5% (already in shadow-reconcile), slippage ≤2× modeled, broker-error <1%.

## Scrapped
`hephaestus/live/broker.py` + `ShadowBroker` — pure duplication of `brokers/base.py` + Atlas reconcile/kill-switch.
The good ideas (slippage/reconciliation telemetry, target-weight protocol) are folded into pieces 2 & 3 above.
