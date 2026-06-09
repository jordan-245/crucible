# Phase 4 — PASS → Live Execution Pipeline (at scale)

Extend the forge from "a strategy that clears all gates" to "real capital trading on the right broker," with
a portfolio of concurrent live strategies.

> **BOARD RATIFIED 2026-06-09 (5–0 CONDITIONAL, HIGH).** memo: `ceo-board/memos/2026-06-09-forge-go-live-policy/`.
> The "skip paper" premise was REVERSED: at $1–5K, real capital pays real fees to learn little + erodes gate
> discipline (risk lesson 2026-06-03). **Real capital is AUM-gated (~$25K) + forward-PAPER evidence — NOT now.**
> The gate is **SHADOW / paper-forward** validation (Alpaca paper on live data — validates execution AND forward
> edge at $0). Forward-paper bar: ≥40–50 trades, +ve net-of-cost expectancy, ≥2 regimes, slippage/CLV-adjusted,
> auto-revert kill-switch. Anti-paralysis exception: ONE ≤$250 first-blood canary on the first shadow-cleared
> pass. First venue = Alpaca PAPER equities; crypto-carry live BLOCKED (leverage/custody gate); IB futures later.
> **Build order: productionize (B) + shadow harness (E) FIRST; DEFER live broker adapters + real-money allocator.**
> Human approval on every go-live/scale-up/new-broker; no autonomous capital. Re-ratify at first shadow-pass or ~$25K AUM.

## The core reframe (drives everything below)
The gates validate **ALPHA** (a statistical edge in a frictionless returns series). They do **NOT** validate
**EXECUTION** (sizing, fills, slippage, borrow, broker API, live data, reconciliation, corporate actions).
These are orthogonal. So: skip the long paper-*alpha* trial; keep a short execution-*validation* bridge.

## What a PASS gives us today (and what it does NOT)
- A PASS = a frozen `signal(panel, **params) -> (daily_returns, trades)` that cleared stage-1 (CPCV/DSR/PBO/
  holdout/deployment-sanity/beta-confound) + stage-2 (MCPT + generalization OR forward-validation).
- It is RESEARCH code: it emits a returns series + a trade ledger. It is NOT a live order router, has no real
  position sizing, no broker wiring, no live-data daily loop. Productionizing that is Stage B.

## Pipeline stages (each is a gate)
**A. PASS event (built).** Green Telegram on full-gate pass -> human review of the wiki experiment page.
   ALPHA gate. The human decides "promote to live pipeline."

**B. Productionization — signal -> executable target portfolio.**
   - Wrap the frozen `signal` as a deterministic daily `target_portfolio(asof_date) -> {symbol: target_weight}`
     callable on LIVE data (the same logic the backtest used, evaluated at today's bar).
   - Emit broker-agnostic ORDER INTENTS: {symbol, target_weight, side}; the engine diffs vs current
     positions -> orders. (Atlas's ingest->plan->approve->execute already has this shape — extend it.)
   - Pin the exact data sources + params from the frozen spec (reproducibility = the live config IS the spec).

**C. Broker & asset-class routing (AU operator).** A `BrokerAdapter` interface
   (place_order/positions/buying_power/reconcile) with one impl per venue:
   | Asset class | Broker | Why | Small-capital fit |
   |---|---|---|---|
   | US equities | **Alpaca** (already integrated) | commission-free, AU-accessible, FRACTIONAL shares, paper+live same API | best — fractional makes $1-5K viable; shorts need marginable borrow |
   | Futures (trend/carry, Boreas) | **Interactive Brokers** | only realistic retail futures; MICRO futures (MES/MNQ/micro metals-energy-FX) | fixed ~$1-5K notional/contract -> needs more capital or 1-2 contracts |
   | Crypto carry (Midas) | perp/spot venue (e.g. Bybit) | funding-carry needs a perp venue | viable small, BUT AU regulatory + custody risk — board call |
   - Routing layer keeps strategy logic broker-agnostic; brokers are swappable.

**D. Capital allocation & risk budgeting (the "at scale" part).** Portfolio-OF-strategies:
   - Allocator gives each live strategy a capital slice: equal-risk-contribution / risk-parity, **capped per
     strategy**, with a **correlation limit** (don't double-fund two correlated edges — the carry+trend lesson).
   - Portfolio risk budget: total gross-exposure cap, portfolio drawdown kill-switch, per-strategy DD kill-switch.
   - A new live strategy enters SMALL and ramps as it proves live execution + live PnL tracking (Stage E).

**E. Execution-validation bridge (the lean non-skip — IMPLEMENTATION risk, not alpha).**
   1. **Shadow (≈1-2 wks, $0):** generate live target orders daily, DON'T place; log + compare to simulated
      fills. Proves: runs on live data, sizing sane, no crashes, intents match backtest expectation.
   2. **Canary (≈2-4 wks, tiny real $):** ~1-2% of the slice (e.g. $100-250). Proves: real fills, slippage,
      borrow, broker mechanics, reconciliation. Real but bounded loss.
   3. **Track-vs-expectation gate:** live PnL must sit within the backtest's expected band (not a path match —
      a sanity band: live Sharpe not catastrophically below backtest, no execution-driven bleed). Diverge -> halt.
   4. Pass all three -> full slice. This is DAYS-WEEKS + pocket change, not a months-long alpha trial.

**F. Live ops & monitoring (daily).** ingest live data -> target_portfolio -> portfolio risk-check ->
   approve (human first; auto within a pre-approved risk envelope once proven) -> execute -> reconcile ->
   dashboard + Telegram. Monitors: per-strategy live-vs-backtest PnL, DD, exposure, broker connectivity,
   reconciliation breaks. Reuse Atlas's daily flow + `remediation_kill_switch`.

**G. Governance.** Human-in-the-loop for: first go-live, capital increases, new broker, new strategy to canary.
   Auto (within pre-approved caps) for: daily rebalances of already-live strategies. Config-promotion guardrails
   + plan-gate tools already enforce approval; extend to the live multi-strategy config.

## Scale & capital reality (retail $1-5K -> growing)
- Fractional equities (Alpaca) are the most capital-efficient first venue. Futures need more capital (fixed
  micro notional); crypto carry is small-viable but carries venue/custody/regulatory risk.
- **Important consequence of the new beta-confound gate:** equity strategies that now pass are likely
  LONG-SHORT / low-beta (real selection alpha), which needs shorting + borrow + margin — operationally harder
  at $1-5K than long-only. So the FIRST live strategies may well be FUTURES (IB micro) or CRYPTO CARRY, not
  equities. Let the gates + capital efficiency pick the first venue, not a prior preference.

## Safeguards (non-negotiable)
Per-strategy + portfolio drawdown kill-switches · gross-exposure cap · per-strategy capital cap · correlation
cap · reconciliation-break halt · broker-connectivity-loss halt · error-rate halt · human approval on go-live /
scale-up / new broker · everything auditable (the risk tools already write audit records).

## Open decisions for the board (go-live is irreversible capital -> board ratifies)
1. Go-live in principle on PASS (yes/no), and the ALPHA bar to promote (full stage-2, or forward-validation too?).
2. The execution-validation policy: shadow->canary->scale (recommended) vs straight-to-capital (operator's ask).
3. Broker set + which asset class goes live FIRST (capital efficiency vs regulatory/custody risk).
4. Capital: total live budget, per-strategy cap, portfolio DD kill-switch level, ramp schedule.
5. Autonomy boundary: what executes auto within the risk envelope vs what needs human approval.

## Build sequence (after the board ratifies)
1. `live/target_portfolio.py` — frozen-spec -> daily target weights on live data (Stage B).
2. `live/broker/` — BrokerAdapter + Alpaca impl first (equities), then IB (futures), then crypto.
3. `live/allocator.py` — capital allocation + portfolio risk budget (Stage D).
4. `live/bridge.py` — shadow + canary harness + track-vs-expectation gate (Stage E).
5. `live/daily.py` — the daily ops loop + monitoring + kill-switches (Stage F), wired to dashboard + Telegram.
6. Governance wiring — approval gates + audit (Stage G), reuse atlas_risk_* tools.
