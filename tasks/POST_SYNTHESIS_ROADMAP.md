# Post-Synthesis Roadmap — what happens next, and when (2026-06-12)

The synthesis is done; the system now runs itself nightly. From here the work changes character:
**less building, more evidence-watching.** Dates below are derived from the board's frozen go-live
policy (memo 2026-06-09, 5-0) and the actual evidence counters as of today.

## The two forward books — evidence horizon

| | val_mom_trend_smallcap | amihud_illiq_tranched_v3 |
|---|---|---|
| deployed | 2026-06-10 | 2026-06-12 (clean redeploy after capital/tif bugs) |
| return days | 3 / 20 needed (G2) | 0 / 20 |
| fills | 99 / 40 needed (G1) ✅ | 0 / 40 |
| expectancy (G3) | -21bps/day (3 days = noise; not meaningful before ~day 15) | — |
| earliest G1–G7 all-scoreable | **~2026-07-08** (20 trading days) | **~2026-07-10** |
| watch items | first OPG fills vs G6 slippage bar (16bps); G7 broker-error reset after the 4 rejected-OPG errors | first clean OPG submission TODAY 10:30 AEST; capital now $5K design size |

Weekly cadence: crucible-forward-evidence (Fri) scores G1–G7 + lifecycle transitions.
Both books flip lifecycle shadow → evidence at ~20 days (early July). Decay rule needs 60 return
days + 2 weekly confirms → earliest possible decay flag ~September (by design — no false urgency).

## The go-live decision — what it will actually look like (board policy, frozen)

Real capital requires ALL of:
1. **Forward-paper gate PASS** (G1–G7: ≥40 fills, ≥20 days, +expectancy, ≥2 regimes, recon clean,
   slippage ≤2× modeled, broker errors <1%) — earliest ~mid-July for val_mom.
2. **AUM floor, unit-economics-derived**: ~$10–15K for micro-tradable cross-asset; **~$25K for
   borrow-dependent equity** (both our current books are smallcap equity — the HIGHER bar applies).
3. **Kill-switch live-tested + human sign-off** (board: one pre-approved ≤$250 "first-blood canary"
   on the first shadow-cleared pass — tuition, not a bet).
4. Stage D allocator (task #7, board-deferred) sizes across books in lifecycle
   evidence/real_capital_candidate only — its input contract shipped with Stage 3.

So the realistic go-live shape: **val_mom or tranched_v3 clears G1–G7 in July → becomes
real_capital_candidate → still waits on the $25K AUM floor** (account is $14.4K paper; real AUM
far below). The binding constraint is AUM, not evidence. The unlock paths are (a) capital
deposit decision (human/board), or (b) a NATIVELY micro-tradable PASS (cross-asset/futures)
where the floor is only $10–15K — which is exactly why the board said BOREAS-first.

## BOREAS thread (the board's preferred first real-capital venue)

- Carry forward-verdict date: **2026-08-28** (frozen). If carry confirms → BOREAS carry+trend on
  IB micro-futures is the first real-money book (natively tradable at $5K, scales 100×).
- Board sequencing: **IB micro adapter must land by 2026-08-28** so a confirming verdict has a
  path the moment it arrives. This is the one BUILD item left on the books. Not started.
  → becomes the next implementation task after a few clean forge nights.

## Forge-side maturation (no code — data accumulating on timers)

| trigger | threshold | ETA at 3 smiths × 3 cycles/night |
|---|---|---|
| Thompson bandit fit over arms | ≥60 arm-labelled outcomes | ~late June (counting started 2026-06-12, 2/60) |
| Regime-coverage demotion activates | 2026-06-26 (phase-in ends) | fixed date |
| Regime-burner falsifiability review | 60 days / ≥20 stage-1 evals | ~mid-August |
| Retirement-rule falsifiability review | 2 quarters of live books | ~December |
| FABLE-5 revert to $0 models | **2026-06-22 06:00 AEST (auto-timer)** | watch: smith quality may shift post-revert — compare arm rewards before/after |

## Standing verification (this week)

1. Tomorrow AM: first full-system forge night — arm draws (organic crossover?), regime-burner
   lines on new experiment pages, consistency severity distribution, no elite.record exceptions.
2. First OPG fills for both books → G6 slippage with auction prints (the 2026-06-11 fix's payoff).
3. Watch tranched_v3's day-1 book build (the redeployed $5K/opg config's first live cycle).

## What I am explicitly NOT doing

- Not adding gates (stack FROZEN). Not fitting the bandit early. Not building Stage D before AUM.
- Not touching the two forward books' configs while evidence accumulates (any change resets the
  evidence clock — the worst trade available).
