"""Gate canary — mutation testing for the gate stack ("gates rot" defense).

A frozen battery of KNOWN-BAD strategies, each designed to be killed by ONE designated
gate. Run weekly (crucible-lint.timer). Any canary that fully PASSES the rails means a
gate has rotted -> loud Telegram alert + nonzero exit. Lesson origin: I3 — the regime
gates passed vacuously for months and nothing noticed.

Isolation: ri.configure() points holdout-ledger/FDR-registry state at a throwaway dir,
and run_experiment(write_wiki=False, alert=False) — canaries never touch the production
registry, never burn real holdout entries, never write wiki pages, never alert PASS.

Assertion hierarchy per canary:
  HARD  PASSED_ALL_GATES must be False           -> violation = RED alert (gate stack breached)
  SOFT  the DESIGNATED gate actually fired       -> miss = YELLOW warning (canary blocked
        earlier than its target gate — that gate went UNTESTED this run; battery degraded)

Usage: python3 -m agent.canary
"""
from __future__ import annotations

import sys
import tempfile

import numpy as np
import pandas as pd

SEED = 20260610          # frozen — the battery must be deterministic
N_NAMES = 100
SECTORS = [f"S{i}" for i in range(10)]
START, END = "2014-01-02", "2023-12-29"
HOLDOUT = "2022-01-01"


def _sector_map():
    return {f"C{i:03d}": SECTORS[i % len(SECTORS)] for i in range(N_NAMES)}


def _panel(rng, drift_common=0.0, drifts_idio=None):
    """Synthetic wide price panel (dates x names): GBM-ish with optional common drift and
    per-name idiosyncratic drifts. Plain wide frame -> _price_matrix uses it directly."""
    idx = pd.date_range(START, END, freq="B")
    n = len(idx)
    common = rng.normal(drift_common / 252, 0.01, n)            # market factor
    idio = rng.normal(0.0, 0.015, (n, N_NAMES))
    if drifts_idio is not None:
        idio = idio + drifts_idio / 252
    rets = common[:, None] * 1.0 + idio
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=idx, columns=[f"C{i:03d}" for i in range(N_NAMES)])


def _top10(score: pd.DataFrame, freq: str = "ME", n: int = 10) -> pd.DataFrame:
    """Equal-weight long the top-n by score, rebalanced at freq (weights held in between)."""
    pe = score.resample(freq).last()
    ranks = pe.rank(axis=1, ascending=False)
    w_p = (ranks <= n).astype(float)
    w_p = w_p.div(w_p.sum(axis=1).replace(0, np.nan), axis=0)
    return w_p.reindex(score.index, method="ffill").fillna(0.0)


def _spec(sid, family, panel, weights_fn, grid=None, hedge_tickers=None, hedge_cap=None):
    """Build a StrategySpec around a frozen weight-matrix function (panel -> W, same-day;
    harness contract: lag inside signal via net_of_cost(W.shift(1)))."""
    from sdk.harness import StrategySpec
    from sdk.signal_kit import net_of_cost, trades_from_weights, market_regime

    smap = _sector_map()

    def signal(p, **params):
        W = weights_fn(p, **params)
        rets = p.pct_change()
        ret = net_of_cost(W.shift(1), rets, cost_bps=8.0, name=sid)
        trades = trades_from_weights(W.shift(1).fillna(0.0), rets, smap,
                                     regimes=market_regime(rets))
        return ret, trades

    return StrategySpec(
        id=sid, family=family, title=f"CANARY {sid}", markets=["synthetic"],
        data_desc="synthetic deterministic panel (canary battery)",
        pre_registration="KNOWN-BAD canary — must NOT pass the rails; see agent/canary.py",
        load_data=lambda: panel, signal=signal, grid=grid or {"default": {}},
        holdout_start=HOLDOUT, deploy_max_positions=10, scope="local",
        hedge_tickers=hedge_tickers, hedge_cap=hedge_cap,
    )


# ---------------------------------------------------------------- the battery

def canary_screen():
    """Zero-edge signal -> designated gate: tier-0 SCREEN (|search Sharpe| < floor).
    Static market-neutral book (5 long / 5 short, no drift anywhere): Sharpe ~0 by
    construction — must die at the very first screen."""
    rng = np.random.default_rng(SEED)
    panel = _panel(rng)

    def w(p, **k):
        W = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        W.iloc[:, :5] = 0.1
        W.iloc[:, 5:10] = -0.1
        return W

    v = _run(_spec("canary-screen-zero-edge", "canary_screen", panel, w))
    fired = v.get("tier") == "SCREEN_FAIL"
    return _judge("screen (tier-0)", v, fired)


def canary_lookahead():
    """Signal peeks at NEXT-day returns -> designated gate: MCPT (cheats reproduce on
    permuted/structureless data; classical metrics all love it).

    Needs a small param grid: DSR/PBO require >=2 grid variants to compute (single-config
    grids report NaN -> tier mechanically FAILs and MCPT never runs -> the canary would die
    at the wrong gate and leave MCPT untested). Every variant cheats identically, so the
    classical stack (DSR/PBO/CPCV/holdout) must love all of them — that's the point."""
    rng = np.random.default_rng(SEED + 1)
    panel = _panel(rng)

    def w(p, top_n: int = 10, **k):
        fwd = p.pct_change().shift(-1)            # tomorrow's return, known today: the leak
        return _top10(fwd, freq="W-FRI", n=top_n)  # undiluted weekly cheat: Sharpe ~5 in-sample

    v = _run(_spec("canary-lookahead-leak", "canary_lookahead", panel, w,
                   grid={"default": {}, "top8": {"top_n": 8}, "top12": {"top_n": 12}}))
    fired = v.get("mcpt_pass") is False
    note = "" if v.get("mcpt_pass") is not None else "never reached MCPT"
    return _judge("MCPT (lookahead)", v, fired, note=note)


def canary_beta_clone():
    """Long-only book riding a bull market with NEGATIVE stock-picking skill (holds the 30
    names given a negative idiosyncratic drift) -> designated gate: beta-confound
    (beta_to_universe high, selection-alpha clearly below the floor)."""
    rng = np.random.default_rng(SEED + 2)
    drifts = np.zeros(N_NAMES)
    drifts[:30] = -0.03                            # the held names are mild losers vs the universe
    panel = _panel(rng, drift_common=0.18, drifts_idio=drifts)

    def w(p, **k):
        W = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        W.iloc[:, :30] = 1.0 / 30
        return W

    v = _run(_spec("canary-beta-clone", "canary_beta", panel, w))
    fired = bool(v.get("beta_confound"))
    return _judge("beta-confound", v, fired)


def canary_overfit():
    """Best-of-40 seed-mined noise, the full grid declared -> designated gates: DSR
    deflation (effective trials) / PBO. The classic selection-bias mirage."""
    rng = np.random.default_rng(SEED + 3)
    panel = _panel(rng)
    rets = panel.pct_change()
    search = rets[rets.index < HOLDOUT]

    def w_seed(p, seed=0):
        r = np.random.default_rng(10_000 + seed)
        score = pd.DataFrame(r.normal(size=p.shape), index=p.index, columns=p.columns)
        return _top10(score.rolling(21).mean())

    # mine in-sample: pick the luckiest of 40 random books (the sin under test)
    best, best_sh = 0, -9e9
    for s in range(40):
        ret = (w_seed(panel, s).shift(1) * search).sum(axis=1)
        sh = ret.mean() / (ret.std() + 1e-12) * np.sqrt(252)
        if sh > best_sh:
            best, best_sh = s, sh
    grid = {"default": {}, **{f"seed_{s}": {"seed": s} for s in range(40) if s != best}}
    spec = _spec("canary-overfit-mined", "canary_overfit", panel,
                 lambda p, seed=best: w_seed(p, seed), grid=grid)
    v = _run(spec)
    bar = v.get("promote_bar") or 0.982
    fired = (v.get("tier") != "PROMOTE") and (
        (v.get("dsr") is not None and v["dsr"] < bar) or (v.get("pbo") or 0) > 0.2
        or v.get("tier") == "SCREEN_FAIL")
    return _judge("DSR/PBO (overfit)", v, fired)


def canary_holdout_double_dip():
    """Same frozen config evaluated TWICE -> designated gate: write-once holdout ledger
    must refuse the second OOS look (holdout_burned + forced holdout FAIL).

    Construction: names get persistent built-in drifts and the (canary-only, knowingly
    omniscient) signal ranks by those drifts — guaranteed to clear the tier-0 screen so
    the first run EARNS and BURNS the one allowed holdout look. The gate under test is
    the LEDGER, not the signal's honesty."""
    rng = np.random.default_rng(SEED + 4)
    drifts = rng.normal(0.0, 0.12, N_NAMES)
    panel = _panel(rng, drifts_idio=drifts)
    top = np.argsort(drifts)[-10:]                # frozen: the 10 best-drift names

    def w(p, **k):
        W = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        W.iloc[:, top] = 0.1
        return W

    spec = _spec("canary-double-dip", "canary_holdout", panel, w)
    v1 = _run(spec)
    if v1.get("tier") == "SCREEN_FAIL":           # never earned the first OOS look
        return _judge("write-once holdout", v1, fired=False,
                      note="blocked at tier-0 screen — holdout gate untested")
    v2 = _run(spec)                               # the second look: must be refused
    fired = bool(v2.get("holdout_burned")) and v2.get("holdout_pass") is False \
        and any("WRITE-ONCE" in r for r in (v2.get("holdout_reasons") or []))
    return _judge("write-once holdout", v2, fired)


def canary_hedge_loophole():
    """Abuses the hedge-sleeve exemption (added 2026-06-12): a 3-name micro-book hiding
    behind a legally-declared, whitelisted, under-cap hedge -> designated gate: the
    deployment-sanity floors must bite on the EX-HEDGE alpha book (peak concurrency /
    realized-vs-design). The exemption must make the gate STRICTER, never looser."""
    rng = np.random.default_rng(SEED + 5)
    drifts = np.zeros(N_NAMES)
    drifts[:3] = 0.25                              # 3 strong names: clears the tier-0 screen
    panel = _panel(rng, drifts_idio=drifts)

    def w(p, **k):
        W = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        W.iloc[:, :3] = 0.15                       # the whole "alpha book" is 3 names
        W.iloc[:, 99] = -0.25                      # continuous short, declared as the hedge
        return W

    # C099 stands in for IWM: rename the column so the whitelist applies
    panel = panel.rename(columns={"C099": "IWM"})
    v = _run(_spec("canary-hedge-loophole", "canary_hedge", panel, w,
                   hedge_tickers=["IWM"], hedge_cap=0.60))
    reasons = " ".join(v.get("deploy_reasons") or [])
    fired = v.get("deployment_passed") is False and (
        "peak_concurrent" in reasons or "realized_vs_design" in reasons or "n_trades" in reasons)
    return _judge("deployment ex-hedge floors", v, fired)


# ---------------------------------------------------------------- harness glue

_ISOLATED = False  # set by _ensure_isolation(); canaries must NEVER touch production state


def _ensure_isolation():
    """Defense-in-depth: guarantee the holdout-ledger/FDR-registry point at a throwaway dir
    BEFORE any canary runs, regardless of entry point. Lesson 2026-06-11: a direct call to a
    canary function (diagnostic harness, REPL, future import) skipped main()'s ri.configure()
    and burned a canary entry into the PRODUCTION write-once ledger + FDR registry (cleaned by
    hand from /root/research-wiki/.registry). Isolation must be structural, not caller-courtesy."""
    global _ISOLATED
    if _ISOLATED:
        return
    import research_integrity as ri
    tmp = tempfile.mkdtemp(prefix="canary-ri-")
    ri.configure(tmp)
    _ISOLATED = True


def _run(spec) -> dict:
    _ensure_isolation()
    from sdk.harness import run_experiment
    return run_experiment(spec, write_wiki=False, alert=False)


def _judge(gate: str, v: dict, fired: bool, note: str = "") -> dict:
    breached = bool(v.get("PASSED_ALL_GATES"))    # HARD invariant
    return {"id": v.get("id"), "gate": gate, "breached": breached, "fired": fired,
            "tier": v.get("tier"), "note": note,
            "detail": {k: v.get(k) for k in ("dsr", "pbo", "search_sharpe", "holdout_sharpe",
                                             "beta_confound", "mcpt_pass", "holdout_burned")}}


BATTERY = [canary_screen, canary_lookahead, canary_beta_clone,
           canary_overfit, canary_holdout_double_dip, canary_hedge_loophole]


def main() -> int:
    _ensure_isolation()                            # isolated ledger/registry — never production
    results = []
    for fn in BATTERY:
        try:
            results.append(fn())
        except Exception as e:                     # a crashed canary = that gate went untested
            results.append({"id": fn.__name__, "gate": "?", "breached": False,
                            "fired": False, "tier": "CRASH",
                            "note": f"{type(e).__name__}: {str(e)[:140]}", "detail": {}})

    red = [r for r in results if r["breached"]]
    yellow = [r for r in results if not r["breached"] and not r["fired"]]
    for r in results:
        flag = "BREACH" if r["breached"] else ("ok" if r["fired"] else "untested")
        print(f"[canary] {flag:9s} {r['id']:28s} gate={r['gate']:22s} tier={r['tier']} "
              f"{r['note']}")

    if red or yellow:
        from sdk.notify import telegram_msg
        lines = ["🐤 <b>Gate canary report</b>"]
        for r in red:
            lines.append(f"🚨 <b>GATE BREACH</b>: known-bad <code>{r['id']}</code> PASSED ALL "
                         f"GATES — the <b>{r['gate']}</b> gate has rotted. Halt promotions "
                         f"until investigated.")
        for r in yellow:
            lines.append(f"⚠️ <code>{r['id']}</code> blocked before its designated gate "
                         f"(<b>{r['gate']}</b> untested this run): {r['note'] or r['tier']}")
        telegram_msg("\n".join(lines))
    else:
        print(f"[canary] all {len(results)} canaries killed by their designated gates — stack healthy")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
