"""Rail 3 — Deployment-sanity (SHARED research-integrity infra).

PROJECT-AGNOSTIC: deployment_sanity/expected_positions are pure. (the old `deployment_smoke` ATLAS
reference runner. For non-equity projects (e.g. Hermes betting) define your own thresholds/analog
(bets/day, game spread, single-game share) or call deployment_sanity on bet-trade dicts. Original:

A cross-OOS battery TIER is meaningless unless the strategy actually trades the book it was
designed for. 2026-06-05: cross_sectional_momentum "PROMOTEd" (DSR 0.926) while a sector-tag bug
capped it to 2 concurrent positions; the "edge" lived in the top 1-2 names. Properly deployed (~14)
it FAILED. A human caught it; at 1000 runs/day no human will. This module auto-FAILs such artifacts.

Spec: research/INTEGRITY_RAILS_SPEC.md (Rail 3). Lesson: tasks/lessons.md 2026-06-05.
Thresholds are PRE-REGISTERED and FROZEN here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Pre-registered, frozen thresholds ────────────────────────────────────────
MIN_TRADES = 50                 # fewer => degenerate / luck, not a strategy
MIN_PEAK_FRAC_OF_DESIGN = 0.25  # peak concurrency must be >= 25% of the intended book size
MIN_PEAK_ABS = 3                # ...and at least 3 names absolute
MAX_SINGLE_NAME_SHARE = 0.40    # max fraction of dollar-position-days in one ticker
MIN_REALIZED_VS_DESIGN = 0.50   # peak_concurrent / expected_positions
DEFAULT_N_SECTORS = 11          # GICS-style sector count for the design-intent calc

# ── Hedge-sleeve exemption (2026-06-12) ─────────────────────────────────────────
# Origin: two genuine Amihud near-misses (holdout Sharpe 1.44/1.31) were force-failed on
# single_name_share because their PRE-REGISTERED index-ETF residual-beta hedge dominated
# position-days. The gate could not tell a declared hedge instrument from an accidental
# concentration bet. Design (anti-loophole, each invariant tested in crucible
# tests/test_hedge_sleeve.py):
#   1. hedge tickers must be on THIS frozen broad-index-ETF whitelist — a single stock can
#      never be declared a "hedge" to dodge concentration.
#   2. the ALPHA book is judged ALONE: hedge trades are excluded from ALL deployment metrics
#      (trade count, concurrency, single-name/sector share), so a thin alpha book cannot
#      hide behind hedge bulk — exclusion makes every other check STRICTER, not looser.
#   3. hedge share of total position-days must stay <= the declared cap (and the hard
#      ceiling below) — an oversized "hedge" is index-substitution in disguise; the
#      beta-confound gate (harness stage-1) independently catches the return-side version.
#   4. no declaration -> byte-identical legacy behavior (frozen designs unaffected).
HEDGE_ETF_WHITELIST = frozenset({
    "SPY", "IVV", "VOO",          # S&P 500
    "IWM", "MDY", "IJH", "IJR",   # small/mid-cap
    "QQQ",                        # Nasdaq-100
    "VTI", "ITOT",                # total US market
    "EFA", "EEM", "ACWI",         # international
    "TLT", "IEF", "SHY",          # treasuries
    "GLD", "USO", "DBC",          # commodities
})
MAX_HEDGE_SHARE_CEILING = 0.60  # hard ceiling: no declared cap may exceed this


def _sector_of(t: Dict[str, Any]) -> str:
    s = t.get("sector")
    if not s:
        s = (t.get("features") or {}).get("sector")
    return s or "Unknown"


def expected_positions(primary_config: Optional[dict], strategy_meta: Optional[dict]) -> int:
    """Design-intent book size = min(top_n or max_positions, sector_cap x n_sectors)."""
    pc = primary_config or {}
    sm = strategy_meta or {}
    max_pos = int(sm.get("max_positions") or 0) or None
    top_n = pc.get("top_n") if isinstance(pc, dict) else None
    top_n = int(top_n) if top_n else None
    sector_cap = int(sm.get("max_sector_concentration") or 0) or None
    n_sectors = int(sm.get("n_sectors") or DEFAULT_N_SECTORS)
    cand = [x for x in (top_n, max_pos) if x]
    base = min(cand) if cand else (max_pos or 10)
    if sector_cap:
        base = min(base, sector_cap * n_sectors)
    return max(1, int(base))


def deployment_sanity(trades: List[Dict[str, Any]],
                      primary_config: Optional[dict] = None,
                      strategy_meta: Optional[dict] = None,
                      hedge_tickers: Optional[List[str]] = None,
                      hedge_cap: Optional[float] = None) -> Dict[str, Any]:
    """Compute deployment metrics + auto-FAIL gates from a closed-trade list.

    hedge_tickers/hedge_cap: optional PRE-REGISTERED hedge sleeve (see module note above
    HEDGE_ETF_WHITELIST). Hedge trades are excluded from the alpha-book metrics; the sleeve
    itself is gated on whitelist membership and position-day share <= hedge_cap (cap itself
    capped by MAX_HEDGE_SHARE_CEILING). Declaring one without the other is a forced fail.

    Returns a dict with metrics, `passed` (bool), and `forced_fail_reasons` (list[str]).
    A False `passed` should force the battery TIER to FAIL regardless of DSR.
    """
    primary_config = primary_config or {}
    strategy_meta = strategy_meta or {}

    hedge_reasons: List[str] = []
    hedge_share = None
    if hedge_tickers or hedge_cap is not None:
        if not hedge_tickers or hedge_cap is None:
            hedge_reasons.append("hedge declaration incomplete: need BOTH hedge_tickers and hedge_cap")
            hedge_tickers = None
        else:
            bad = sorted(set(hedge_tickers) - HEDGE_ETF_WHITELIST)
            if bad:
                hedge_reasons.append(
                    f"hedge tickers {bad} not on the broad-index-ETF whitelist — "
                    f"single names can never be declared hedges")
                hedge_tickers = None
            elif hedge_cap > MAX_HEDGE_SHARE_CEILING:
                hedge_reasons.append(
                    f"hedge_cap {hedge_cap:.2f} > hard ceiling {MAX_HEDGE_SHARE_CEILING} — "
                    f"a 'hedge' that big is index-substitution in disguise")
                hedge_tickers = None

    def _w(t):
        e, x = t.get("entry_date"), t.get("exit_date")
        if e is None or x is None:
            return 0.0
        hd = t.get("hold_days")
        hd = float(hd) if hd is not None else float(max(0, (pd.Timestamp(x) - pd.Timestamp(e)).days))
        return (hd + 1e-9) * (float(t.get("position_value") or 0.0) or 1.0)

    if hedge_tickers:
        hset = set(hedge_tickers)
        hedge_trades = [t for t in trades if t.get("ticker") in hset]
        alpha_trades = [t for t in trades if t.get("ticker") not in hset]
        total_w = sum(_w(t) for t in trades)
        hedge_share = (sum(_w(t) for t in hedge_trades) / total_w) if total_w > 0 else 0.0
        if hedge_share > hedge_cap:
            hedge_reasons.append(
                f"hedge_share {hedge_share:.2f} > declared cap {hedge_cap:.2f} — "
                f"the sleeve is carrying the book, not trimming it")
        trades = alpha_trades  # the ALPHA book is judged alone (stricter, not looser)

    n = len(trades)
    out: Dict[str, Any] = {"n_trades": n, "passed": True, "forced_fail_reasons": list(hedge_reasons)}
    if hedge_share is not None:
        out["hedge_share"] = round(hedge_share, 3)
        out["hedge_tickers"] = sorted(hedge_tickers) if hedge_tickers else []
    if n == 0:
        out["passed"] = False
        out["forced_fail_reasons"].append("no trades")
        return out

    ev: List[tuple] = []
    pos_days_by_ticker: Dict[str, float] = {}
    pos_days_by_sector: Dict[str, float] = {}
    total_pos_days = 0.0
    hold_days: List[float] = []
    entry_dts: List[pd.Timestamp] = []
    exit_dts: List[pd.Timestamp] = []
    for t in trades:
        e, x = t.get("entry_date"), t.get("exit_date")
        if e is None or x is None:
            continue
        e, x = pd.Timestamp(e), pd.Timestamp(x)
        entry_dts.append(e); exit_dts.append(x)
        ev.append((e, 1)); ev.append((x, -1))   # (ts,-1) sorts before (ts,+1): exits first, no overcount
        hd = t.get("hold_days")
        hd = float(hd) if hd is not None else float(max(0, (x - e).days))
        hd += 1e-9
        pv = float(t.get("position_value") or 0.0) or 1.0   # dollar weight; equal-weight fallback
        w = hd * pv
        tk = t.get("ticker", "?")
        sec = _sector_of(t)
        pos_days_by_ticker[tk] = pos_days_by_ticker.get(tk, 0.0) + w
        pos_days_by_sector[sec] = pos_days_by_sector.get(sec, 0.0) + w
        total_pos_days += w
        hold_days.append(hd)

    ev.sort()
    cur = peak = 0
    conc_track: List[int] = []
    for _, delta in ev:
        cur += delta
        peak = max(peak, cur)
        conc_track.append(cur)
    avg_conc = float(np.mean([c for c in conc_track if c > 0])) if conc_track else 0.0

    years = 1.0
    if entry_dts and exit_dts:
        span = (max(exit_dts) - min(entry_dts)).days
        years = max(span / 365.25, 1e-6)

    exp = expected_positions(primary_config, strategy_meta)
    realized_vs_design = peak / exp if exp else 0.0
    single_name_share = (max(pos_days_by_ticker.values()) / total_pos_days) if total_pos_days > 0 else 1.0
    max_sector_share = (max(pos_days_by_sector.values()) / total_pos_days) if total_pos_days > 0 else 1.0

    out.update({
        "peak_concurrent": peak,
        "avg_concurrent": round(avg_conc, 2),
        "trades_per_year": round(n / years, 1),
        "expected_positions": exp,
        "realized_vs_design": round(realized_vs_design, 3),
        "sector_spread": len(pos_days_by_sector),
        "max_sector_share": round(max_sector_share, 3),
        "single_name_share": round(single_name_share, 3),
        "median_hold_days": round(float(np.median(hold_days)), 1) if hold_days else None,
    })

    reasons: List[str] = out["forced_fail_reasons"]
    if n < MIN_TRADES:
        reasons.append(f"n_trades {n} < {MIN_TRADES}")
    peak_floor = max(MIN_PEAK_ABS, MIN_PEAK_FRAC_OF_DESIGN * exp)
    if peak < peak_floor:
        reasons.append(
            f"peak_concurrent {peak} < floor {peak_floor:.1f} "
            f"(expected {exp}) — not deploying as designed")
    if single_name_share > MAX_SINGLE_NAME_SHARE:
        reasons.append(
            f"single_name_share {single_name_share:.2f} > {MAX_SINGLE_NAME_SHARE} "
            f"— accidental single-name concentration")
    if realized_vs_design < MIN_REALIZED_VS_DESIGN:
        reasons.append(
            f"realized_vs_design {realized_vs_design:.2f} < {MIN_REALIZED_VS_DESIGN} "
            f"— engine constraints throttling the book")
    out["passed"] = len(reasons) == 0
    return out


# NOTE (S6 2026-06-10): deployment_smoke (Atlas reference runner) REMOVED as dead code — imported
# nonexistent atlas modules, zero callers. Projects call deployment_sanity() on their own trades.

__all__ = ["deployment_sanity", "expected_positions", 
           "MIN_TRADES", "MIN_PEAK_FRAC_OF_DESIGN", "MIN_PEAK_ABS",
           "MAX_SINGLE_NAME_SHARE", "MIN_REALIZED_VS_DESIGN",
           "HEDGE_ETF_WHITELIST", "MAX_HEDGE_SHARE_CEILING"]
