"""Stage-2 CROSS-MARKET GENERALIZATION runner (the fluke-confirmation step for broad-scope candidates).

A stage-1 gate pass is only a CANDIDATE (see methodology/promotion-policy.md). For a BROAD-mechanism edge,
confirmation = the SAME premise must show POSITIVE OOS (untouched holdout) in a MAJORITY of pre-registered
untouched universes. This packages that battery (built ad-hoc for BAB) into a reusable, report-ALL runner.

Usage: adapt `UNIVERSES` to the candidate, run, read the breadth verdict. ALWAYS report every universe — a
real mechanism shows broad (even if weak) OOS positivity; an overfit one shows ONE lucky universe + negatives.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
sys.path.insert(0, str(_P(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sdk.adapters import yf_panel

HOLDOUT = "2022-01-01"


from sdk.stats import sharpe_or_none as sharpe  # canonical (was a divergent local copy)


def split(ret):
    ret = pd.Series(ret).dropna()
    return ret[ret.index < HOLDOUT], ret[ret.index >= HOLDOUT], ret


def bab_panel(panel, beta_lb=252, vol_lb=63, target_vol=0.10, cost_bps=8.0, hold="ME"):
    """Generic beta-neutral defensive (BAB) strategy on ANY return panel — long low-beta, short high-beta,
    monthly, lagged, net-of-cost. The reusable 'defensive premium everywhere' construction."""
    px = panel.sort_index().ffill(limit=3)
    rets = px.pct_change()
    mkt = rets.mean(axis=1)
    varm = mkt.rolling(beta_lb, min_periods=beta_lb // 2).var()
    betas = pd.DataFrame({c: rets[c].rolling(beta_lb, min_periods=beta_lb // 2).cov(mkt) / varm for c in rets.columns})
    tgt = pd.DataFrame(index=betas.resample(hold).last().index, columns=rets.columns, dtype=float)
    for d in tgt.index:
        b = betas.loc[:d].iloc[-1].dropna()
        if len(b) < 6:
            continue
        z = b.rank() - b.rank().mean()
        wl = -z.clip(upper=0); ws = z.clip(lower=0)
        wl = wl / wl.sum() if wl.sum() > 0 else wl * 0
        ws = ws / ws.sum() if ws.sum() > 0 else ws * 0
        bl = (wl * b).sum(); bh = (ws * b).sum()
        row = pd.Series(0.0, index=rets.columns)
        if bl > 1e-6:
            row[wl.index] += wl / bl
        if bh > 1e-6:
            row[ws.index] -= ws / bh
        tgt.loc[d] = row
    w = tgt.reindex(rets.index, method="ffill").shift(1).fillna(0.0)
    g0 = (w * rets).sum(axis=1)
    rv = g0.rolling(vol_lb).std() * np.sqrt(252)
    w = w.mul((target_vol / rv).clip(upper=3).shift(1).fillna(1.0), axis=0)
    g = (w * rets).sum(axis=1)
    cost = w.diff().abs().sum(axis=1) * (cost_bps / 1e4)
    return (g - cost).fillna(0.0)


def breadth_verdict(results: dict, min_frac=0.60):
    """results = {universe: holdout_sharpe}. CONFIRM if a MAJORITY have positive OOS holdout."""
    vals = [v for v in results.values() if v is not None]
    pos = sum(1 for v in vals if v > 0)
    frac = pos / len(vals) if vals else 0.0
    ok = frac >= min_frac
    print(f"\n  BREADTH: {pos}/{len(vals)} universes positive OOS ({frac:.0%}) -> "
          f"{'CONFIRMED (generalises)' if ok else 'REJECTED (overfit outlier — does NOT generalise)'}")
    return ok


# Example registry of untouched ETF universes for an equity/defensive 'broad' factor.
SECTOR_ETFS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
CROSS_ASSET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ", "EWA", "EWC", "EWH", "FXI",
               "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "GLD", "SLV", "DBC", "USO", "VNQ"]


def run_defensive_battery():
    """The canonical broad-factor confirmation (what rejected BAB): run the defensive premise across
    untouched ETF universes + report ALL holdouts + the breadth verdict."""
    res = {}
    for name, etfs in [("sector-ETFs", SECTOR_ETFS), ("cross-asset/intl-ETFs", CROSS_ASSET)]:
        s, h, full = split(bab_panel(yf_panel(etfs, start="2004-01-01")))
        res[name] = sharpe(h)
        print(f"  {name:24s} search {sharpe(s)} | HOLDOUT {sharpe(h)} | full {sharpe(full)}")
    return breadth_verdict(res)


if __name__ == "__main__":
    run_defensive_battery()
