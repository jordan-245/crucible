"""HARDENED Stage-2 cross-market generalization for the Value×Momentum candidates.

Upgrades forward/generalize_valmom.py in two ways the board flagged:
  1. CPCV (not a single 2022+ split): per cap-tier we run the SAME rails the harness uses
     -- ri.assemble_bundle(in_sample_returns, trades, grid_returns) -> median CPCV Sharpe, PBO, DSR --
     so an untouched tier must be robust ACROSS folds, not just lucky on one OOS window.
  2. Both candidates: the mid-cap composite (smith3_68617, the 0.994 discovery) AND the cleaner
     trend-overlay sibling (smith2_96154, PBO 0.078) -- the better deployment form.

Discipline: report ALL tiers (incl. discovery tier as a self-reproduction sanity check). A BROAD
factor claim GENERALISES only if every UNTOUCHED tier is OOS-positive AND median_cpcv>0 AND pbo<=0.5.
One lucky tier + a weak/over-fit other tier => reject as a single-universe overfit (the BAB lesson).

Pure-compute: no capital, no writes to the live system. Result -> forward/valmom_generalization_cpcv.jsonl.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime

sys.path.insert(0, "/root/hephaestus")
import numpy as np
import pandas as pd

import research_integrity as ri
from sdk.adapters import sep_panel, sf1, us_universe

HOLDOUT = "2022-01-01"
TIERS = ["Large", "Mid", "Small"]


def _sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return round(float(r.mean() / r.std() * np.sqrt(ann)), 3) if len(r) > 20 and r.std() > 0 else None


# --------------------------------------------------------------------------- panel rebuilders
# Each mirrors the variant's OWN load_data() but at a CHOSEN cap tier, preserving that variant's
# exact panel schema (smith3: concat{px,bvps}+attrs['sector']; smith2: px+attrs['bvps','sector_map']).

def build_panel_smith3(M, marketcap, per_sector_n=120):
    tic_sector = {}
    for s in M.SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             marketcap=marketcap, include_delisted=True, top_n=per_sector_n)
        except Exception:
            ts = []
        for t in (ts or []):
            tic_sector.setdefault(t, s)
    if len(tic_sector) < 100:
        try:
            ts = us_universe(category="Domestic Common Stock", marketcap=marketcap,
                             include_delisted=True, top_n=1200)
        except Exception:
            ts = []
        for t in (ts or []):
            tic_sector.setdefault(t, M.SECTORS[hash(t) % len(M.SECTORS)])
    tickers = sorted(tic_sector)
    px = sep_panel(tickers, M.START, field="closeadj").sort_index()
    px = px.reindex(columns=[c for c in tickers if c in px.columns])
    try:
        bvps = M._pit_panel(sf1(list(px.columns), ["bvps"], dimension="ARQ"), "bvps", px.index, px.columns)
    except Exception:
        bvps = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    panel = pd.concat({"px": px, "bvps": bvps}, axis=1)
    panel.attrs["sector"] = tic_sector
    M._SECTOR_MAP = tic_sector
    return panel, int(px.shape[1])


def build_panel_smith2(M, marketcap, per_sector_n=130):
    sector_map, tickers = {}, []
    for sec in M.SECTORS:
        try:
            ts = us_universe(sector=sec, category="Domestic Common Stock",
                             marketcap=marketcap, include_delisted=True, top_n=per_sector_n)
        except Exception:
            ts = []
        for t in ts:
            sector_map[t] = sec
        tickers.extend(ts)
    tickers = sorted(set(tickers))
    px = sep_panel(tickers, M.START, field="closeadj").sort_index()
    px = px.loc[:, px.columns.isin(tickers)]
    px.attrs["bvps"] = sf1(list(px.columns), ["bvps"], dimension="ARQ")
    px.attrs["sector_map"] = sector_map
    return px, int(px.shape[1])


VARIANTS = [
    {"mod": "auto_value_momentum_complementary_combination_smith3_68617",
     "label": "mid_composite", "discovery_tier": "Mid", "builder": build_panel_smith3},
    {"mod": "auto_value_momentum_complementary_combination_smith2_96154",
     "label": "trend_overlay", "discovery_tier": "Small", "builder": build_panel_smith2},
]


def run_tier(M, builder, marketcap):
    """Full rails on one cap tier: in-sample CPCV bundle (median_cpcv/PBO/DSR) + OOS Sharpe."""
    panel, n_names = builder(M, marketcap)
    full_ret, trades = M.signal(panel, **M.SPEC.default_params)
    full_ret = pd.Series(full_ret).dropna()
    search = full_ret[full_ret.index < HOLDOUT]
    holdout = full_ret[full_ret.index >= HOLDOUT]

    grid = {}
    for label, kw in (M.SPEC.grid or {"default": {}}).items():
        try:
            r = pd.Series(M.signal(panel, **{**M.SPEC.default_params, **kw})[0]).dropna()
            grid[label] = r[r.index < HOLDOUT]
        except Exception as e:
            grid[label] = pd.Series(dtype=float)
            print(f"      grid '{label}' failed: {type(e).__name__}: {str(e)[:80]}")

    res = ri.assemble_bundle(search.values, trades, grid_returns=grid)
    b = (res.get("bundle") or {}) if isinstance(res, dict) else {}
    return {
        "tier": marketcap, "n_names": n_names, "n_trades": len(trades),
        "IS_sharpe": _sharpe(search), "OOS_sharpe": _sharpe(holdout), "full_sharpe": _sharpe(full_ret),
        "median_cpcv": round(float(b["median_cpcv_sharpe"]), 3) if b.get("median_cpcv_sharpe") is not None else None,
        "pbo": round(float(b["pbo"]), 3) if b.get("pbo") is not None else None,
        "dsr": round(float(b["dsr"]), 4) if b.get("dsr") is not None else None,
        "frac_paths_positive": round(float(b["frac_paths_positive"]), 3) if b.get("frac_paths_positive") is not None else None,
    }


def main():
    out = []
    for V in VARIANTS:
        M = importlib.import_module(f"strategies.{V['mod']}")
        print(f"\n=== {V['label']} ({V['mod']}) | discovery tier = {V['discovery_tier']} ===")
        rows = []
        for tier in TIERS:
            try:
                row = run_tier(M, V["builder"], tier)
            except Exception as e:
                row = {"tier": tier, "error": f"{type(e).__name__}: {str(e)[:160]}"}
                print(f"   {tier}: ERROR {row['error']}")
            rows.append(row)
            if "error" not in row:
                tag = "DISCOVERY" if tier == V["discovery_tier"] else "untouched"
                print(f"   {tier:<6} [{tag:<9}] n={row['n_names']:<4} IS={row['IS_sharpe']} "
                      f"OOS={row['OOS_sharpe']} | CPCV_med={row['median_cpcv']} "
                      f"PBO={row['pbo']} DSR={row['dsr']} fracpos={row['frac_paths_positive']}")
        # verdict: untouched tiers must be OOS-positive AND robust (median_cpcv>0, pbo<=0.5)
        untouched = [r for r in rows if "error" not in r and r["tier"] != V["discovery_tier"]]
        ok = [r for r in untouched
              if (r["OOS_sharpe"] or -9) > 0 and (r["median_cpcv"] or -9) > 0 and (r["pbo"] if r["pbo"] is not None else 9) <= 0.5]
        generalises = len(untouched) >= 2 and len(ok) == len(untouched)
        verdict = ("GENERALISES (broad factor, CPCV-hardened)" if generalises
                   else f"DOES NOT robustly generalise -> {len(ok)}/{len(untouched)} untouched tiers pass CPCV+OOS")
        print(f"   >>> {V['label']}: {verdict}")
        out.append({"variant": V["label"], "module": V["mod"], "discovery_tier": V["discovery_tier"],
                    "tiers": rows, "untouched_pass": f"{len(ok)}/{len(untouched)}", "verdict": verdict})

    rec = {"ts": datetime.now().isoformat(), "holdout": HOLDOUT,
           "method": "per-tier ri.assemble_bundle CPCV/PBO/DSR + OOS Sharpe (rails-faithful)",
           "results": out}
    with open("/root/hephaestus/forward/valmom_generalization_cpcv.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("\n=== FULL JSON ===")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
