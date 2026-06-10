"""STAGE-2 battery for amihud_illiquidity_smallcap (the 2026-06-09 PROMOTE-tier CANDIDATE).

Lesson order (value×mom 0.994 post-mortem): MCPT FIRST — it catches what generalization missed
(a confound that replicates across universes still fails on structureless data). Then breadth.

1) MCPT (N perms): permute each asset's PRICE returns (volume kept aligned — Amihud's denominator
   is px*vol so dollar-volume re-derives consistently), re-run the frozen signal, p = P(perm >= real).
   PASS bar: p <= 0.05.
2) Cross-universe generalization: the SAME frozen signal (default params, zero re-search) on 3
   DISJOINT untouched universes within the pre-registration's "mid/large/sector slices":
     mid_a   = Mid-cap, even-indexed sectors (6)
     mid_b   = Mid-cap, odd-indexed sectors (5)   (disjoint from mid_a by construction)
     large   = Large-cap, all sectors
   Score ONLY the untouched holdout (>= 2022-01-01). Breadth verdict: >=60% of >=3 positive.

CONFIRMED = MCPT pass AND breadth pass. Report-ALL discipline: every number lands in the JSON
regardless of outcome. Results -> forward/amihud_battery_results.json (atomic write at the end).
"""
import importlib
import json
import sys
import time

sys.path.insert(0, "/root/hephaestus")
import numpy as np
import pandas as pd

MOD = "strategies.auto_amihud_illiquidity_risk_premium_in_survi_smith2_99439"
HOLDOUT = "2022-01-01"
N_PERMS = 50
OUT = "/root/hephaestus/forward/amihud_battery_results.json"


def sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if len(r) > 20 and r.std() > 0 else 0.0


def permute_amihud_panel(panel, rng):
    """Permute each asset's daily price returns (destroys serial + cross-sectional structure,
    keeps marginal distribution); rebuild prices; keep the volume block as-is."""
    px = panel["price"]
    rets = px.pct_change()
    out = {}
    for c in rets.columns:
        s = rets[c]
        mask = s.notna()
        perm = s.copy()
        perm[mask] = rng.permutation(s[mask].values)
        out[c] = perm
    sh = pd.DataFrame(out, index=rets.index)
    px_perm = (1 + sh.fillna(0)).cumprod()
    px_perm = px_perm.div(px_perm.bfill().iloc[0]).mul(px.bfill().iloc[0])
    px_perm = px_perm.where(px.notna())  # preserve the listing/delisting NaN shape
    new = pd.concat({"price": px_perm, "volume": panel["volume"]}, axis=1)
    new.attrs["sector_map"] = panel.attrs.get("sector_map", {})
    return new


def build_universe_panel(m, marketcap, sectors):
    """Mirror the candidate's load_data at a chosen cap tier / sector subset."""
    from sdk.adapters import sep_panel, us_universe
    smap = {}
    for s in sectors:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             marketcap=marketcap, include_delisted=True, top_n=120)
        except Exception:
            ts = []
        for t in (ts or []):
            smap.setdefault(t, s)
    if len(smap) < 60:
        raise RuntimeError(f"universe too small: {marketcap}/{len(sectors)} sectors -> {len(smap)} names")
    tickers = sorted(smap)
    px = sep_panel(tickers, m.START, field="closeadj").sort_index()
    vol = sep_panel(tickers, m.START, field="volume").reindex(index=px.index, columns=px.columns)
    panel = pd.concat({"price": px, "volume": vol}, axis=1)
    panel.attrs["sector_map"] = smap
    return panel


def main():
    t0 = time.time()
    m = importlib.import_module(MOD)
    res = {"id": "amihud_illiquidity_smallcap", "module": MOD, "started": pd.Timestamp.now().isoformat(),
           "n_perms": N_PERMS, "holdout": HOLDOUT}

    # ---- real run on the discovery (small-cap) panel ----
    print("[battery] loading small-cap discovery panel...", flush=True)
    panel = m.load_data()
    real_ret = pd.Series(m.signal(panel, **m.SPEC.default_params)[0]).dropna()
    res["real_full_sharpe"] = round(sharpe(real_ret), 3)
    res["real_holdout_sharpe"] = round(sharpe(real_ret[real_ret.index >= HOLDOUT]), 3)
    print(f"[battery] real: full {res['real_full_sharpe']} | holdout {res['real_holdout_sharpe']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 1) MCPT ----
    rng = np.random.default_rng(0)
    perms = []
    for i in range(N_PERMS):
        try:
            p = permute_amihud_panel(panel, rng)
            perms.append(sharpe(m.signal(p, **m.SPEC.default_params)[0]))
        except Exception as e:
            print(f"[battery] perm {i} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        if (i + 1) % 10 == 0:
            arr = np.array(perms)
            print(f"[battery] MCPT {i+1}/{N_PERMS} | perm mean {arr.mean():.2f} max {arr.max():.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(perms) if perms else np.array([0.0])
    real = res["real_full_sharpe"]
    pval = float((np.sum(arr >= real) + 1) / (len(arr) + 1))
    res["mcpt"] = {"n_ran": len(perms), "perm_mean": round(float(arr.mean()), 3),
                   "perm_p95": round(float(np.percentile(arr, 95)), 3),
                   "perm_max": round(float(arr.max()), 3), "p_value": round(pval, 4),
                   "pass": pval <= 0.05}
    print(f"[battery] MCPT: p={pval:.4f} -> {'PASS' if pval <= 0.05 else 'FAIL'}", flush=True)
    del panel  # free the discovery panel before loading generalization universes

    # ---- 2) cross-universe generalization (3 disjoint pre-registered slices) ----
    S = m.SECTORS
    universes = {"mid_a": ("Mid", S[0::2]), "mid_b": ("Mid", S[1::2]), "large": ("Large", S)}
    gen = {}
    for name, (cap, secs) in universes.items():
        try:
            print(f"[battery] generalization universe {name} ({cap}, {len(secs)} sectors)...", flush=True)
            up = build_universe_panel(m, cap, secs)
            r_u = pd.Series(m.signal(up, **m.SPEC.default_params)[0]).dropna()
            h = r_u[r_u.index >= HOLDOUT]
            gen[name] = {"n_names": int(up["price"].shape[1]),
                         "insample_sharpe": round(sharpe(r_u[r_u.index < HOLDOUT]), 2),
                         "holdout_sharpe": round(sharpe(h), 2) if len(h) > 20 else None}
            print(f"[battery]   {name}: IS {gen[name]['insample_sharpe']} | OOS {gen[name]['holdout_sharpe']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del up
        except Exception as e:
            gen[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(f"[battery]   {name} FAILED: {gen[name]['error']}", flush=True)
    res["generalization"] = gen
    oos = [u["holdout_sharpe"] for u in gen.values() if isinstance(u, dict) and u.get("holdout_sharpe") is not None]
    pos = sum(1 for v in oos if v > 0)
    breadth_pass = len(oos) >= 3 and pos / len(oos) >= 0.60
    res["breadth"] = {"ran": len(oos), "positive": pos,
                      "pass": breadth_pass,
                      "note": (f"{pos}/{len(oos)} untouched universes positive OOS"
                               if len(oos) >= 3 else f"INCONCLUSIVE: only {len(oos)}/3 universes ran")}

    res["CONFIRMED"] = bool(res["mcpt"]["pass"] and breadth_pass)
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["finished"] = pd.Timestamp.now().isoformat()
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    import os
    os.replace(tmp, OUT)
    print(f"[battery] DONE in {res['elapsed_s']}s -> CONFIRMED={res['CONFIRMED']} | {OUT}", flush=True)


if __name__ == "__main__":
    main()
