"""forward/battery.py — the generic stage-2 battery (MCPT + cross-universe breadth).

amihud_battery.py and generalize_valmom_cpcv.py were ~80% this file; future batteries
should be a thin config over run_battery() instead of a fresh copy-paste evolution.
Uses the CANONICAL implementations: harness MCPT (MultiIndex-aware, correct-null,
parallel) and sdk.stats — never local re-definitions.

A battery's verdict is wiki-worthy evidence: results are written atomically to JSON
AND appended to the experiment's wiki page (O5 — stage-2 evidence must live in the
system's memory, not a gitignored directory).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from crucible_paths import ROOT, WIKI  # noqa: F401
from sdk.harness import _stage2_mcpt  # canonical MCPT (parallel, correct null)
from sdk.stats import sharpe, sharpe_or_none


def breadth_verdict(results: dict, min_frac: float = 0.60) -> tuple:
    """results = {universe: holdout_sharpe|None}. CONFIRM if >=min_frac of >=3 ran universes
    are OOS-positive. Returns (ok, note)."""
    vals = [v for v in results.values() if v is not None]
    pos = sum(1 for v in vals if v > 0)
    if len(vals) < 3:
        return False, f"INCONCLUSIVE: only {len(vals)} universes ran (need >=3)"
    frac = pos / len(vals)
    ok = frac >= min_frac
    return ok, (f"{pos}/{len(vals)} universes positive OOS ({frac:.0%}) -> "
                f"{'CONFIRMED (generalises)' if ok else 'REJECTED (overfit outlier)'}")


def run_battery(spec, panel_loader, universes: list, holdout_start: str,
                n_perms: int = 50, beta_to_universe: float | None = None,
                out_json: "Path | str | None" = None) -> dict:
    """MCPT first (it kills construction artifacts breadth can't see), then breadth.

    spec:          StrategySpec (frozen — default_params only, no re-search)
    panel_loader:  label -> panel (each universe UNTOUCHED + pre-declared)
    universes:     labels; the first is conventionally the discovery universe
    Returns the verdict dict (also written atomically to out_json if given).
    """
    t0 = time.time()
    res = {"id": spec.id, "started": pd.Timestamp.now().isoformat(),
           "n_perms": n_perms, "holdout": holdout_start}

    # real run on the discovery panel
    disc = panel_loader(universes[0])
    real = pd.Series(spec.signal(disc, **spec.default_params)[0]).dropna()
    res["real_full_sharpe"] = round(sharpe(real), 3)
    res["real_holdout_sharpe"] = round(sharpe(real[real.index >= holdout_start]), 3)

    # 1) MCPT — FIRST (the law: META-LESSONS "MCPT-before-breadth")
    mcpt_res, mcpt_pass = _stage2_mcpt(spec, disc, res["real_full_sharpe"],
                                       n=n_perms, beta_to_universe=beta_to_universe)
    res["mcpt"], res["mcpt_pass"] = mcpt_res, bool(mcpt_pass)

    # 2) breadth — only if MCPT passed (a construction artifact replicates everywhere;
    #    running breadth after an MCPT fail is wasted compute and false comfort)
    if mcpt_pass:
        gen = {}
        for u in universes[1:]:
            try:
                r_u = pd.Series(spec.signal(panel_loader(u), **spec.default_params)[0]).dropna()
                gen[u] = sharpe_or_none(r_u[r_u.index >= holdout_start])
            except Exception as e:
                gen[u] = None
                print(f"[battery] universe {u} failed: {type(e).__name__}: {str(e)[:120]}")
        ok, note = breadth_verdict(gen)
        res["generalization"], res["breadth_pass"], res["breadth_note"] = gen, ok, note
    else:
        res["generalization"], res["breadth_pass"] = None, None
        res["breadth_note"] = "skipped: MCPT failed (artifact — breadth would be false comfort)"

    res["verdict"] = "PASS" if (res["mcpt_pass"] and res.get("breadth_pass")) else "FAIL"
    res["elapsed_s"] = round(time.time() - t0, 1)

    if out_json:
        out_json = Path(out_json)
        fd, tmp = tempfile.mkstemp(dir=str(out_json.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(res, f, indent=2, default=str)
        os.replace(tmp, out_json)

    # O5: battery evidence goes to the wiki, not just a gitignored JSON
    page = WIKI / "experiments" / f"{spec.id}.md"
    if page.exists():
        with open(page, "a") as f:
            f.write(f"\n\n## Stage-2 battery ({pd.Timestamp.now():%Y-%m-%d})\n"
                    f"- MCPT: p={mcpt_res.get('p_value', mcpt_res.get('p_value_lb'))} "
                    f"(n={mcpt_res.get('n_ran')}) -> {'PASS' if mcpt_pass else 'FAIL'}\n"
                    f"- breadth: {res['breadth_note']}\n"
                    f"- verdict: **{res['verdict']}**\n")
    return res
