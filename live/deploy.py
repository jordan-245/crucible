"""hephaestus/live/deploy.py — bridge a forge PASS into the Atlas Paper Book.

Computes TODAY's target weights from a strategy's signal (the positions still held at the latest ledger date),
writes them to the Atlas contract file (``/root/atlas/data/live/<name>/target.json``), and registers the strategy
via Atlas's ``deploy_pass``. The Atlas daily loop then paper-trades it on live data. PASS -> paper is autonomous
(no real capital; promotion to real capital stays human-gated — board 2026-06-09).

The weight extraction is GENERIC: every forge strategy emits the same entry/exit/position_value trade ledger for
the research-integrity rails, so "held at the latest date" recovers today's book without per-strategy code.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ATLAS = Path("/root/atlas")
ATLAS_LIVE = ATLAS / "data" / "live"
HEPH = Path(__file__).resolve().parents[1]


def extract_target_weights(trades: list) -> dict:
    """Today's target book = positions held at the latest ledger date, normalized to sum(|w|)=1."""
    if not trades:
        return {}
    last = max(t["exit_date"] for t in trades)
    raw: dict = {}
    for t in trades:
        if t.get("exit_date") == last and float(t.get("position_value", 0) or 0) != 0:
            raw[t["ticker"]] = raw.get(t["ticker"], 0.0) + float(t["position_value"])
    tot = sum(abs(v) for v in raw.values()) or 1.0
    return {k: round(v / tot, 6) for k, v in raw.items()}


def _load_spec(strategy_path: str):
    p = Path(strategy_path)
    if str(HEPH) not in sys.path:
        sys.path.insert(0, str(HEPH))                 # so the strategy's `from sdk...` resolves
    s = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod.SPEC


def _run_signal(spec):
    panel = spec.load_data()
    net, trades = spec.signal(panel, **spec.default_params)
    return net, trades


def compute_target_weights(spec) -> dict:
    _net, trades = _run_signal(spec)
    return extract_target_weights(trades)


def compute_expectation(net, holdout_start: str = "2022-01-01") -> dict:
    """Modeled daily-return distribution for the track-vs-expectation gate. Use the HOLDOUT
    (un-optimized) slice when it has enough data — the honest forward estimate, fewer false 'diverging' flags."""
    import numpy as np, pandas as pd
    r = pd.Series(net).dropna()
    try:
        h = r[r.index >= holdout_start]
        if len(h) >= 60:
            r = h
    except Exception:
        pass
    if len(r) < 30 or float(r.std()) == 0:
        return {}
    return {"daily_mean": round(float(r.mean()), 8), "daily_std": round(float(r.std()), 8),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3)}


def write_target(name: str, weights: dict, strategy_path: str) -> Path:
    d = ATLAS_LIVE / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "target.json"
    f.write_text(json.dumps({"asof": datetime.date.today().isoformat(), "weights": weights,
                             "strategy_path": strategy_path}, indent=2))
    return f


def deploy_to_paper(strategy_path: str, *, name: Optional[str] = None, capital: float = 100000.0) -> dict:
    spec = _load_spec(strategy_path)
    name = name or spec.id.replace("-", "_")
    net, trades = _run_signal(spec)
    weights = extract_target_weights(trades)
    exp = compute_expectation(net, getattr(spec, "holdout_start", "2022-01-01"))
    write_target(name, weights, strategy_path)
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '/root/atlas'); from live.providers import deploy_pass; "
         f"deploy_pass({name!r}, capital={capital}, broker='alpaca', expectation={exp!r}, strategy_path={strategy_path!r})"],
        cwd=str(ATLAS), check=False,
    )
    return {"name": name, "n_positions": len(weights), "expectation": exp, "weights": weights}


def refresh_all() -> list:
    """Recompute target.json for every deployed paper strategy (read from the Atlas registry)."""
    out, reg_f = [], ATLAS / "config" / "live_strategies.json"
    reg = json.loads(reg_f.read_text()) if reg_f.exists() else []
    for s in reg:
        meta_f = ATLAS_LIVE / s["name"] / "meta.json"
        sp = json.loads(meta_f.read_text()).get("strategy_path") if meta_f.exists() else None
        if not sp or not Path(sp).exists():
            continue
        try:
            write_target(s["name"], compute_target_weights(_load_spec(sp)), sp)
            out.append(s["name"])
        except Exception as e:
            out.append(f"{s['name']}:ERR:{str(e)[:60]}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["deploy", "refresh"])
    ap.add_argument("--path", help="strategy file (for deploy)")
    ap.add_argument("--name")
    ap.add_argument("--capital", type=float, default=10000.0)
    a = ap.parse_args()
    if a.cmd == "deploy":
        print(deploy_to_paper(a.path, name=a.name, capital=a.capital))
    else:
        print("refreshed:", refresh_all())
