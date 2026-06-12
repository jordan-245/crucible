"""live/deploy.py — bridge a crucible PASS into a paper-trading execution host.

DEPLOY CONTRACT (what a target directory must provide — Atlas is the reference implementation):
  1. ``<target>/data/live/<name>/target.json``   — we WRITE today's target weights here
  2. ``atlas.execution.providers.deploy_pass(name, *, capital, broker, expectation, strategy_path)`` in <target>
     — we CALL this to register the strategy with the host's daily paper loop
  3. ``<target>/config/live_strategies.json``    — we READ the registry for refresh_all()
Set CRUCIBLE_DEPLOY to point at any host implementing this contract, or "" to disable deployment
(verdicts still record; nothing is paper-traded).

Computes TODAY's target weights from a strategy's signal (the positions still held at the latest ledger
date). PASS -> paper is autonomous (no real capital; promotion to real capital stays human-gated).

The weight extraction is GENERIC: every crucible strategy emits the same entry/exit/position_value trade
ledger for the research-integrity rails, so "held at the latest date" recovers today's book without
per-strategy code.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Script-mode safety: `python3 live/deploy.py` puts live/ (not the repo root) on sys.path,
# so `crucible_paths` wouldn't resolve — this broke the first automated forward-paper run
# (2026-06-10 23:45, 'weight refresh FAILED'). Make the import work from any caller.
_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from crucible_paths import DEPLOY_TARGET as ATLAS
ATLAS_LIVE = (ATLAS / "data" / "live") if ATLAS else None
HEPH = Path(__file__).resolve().parents[1]


def _require_target():
    if ATLAS is None:
        raise RuntimeError("CRUCIBLE_DEPLOY is unset/empty — paper deployment disabled (verdicts unaffected)")


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


from contextlib import contextmanager


@contextmanager
def _live_lock(timeout: float = 30.0):
    """flock over <atlas>/data/live/.lock — same contract as atlas.kernel.lockfile.live_lock.
    DELIBERATELY duplicated (~10 lines) rather than imported across the repo seam
    (2026-06-12 review: no cross-repo Python imports; the lock FILE is the contract)."""
    import fcntl
    import time
    lock = ATLAS_LIVE / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock, "w")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"live lock busy > {timeout}s — another writer wedged")
                time.sleep(0.2)
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


def write_target(name: str, weights: dict, strategy_path: str) -> Path:
    _require_target()
    d = ATLAS_LIVE / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "target.json"
    tmp = f.with_suffix(".json.tmp")
    with _live_lock():
        tmp.write_text(json.dumps({"asof": datetime.date.today().isoformat(), "weights": weights,
                                   "strategy_path": strategy_path}, indent=2))
        tmp.replace(f)
    return f


# Default paper allocation for an auto-deployed PASS. $5K is the pre-registered "deployable at $5K"
# design size every smith hypothesis is required to honor; 2026-06-12 lesson: the old 100_000 default
# deployed tranched_v3 at 7x account equity -> 369/498 orders rejected (insufficient buying power).
DEFAULT_CAPITAL = float(os.environ.get("CRUCIBLE_PAPER_CAPITAL", "5000"))


def deploy_to_paper(strategy_path: str, *, name: Optional[str] = None, capital: float = DEFAULT_CAPITAL,
                    broker: str = os.environ.get("CRUCIBLE_BROKER", "alpaca"),
                    tif: str = "opg") -> dict:
    _require_target()
    spec = _load_spec(strategy_path)
    name = name or spec.id.replace("-", "_")
    net, trades = _run_signal(spec)
    weights = extract_target_weights(trades)
    exp = compute_expectation(net, getattr(spec, "holdout_start", "2022-01-01"))
    write_target(name, weights, strategy_path)
    # #34 one-direction contract: the deploy REQUEST artifact. During the soak the subprocess
    # registration below still runs (behavior unchanged); atlas's intake consumer verifies in
    # shadow mode that the artifact alone would produce the same registry entry. After a clean
    # soak the subprocess dies and the artifact becomes the only channel.
    req = {"schema_version": 1, "name": name, "strategy_path": strategy_path,
           "capital": capital, "broker": broker, "tif": tif, "expectation": exp,
           "requested_at": datetime.datetime.now().isoformat(timespec="seconds")}
    d = ATLAS_LIVE / name
    with _live_lock():
        tmp = d / "deploy_request.json.tmp"
        tmp.write_text(json.dumps(req, indent=2))
        tmp.replace(d / "deploy_request.json")
    # check=True + captured output + Telegram on failure (2026-06-12 review finding: check=False
    # left an orphaned half-deploy possible — target.json written but no registry entry — silently).
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(ATLAS)!r}); from atlas.execution.providers import deploy_pass; "
         f"deploy_pass({name!r}, capital={capital}, broker={broker!r}, expectation={exp!r}, "
         f"strategy_path={strategy_path!r}, tif={tif!r})"],
        cwd=str(ATLAS), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        from sdk.notify import telegram_msg
        err = (proc.stderr or proc.stdout or "").strip()[-500:]
        telegram_msg(f"\u26a0\ufe0f DEPLOY REGISTRATION FAILED: {name}\n"
                     f"target.json was written but atlas registry append failed — "
                     f"ORPHANED HALF-DEPLOY, fix before next daily cycle.\n{err}")
        raise RuntimeError(f"deploy_pass registration failed for {name}: {err}")
    return {"name": name, "n_positions": len(weights), "expectation": exp, "weights": weights}


def refresh_all() -> list:
    """Recompute target.json for every deployed paper strategy (read from the host registry)."""
    _require_target()
    out, reg_f = [], ATLAS / "config" / "live_strategies.json"
    reg = json.loads(reg_f.read_text(encoding="utf-8")) if reg_f.exists() else []
    for s in reg:
        meta_f = ATLAS_LIVE / s["name"] / "meta.json"
        sp = json.loads(meta_f.read_text(encoding="utf-8")).get("strategy_path") if meta_f.exists() else None
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
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    a = ap.parse_args()
    if a.cmd == "deploy":
        print(deploy_to_paper(a.path, name=a.name, capital=a.capital))
    else:
        print("refreshed:", refresh_all())
