"""Rail 1 — write-once holdout partition (SHARED research-integrity infra).

PROJECT-AGNOSTIC: holdout_gate/config_hash/ledger are pure + reusable. `evaluate_holdout` is the
ATLAS reference runner (lazy-imports the Atlas engine) — each project writes its OWN runner that
produces holdout-period returns+trades, then calls holdout_gate(). Paths are RESEARCH_INTEGRITY_DIR-
configurable. Original:

Battery SEARCH runs are quarantined to data strictly before `holdout_start` (the loop physically
cannot read holdout rows during search). A candidate that reaches PROMOTE is evaluated on the
holdout EXACTLY ONCE via `evaluate_holdout` — enforced by an append-only single-use ledger so a
config cannot be iterated against the holdout. A PROMOTE that degrades on the holdout is downgraded
to FAIL and burned.

Spec: research/INTEGRITY_RAILS_SPEC.md (Rail 1).
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import os
PROJECT = Path(__file__).resolve().parents[2]
_OVERRIDE_DIR = None  # set via research_integrity.configure(state_dir)


def _state_dir() -> Path:
    """Resolve the state dir LAZILY (per call), so import order can never silently point
    holdout/registry state at os.getcwd() — the M5 config-by-import-order trap."""
    return _OVERRIDE_DIR or Path(os.environ.get("RESEARCH_INTEGRITY_DIR", os.getcwd()))


class _LazyPath:
    """Path-like whose location is resolved at USE time (back-compat for module constants)."""
    def __init__(self, name): self._name = name
    def _p(self) -> Path: return _state_dir() / self._name
    def __getattr__(self, a): return getattr(self._p(), a)
    def __fspath__(self): return str(self._p())
    def __str__(self): return str(self._p())
    def __truediv__(self, o): return self._p() / o


HOLDOUT_CFG = _LazyPath("holdout.json")          # per-project: set RESEARCH_INTEGRITY_DIR or configure()
LEDGER = _LazyPath("holdout_ledger.jsonl")       # write-once single-use ledger (per project)

# Pre-registered holdout-gate thresholds (frozen).
MIN_HOLDOUT_SHARPE = 0.0           # must be net-positive on truly unseen data
MAX_DEGRADATION_PCT = -50.0        # holdout Sharpe may fall at most 50% vs search


def load_holdout_config() -> Optional[dict]:
    if not HOLDOUT_CFG.exists():
        return None
    try:
        return json.load(open(HOLDOUT_CFG))
    except Exception:
        return None


def holdout_start_ts() -> Optional[pd.Timestamp]:
    cfg = load_holdout_config()
    if cfg and cfg.get("holdout_start"):
        return pd.Timestamp(cfg["holdout_start"])
    return None


def config_hash(strategy: str, primary_config: Optional[dict], market: str) -> str:
    payload = json.dumps({"s": strategy, "m": market, "p": primary_config or {}},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def ledger_lookup(h: str) -> Optional[dict]:
    if not LEDGER.exists():
        return None
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("config_hash") == h:
            return rec
    return None


def ledger_append(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def holdout_gate(holdout_sharpe: float, degradation_pct: Optional[float],
                 deployment_passed: bool) -> Tuple[bool, List[str]]:
    """Pure gate logic (testable without a backtest)."""
    reasons: List[str] = []
    if not (holdout_sharpe == holdout_sharpe and holdout_sharpe > MIN_HOLDOUT_SHARPE):
        reasons.append(f"holdout_sharpe {holdout_sharpe:.3f} <= {MIN_HOLDOUT_SHARPE}")
    if degradation_pct is not None and degradation_pct < MAX_DEGRADATION_PCT:
        reasons.append(f"degradation {degradation_pct:.1f}% < {MAX_DEGRADATION_PCT}%")
    if not deployment_passed:
        reasons.append("holdout deployment-sanity FAIL")
    return (len(reasons) == 0), reasons


def evaluate_holdout(strategy: str, primary_config: Optional[dict], market: str = "sp500",
                     max_positions: int = 35, search_sharpe: Optional[float] = None,
                     allow_reuse: bool = False) -> Dict[str, Any]:
    """Run the FROZEN primary config on the quarantined holdout ONCE. Single-use enforced.

    Returns a record dict with `ok`, `passed`, holdout metrics, and `gate_reasons`.
    """
    import sys
    sys.path.insert(0, str(PROJECT))
    import scripts.validate_oos as vo
    from backtest.engine import BacktestEngine
    from scripts.strategy_evaluator import STRATEGY_REGISTRY, load_sandbox_strategy
    from utils.config import get_active_config
    from research.cross_oos import metrics as cm
    from research.cross_oos.deployment import deployment_sanity

    hs = holdout_start_ts()
    if hs is None:
        return {"ok": False, "reason": "no config/holdout.json configured"}

    h = config_hash(strategy, primary_config, market)
    prior = ledger_lookup(h)
    if prior and not allow_reuse:
        return {"ok": False, "reason": "single-use: holdout already evaluated for this config",
                "config_hash": h, "prior": prior}

    cfg_h = load_holdout_config() or {}
    warm_days = int(cfg_h.get("warmup_days", 400))

    data = vo.load_data(market=market)
    data = {k: v for k, v in data.items() if len(v) >= 260}
    warm = hs - pd.Timedelta(days=warm_days)
    d_hold = {k: v[v.index >= warm] for k, v in data.items() if len(v[v.index >= warm]) >= 60}
    if not d_hold:
        return {"ok": False, "reason": "no data in holdout window", "config_hash": h}

    base = get_active_config(market)
    base.setdefault("strategies", {})[strategy] = {"enabled": True, **(primary_config or {})}
    base.setdefault("risk", {})["max_open_positions"] = max_positions
    cls = STRATEGY_REGISTRY.get(strategy) or load_sandbox_strategy(strategy)
    res = BacktestEngine(base).run_walkforward(d_hold, [cls(base)])

    ec = pd.Series(res.equity_curve, dtype=float).dropna()
    ec_h = ec[ec.index >= hs]
    r_h = ec_h.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    holdout_sharpe = float(cm.annualized_sharpe(r_h.to_numpy())) if len(r_h) > 10 else float("nan")
    cum = float(ec_h.iloc[-1] / ec_h.iloc[0] - 1.0) if len(ec_h) > 1 else float("nan")
    trades_h = [t for t in res.trades
                if t.get("entry_date") is not None and pd.Timestamp(t["entry_date"]) >= hs]
    dep = deployment_sanity(trades_h, primary_config=primary_config,
                            strategy_meta={"max_positions": max_positions,
                                           "max_sector_concentration": base.get("risk", {}).get("max_sector_concentration", 2)})
    deg = None
    if search_sharpe is not None and abs(search_sharpe) > 1e-9 and holdout_sharpe == holdout_sharpe:
        deg = round((holdout_sharpe - search_sharpe) / abs(search_sharpe) * 100.0, 1)

    passed, reasons = holdout_gate(holdout_sharpe, deg, dep["passed"])

    rec = {
        "ok": True, "ts": datetime.datetime.now().isoformat(),
        "strategy": strategy, "market": market, "config_hash": h,
        "primary_config": primary_config or {}, "holdout_start": str(hs.date()),
        "holdout_sharpe": round(holdout_sharpe, 4) if holdout_sharpe == holdout_sharpe else None,
        "holdout_cum_return": round(cum, 4) if cum == cum else None,
        "holdout_trades": len(trades_h),
        "search_sharpe": search_sharpe, "degradation_vs_search_pct": deg,
        "deployment_passed": dep["passed"], "deployment": dep,
        "passed": passed, "gate_reasons": reasons,
    }
    if not (prior and allow_reuse):
        ledger_append({k: rec[k] for k in (
            "ts", "strategy", "market", "config_hash", "primary_config", "holdout_start",
            "holdout_sharpe", "holdout_cum_return", "holdout_trades",
            "degradation_vs_search_pct", "deployment_passed", "passed", "gate_reasons")})
    return rec


__all__ = ["load_holdout_config", "holdout_start_ts", "config_hash", "ledger_lookup",
           "ledger_append", "holdout_gate", "evaluate_holdout",
           "MIN_HOLDOUT_SHARPE", "MAX_DEGRADATION_PCT"]
