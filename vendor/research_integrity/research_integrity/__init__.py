"""research_integrity — shared research-integrity rails (portfolio-wide).

The methodology that caught Atlas's survivorship mirage + a DSR-0.986 holdout failure, packaged so
EVERY edge-search project (Hermes, Midas, Credibility, Atlas) is validated to the same standard
instead of reinventing the harness.

Three rails + a cross-OOS battery, all PURE functions over a per-period return series + trade list,
so they bolt onto ANY project's backtest/walk-forward output:

  cross-OOS battery : assemble_bundle(returns, trades, grid) -> evaluate_tiers(...)  [cpcv/PBO/DSR/regime]
  Rail 1 holdout    : holdout_gate(...) + config_hash + single-use ledger (quarantine train, test once)
  Rail 2 FDR bar    : promote_dsr(n_families) + registry (distinct_families) — across-hypothesis MTC
  Rail 3 deployment : deployment_sanity(trades, ...) — did the strategy actually deploy as designed

Per-project state (holdout config, ledger, registry) lives under $RESEARCH_INTEGRITY_DIR (default cwd).
`evaluate_holdout`/`deployment_smoke` in holdout.py/deployment.py are the ATLAS reference runners;
other projects supply their own runner (produce returns+trades) and call the gates. See README.md.
"""
from __future__ import annotations

from . import cpcv, gates, metrics, overfitting, splitters
from .adapter import (
    assemble_bundle, evaluate, evaluate_tiers, daily_returns,
    promote_dsr, SCREEN_DSR, PROMOTE_DSR, PROMOTE_DSR_CAP,
)
from .deployment import deployment_sanity, expected_positions
from .holdout import holdout_gate, config_hash, ledger_lookup, ledger_append, MIN_HOLDOUT_SHARPE, MAX_DEGRADATION_PCT
from . import registry
from .registry import family_of, distinct_families, append_run

__all__ = [
    "cpcv", "gates", "metrics", "overfitting", "splitters", "registry",
    "assemble_bundle", "evaluate", "evaluate_tiers", "daily_returns",
    "promote_dsr", "SCREEN_DSR", "PROMOTE_DSR", "PROMOTE_DSR_CAP",
    "deployment_sanity", "expected_positions",
    "holdout_gate", "config_hash", "ledger_lookup", "ledger_append",
    "MIN_HOLDOUT_SHARPE", "MAX_DEGRADATION_PCT",
    "family_of", "distinct_families", "append_run",
]


def configure(state_dir):
    """Point holdout-ledger + FDR-registry state at an explicit directory (overrides the
    RESEARCH_INTEGRITY_DIR env var). Call BEFORE any ledger/registry operation."""
    from pathlib import Path as _Path
    from . import holdout as _h
    _h._OVERRIDE_DIR = _Path(state_dir)
