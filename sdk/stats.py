"""sdk/stats.py — THE canonical stats helpers. One definition, everywhere.

History: `sharpe` was independently re-defined in EIGHT files with subtly different
rounding (none/2dp/3dp) and min-length rules (none/len>20) — divergent stats helpers
in a statistics shop is how inconsistent verdicts happen. The harness's definition is
canonical (no min-length, no rounding — presentation belongs at the call site).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(r, ann: int = 252) -> float:
    """Annualized Sharpe. CANONICAL (harness definition): no min-length gate, no rounding.
    Insufficient/degenerate data -> 0.0."""
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def sharpe_or_none(r, ann: int = 252, min_obs: int = 20, ndigits: int | None = 2):
    """Battery/report variant: None when there's too little data to mean anything
    (so tables show a hole, not a fake 0), optional rounding for display."""
    r = pd.Series(r).dropna()
    if len(r) <= min_obs or r.std() == 0:
        return None
    s = float(r.mean() / r.std() * np.sqrt(ann))
    return round(s, ndigits) if ndigits is not None else s


def maxdd(r) -> float:
    """Max drawdown of a daily-returns series (negative number, e.g. -0.23)."""
    eq = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def split_holdout(r, holdout_start: str):
    """(search, holdout) split of a DatetimeIndex returns series at the quarantine date."""
    r = pd.Series(r).dropna()
    return r[r.index < holdout_start], r[r.index >= holdout_start]
