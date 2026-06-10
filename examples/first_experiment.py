"""Your first experiment — a hand-written strategy through the full gate stack.

No LLM, no paid data, no execution host: just yfinance + the rails. Run:

    python3 examples/first_experiment.py

This is the same `run_experiment()` the autonomous smiths call. Expect a verdict dict
and (if CRUCIBLE_WIKI is set up) a wiki page under experiments/. A FAIL verdict is the
normal, honest outcome for a simple demo signal — the point is watching the gates work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sdk.adapters import yf_panel
from sdk.harness import StrategySpec, run_experiment

TICKERS = ["SPY", "TLT", "GLD", "USO", "FXE", "EEM", "IWM", "QQQ", "XLE", "XLU"]


def load_data() -> pd.DataFrame:
    return yf_panel(TICKERS, start="2010-01-01")


def signal(panel: pd.DataFrame, lookback: int = 126, **_) -> tuple:
    """Toy cross-asset time-series momentum: long what's up over `lookback` days, short what's
    down, inverse-vol weighted, lagged one day (no look-ahead). Returns (daily_returns, trades)."""
    rets = panel.pct_change()
    mom = panel.pct_change(lookback)
    vol = rets.rolling(63).std()
    w = (np.sign(mom) / vol)
    w = w.div(w.abs().sum(axis=1), axis=0).shift(1)          # normalize then LAG
    daily = (w * rets).sum(axis=1)

    # trade ledger for deployment-sanity (monthly snapshots of the book)
    trades = []
    for dt, row in w.resample("ME").last().dropna(how="all").iterrows():
        for tkr, wt in row.dropna().items():
            if abs(wt) > 0.01:
                trades.append({"ticker": tkr, "entry_date": str(dt.date()),
                               "exit_date": str(dt.date()), "position_value": float(wt) * 100_000,
                               "pnl": 0.0, "sector": tkr})
    return daily, trades


SPEC = StrategySpec(
    id="example-tsmom-etf",
    family="example_tsmom",
    title="Example: cross-asset ETF time-series momentum (demo)",
    markets=["etf"],
    data_desc="FREE — yfinance, 10 liquid ETFs, 2010+",
    pre_registration=(
        "FROZEN BEFORE RUNNING: 126d momentum sign, 63d inverse-vol weights, 1-day lag, "
        "daily rebalance, 10-ETF universe as listed. Grid = lookback {63, 126, 252} for "
        "honest DSR search-burden. Holdout 2022+. PASS/FAIL accepted as-is."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 126},
    grid={"lb63": {"lookback": 63}, "lb252": {"lookback": 252}},
    holdout_start="2022-01-01",
    deploy_max_positions=10,
    scope="local",  # demo: no generalization universes declared
)

if __name__ == "__main__":
    verdict = run_experiment(SPEC, write_wiki=True, alert=False)
    print("\n=== VERDICT ===")
    for k in ["tier", "dsr", "promote_bar", "pbo", "holdout_sharpe", "holdout_pass",
              "beta_confound", "mcpt_pass", "stage1_pass", "PASSED_ALL_GATES"]:
        print(f"  {k}: {verdict.get(k)}")
