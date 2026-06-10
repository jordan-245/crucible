"""signal_kit tests: lookahead-safety, cost-model correctness, ledger equivalence
with the deployed val_mom implementation's semantics, PIT discipline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel  # noqa: E402
from sdk.stats import sharpe, sharpe_or_none, maxdd, split_holdout  # noqa: E402


@pytest.fixture()
def rw():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=500)
    rets = pd.DataFrame(rng.normal(0, 0.01, (500, 6)), index=idx,
                        columns=list("ABCDEF"))
    return idx, rets


def test_xs_zscore_winsorizes_and_preserves_nan(rw):
    idx, rets = rw
    df = rets.copy()
    df.iloc[10, 0] = 50.0          # absurd outlier
    df.iloc[20, 3] = np.nan
    z = xs_zscore(df)
    assert abs(z.iloc[10, 0]) < 5, "winsorization must bound the outlier's z"
    assert np.isnan(z.iloc[20, 3]), "NaN input must stay NaN (no fake neutral score)"
    row = z.iloc[40].dropna()
    assert row.mean() == pytest.approx(0, abs=1e-9)
    assert row.std() == pytest.approx(1, abs=1e-9)


def test_net_of_cost_charges_turnover(rw):
    idx, rets = rw
    W = pd.DataFrame(1 / 6, index=idx, columns=rets.columns).shift(1)
    static = net_of_cost(W, rets, cost_bps=8.0)
    # static book after day 1: zero turnover -> net == gross
    gross = (W * rets).sum(axis=1)
    assert np.allclose(static.iloc[5:], gross.iloc[5:].fillna(0))
    # daily full flip: 2.0 turnover/day -> exactly 1.6bps/day drag
    Wflip = W.copy()
    Wflip.iloc[::2] = -Wflip.iloc[::2]
    flip = net_of_cost(Wflip, rets, cost_bps=8.0)
    drag = ((Wflip * rets).sum(axis=1) - flip).iloc[10:]
    assert (drag.round(8) >= 0).all() and drag.max() == pytest.approx(2 * 8e-4, rel=0.01)


def test_trades_ledger_run_lengths(rw):
    idx, rets = rw
    W = pd.DataFrame(0.0, index=idx, columns=rets.columns)
    W.iloc[10:20, 0] = 0.1     # A held 10 days
    W.iloc[30:32, 1] = -0.2    # B short 2 days
    trades = trades_from_weights(W, rets, {"A": "Tech", "B": "Energy"})
    assert len(trades) == 2
    a = next(t for t in trades if t["ticker"] == "A")
    assert a["hold_days"] == 10 and a["sector"] == "Tech"
    assert a["entry_date"] == idx[10].strftime("%Y-%m-%d")
    assert a["exit_date"] == idx[19].strftime("%Y-%m-%d")
    b = next(t for t in trades if t["ticker"] == "B")
    assert b["hold_days"] == 2 and b["position_value"] < 0


def test_pit_panel_no_lookahead():
    """A Q4 number filed Feb-15 must be NaN before Feb-15 and visible from Feb-15."""
    sf1 = pd.DataFrame({
        "ticker": ["X", "X"],
        "datekey": [pd.Timestamp("2021-02-15"), pd.Timestamp("2021-05-10")],
        "bvps": [10.0, 12.0],
    })
    dates = pd.bdate_range("2021-01-01", "2021-06-30")
    p = pit_panel(sf1, "bvps", dates, ["X"])
    assert p.loc["2021-02-12", "X"] != p.loc["2021-02-12", "X"] or np.isnan(p.loc["2021-02-12", "X"])
    assert p.loc["2021-02-15", "X"] == 10.0
    assert p.loc["2021-05-07", "X"] == 10.0   # old value until next filing
    assert p.loc["2021-05-10", "X"] == 12.0


def test_stats_canonical():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000),
                  index=pd.bdate_range("2020-01-01", periods=1000))
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert sharpe_or_none(pd.Series([0.01] * 5)) is None, "too few obs -> None not 0"
    assert maxdd(pd.Series([0.1, -0.5, 0.0])) == pytest.approx(-0.5)
    s, h = split_holdout(r, "2022-01-01")
    assert s.index.max() < pd.Timestamp("2022-01-01") <= h.index.min()


def test_ledger_matches_deployed_valmom_semantics(rw):
    """trades_from_weights must reproduce the deployed val_mom module's ledger logic
    (same run-length, same pnl/position_value accounting) on a shared input."""
    idx, rets = rw
    rng = np.random.default_rng(5)
    W = pd.DataFrame(rng.choice([0.0, 0.05, -0.05], (500, 6), p=[0.7, 0.2, 0.1]),
                     index=idx, columns=rets.columns)
    smap = {c: "S" for c in rets.columns}
    kit = trades_from_weights(W, rets, smap, book=1_000_000.0)

    # reference: the deployed implementation's inner loop, verbatim semantics
    ref = []
    W_arr, R_arr = W.fillna(0.0).values, rets.fillna(0.0).values
    dstr = [d.strftime("%Y-%m-%d") for d in idx]
    for cj, t in enumerate(W.columns):
        col = W_arr[:, cj]
        mask = np.abs(col) > 1e-6
        i, n = 0, len(col)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                ref.append((t, dstr[i], dstr[j], int(j - i + 1),
                            float(np.nanmean(col[i:j + 1]) * 1e6),
                            float(np.nansum(col[i:j + 1] * R_arr[i:j + 1, cj]) * 1e6)))
                i = j + 1
            else:
                i += 1
    assert len(kit) == len(ref)
    for k, r in zip(kit, ref):
        assert (k["ticker"], k["entry_date"], k["exit_date"], k["hold_days"]) == r[:4]
        assert k["position_value"] == pytest.approx(r[4])
        assert k["pnl"] == pytest.approx(r[5])
