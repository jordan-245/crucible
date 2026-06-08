"""Agent-proposed strategy: VRP + Trend two-premium book.
VRP leg = short VXX (rolling ~30d VIX futures = the volatility risk premium, pro-cyclical, crash tail).
Trend leg = the validated Boreas 21-market TSMOM (crisis-alpha hedge). Vol-matched 50/50.
Pre-registered hypothesis: trend cuts the VRP crash drawdown at little Sharpe cost (corr<0).
Data note: VXX is 2018+ (full 2009+ VRP needs CBOE VX futures CSVs — v2 data build).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/boreas/research")
import numpy as np, pandas as pd, yfinance as yf
from tsmom import run_tsmom
from sdk.harness import StrategySpec

COST = 8.0 / 1e4


def load_data():
    d = yf.download("VXX", start="2018-01-01", progress=False, auto_adjust=False)
    close = d["Close"] if "Close" in d.columns else d.iloc[:, [0]]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.DataFrame({"VXX": close.dropna()})


def _vol_scale(r, tgt=0.10, ann=252):
    v = r.std() * np.sqrt(ann)
    return r * (tgt / v) if v > 0 else r


def signal(panel, vrp_vol=0.10, vrp_cap=2.0, blend=0.5, vol_lb=60, **trend_kw):
    vxx = pd.Series(panel["VXX"]).copy(); vxx.index = pd.to_datetime(vxx.index).normalize()
    r_vxx = vxx.pct_change()
    # VRP leg = SHORT VXX, inverse-vol sized (target vrp_vol), weekly, hard cap for the Volmageddon tail
    rv = r_vxx.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252)
    pos = (-1.0 * (vrp_vol / rv.replace(0, np.nan))).clip(-vrp_cap, vrp_cap)
    pos = pos.resample("W-FRI").last().reindex(vxx.index, method="ffill").shift(1).fillna(0.0)
    vrp = (pos * r_vxx - pos.diff().abs().fillna(0) * COST).dropna()
    vrp.index = pd.to_datetime(vrp.index).normalize()
    # Trend leg (Boreas 21-market)
    trend, ttrades = run_tsmom(**trend_kw)
    trend.index = pd.to_datetime(trend.index).normalize()
    # align + vol-match + blend
    df = pd.concat([vrp.rename("vrp"), trend.rename("trend")], axis=1).dropna()
    combo = blend * _vol_scale(df["vrp"]) + (1 - blend) * _vol_scale(df["trend"])
    combo.name = "vrp_trend"
    # trades = trend trades (in overlap) + VRP sign-runs
    lo = df.index.min()
    trades = [t for t in ttrades if pd.Timestamp(t["entry_date"]) >= lo]
    s = np.sign(pos).fillna(0.0); cur, ent = 0.0, None
    for dt, sg in s.items():
        if sg != cur:
            if cur != 0 and ent is not None and dt >= lo:
                trades.append({"ticker": "VXX", "sector": "VRP", "entry_date": str(ent.date()),
                               "exit_date": str(dt.date()), "hold_days": 5, "position_value": 1.0, "pnl": 0.0})
            cur, ent = sg, dt
    return combo, trades


SPEC = StrategySpec(
    id="vrp-trend-book",
    family="vrp_trend_combo",
    title="VRP + Trend two-premium book (agent-proposed) — short-vol hedged by CTA trend",
    markets=["volatility", "futures"],
    data_desc="VXX 2018+ (yfinance) for VRP + Boreas 21-market trend. (Full 2009+ needs CBOE VX CSVs - v2.)",
    pre_registration=("VRP leg = SHORT VXX (rolling ~30d VIX futures), inverse-vol to 10% target, weekly, "
                      "hard cap 2.0 for the Volmageddon tail, 8bps. Trend leg = frozen Boreas TSMOM. "
                      "Vol-matched 50/50, weekly. FROZEN. Hypothesis: trend (crisis-alpha) cuts the VRP "
                      "crash drawdown with corr(VRP,trend)<0; the combined book clears the rails."),
    load_data=load_data, signal=signal,
    default_params={}, grid={"default": {}, "cap_tight": {"vrp_cap": 1.5}, "cap_loose": {"vrp_cap": 3.0},
                             "blend_vrp": {"blend": 0.65}, "blend_trend": {"blend": 0.35},
                             "vrpvol_low": {"vrp_vol": 0.07}, "vollb_120": {"vol_lb": 120}},
    holdout_start="2022-01-01", deploy_max_positions=22,
)
