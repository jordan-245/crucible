from sdk.harness import StrategySpec
from sdk.adapters import (sep_panel, us_universe, sf1, yf_panel, fred_series,
                          trend_returns, carry_returns, inv_vol_position)
import numpy as np, pandas as pd

# ============================================================================
# Regime-gated short-vol VRP harvest (VIX vs 3M-VIX term-structure gated),
# tested STANDALONE; trend available only as a SMALL tail-overlay (grid variant).
#
# Premium: Volatility Risk Premium (VRP). Pro-cyclical — earns the roll-down in
# calm/contango, crashes in vol spikes. The term-structure switch is a RISK
# FILTER (harvest only when paid), NOT a return forecast.
#
# Gate-0 notes baked in:
#  - FRED VIXCLS (spot) + VXVCLS (3M VIX) -> ratio R = VIX/VIX3M (free, daily).
#  - SVXY changed leverage (-1x -> -0.5x) after Feb-2018 "Volmageddon"; we start
#    the sample 2018-03-15 so the entire return stream is the CONSISTENT -0.5x
#    spec (no spliced-discontinuity look-back). One major stress (Mar-2020) is
#    in-sample; 2022 sits in the write-once holdout.
# ============================================================================

SVXY_START = "2018-03-15"   # safely past the Feb-2018 -1x -> -0.5x change; consistent spec
CAPITAL = 10_000.0
ANN = 252.0


def load_data() -> pd.DataFrame:
    # Term structure (FREE/FRED): spot VIX and CBOE 3-Month VIX
    fred = fred_series({"VIXCLS": "vix", "VXVCLS": "vix3m"}, start=SVXY_START)
    # Tradable short-vol stream (FREE/yfinance ETF) — consistent -0.5x SVXY
    px = yf_panel(["SVXY"], SVXY_START)
    svxy = px["SVXY"] if "SVXY" in getattr(px, "columns", []) else px.iloc[:, 0]
    svxy_ret = svxy.pct_change()

    panel = pd.DataFrame({
        "svxy_ret": svxy_ret,
        "vix": fred["vix"],
        "vix3m": fred["vix3m"],
    })
    # R<1 => contango (positive roll-down => VRP is being paid); R>=1 => backwardation
    panel["ratio"] = panel["vix"] / panel["vix3m"]
    return panel.dropna()


def _weekly_hold(daily: pd.Series) -> pd.Series:
    """Keep value only on the first trading day of each ISO week, ffill (weekly rebalance)."""
    idx = daily.index
    wk = pd.Series(idx, index=idx).dt.to_period("W")
    first = ~wk.duplicated()
    return daily.where(first.values).ffill()


def signal(panel, **params):
    thr        = params.get("thr", 0.95)            # contango threshold on R = VIX/VIX3M
    target_vol = params.get("target_vol", 0.15)     # annualized vol target
    vol_lb     = params.get("vol_lb", 20)           # realized-vol lookback (days)
    max_lev    = params.get("max_lev", 1.0)         # cap leverage (low-vol -> no blow-up)
    cost_bps   = params.get("cost_bps", 8.0)        # ~8bps per unit turnover
    trend_w    = params.get("trend_overlay_w", 0.0) # 0.0 => STANDALONE (default/primary)

    r   = panel["svxy_ret"].astype(float)
    rat = panel["ratio"].astype(float)

    # --- regime gate: harvest VRP ONLY in contango. Same-day signal; lagged below. ---
    gate = (rat < thr).astype(float)

    # --- inverse-vol sizing, weekly rebalanced ---
    realized = r.rolling(vol_lb).std() * np.sqrt(ANN)
    invvol = (target_vol / realized).clip(upper=max_lev)
    invvol_wk = _weekly_hold(invvol)

    # weekly size * daily risk-off gate, then LAG 1 day => no look-ahead (uses prior-day R)
    raw_pos = (gate * invvol_wk).fillna(0.0)
    pos = raw_pos.shift(1).fillna(0.0)

    gross = pos * r
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    vrp_net = (gross - cost).fillna(0.0)
    vrp_net.name = "vrp_regime"

    # --- STANDALONE by default; optional SMALL trend TAIL-overlay (never reflexive 50/50) ---
    tr_trades = []
    if trend_w and trend_w > 0:
        tr, tr_trades = trend_returns()
        tr = tr.reindex(vrp_net.index).fillna(0.0)
        tr_vol = float(tr.std() * np.sqrt(ANN))
        tr_scaled = tr * (target_vol / tr_vol) if tr_vol > 0 else tr * 0.0
        rets = (1.0 - trend_w) * vrp_net + trend_w * tr_scaled   # vrp keeps the bulk of the budget
    else:
        rets = vrp_net
    rets = rets.dropna()
    rets.name = "vrp_regime"

    # --- trades: one per held WEEK of SVXY (single-instrument VRP timing) ---
    # NOTE: standalone is genuinely single-name (sector="Volatility") -> deployment-sanity
    # concentration is EXPECTED at the standalone science stage; the diversified deployable
    # book is the trend-overlay variant (adds 21 markets / 4 sectors).
    trades = []
    held = pos > 1e-6
    wk = pd.Series(pos.index, index=pos.index).dt.to_period("W")
    daily_pnl = vrp_net * CAPITAL
    frame = pd.DataFrame({"pos": pos, "held": held, "wk": wk, "pnl": daily_pnl})
    for _, grp in frame.groupby("wk"):
        g = grp[grp["held"]]
        if len(g) == 0:
            continue
        trades.append({
            "ticker": "SVXY",
            "sector": "Volatility",
            "entry_date": g.index[0].strftime("%Y-%m-%d"),
            "exit_date": g.index[-1].strftime("%Y-%m-%d"),
            "hold_days": int(len(g)),
            "position_value": float(g["pos"].mean() * CAPITAL),
            "pnl": float(g["pnl"].sum()),
        })

    # diversified trend-overlay trades (only present when the overlay is active)
    trades.extend(tr_trades)

    return rets, trades


SPEC = StrategySpec(
    id="vrp_regime_termstructure",
    family="volatility_risk_premium",
    title="Regime-gated short-vol VRP (VIX/VIX3M term-structure) — standalone, trend as small tail-overlay",
    markets=["SVXY"],
    data_desc=("FRED VIXCLS (spot VIX) + VXVCLS (CBOE 3M VIX) -> term-structure ratio R=VIX/VIX3M; "
               "yfinance SVXY (consistent -0.5x post Feb-2018) as the tradable short-vol stream. "
               "Trend overlay (grid variant only) = validated 21-market trend_returns(). OWNED/FREE only."),
    pre_registration=(
        "HYPOTHESIS: A short-volatility position in SVXY harvests the VRP roll-down, but only when the "
        "VIX term structure is in CONTANGO (R=VIX/VIX3M below threshold). Gating to cash in backwardation "
        "is a RISK FILTER (harvest only when paid), not a forecast, and should exit before the worst of a "
        "spike-crash. FROZEN PARAMS: thr=0.95, target_vol=0.15, vol_lb=20, max_lev=1.0, cost=8bps/turn, "
        "weekly inverse-vol rebalance with a DAILY prior-day-R risk-off gate, signals lagged 1 day.\n"
        "PLAN: test STANDALONE through the rails FIRST (search Sharpe, write-once 2022+ holdout, DSR/PBO, "
        "|search Sharpe|>0.3 sanity). The prior [[experiments/vrp-trend-book]] FAIL was a static short-vol "
        "leg reflexively blended 50/50 with trend — a likely false-fail by trend dilution. This differs by: "
        "(1) the CORE signal is the VIX/VIX3M term-structure regime switch (absent before); (2) it is judged "
        "STANDALONE; (3) trend, if added at all, is a SMALL tail-overlay (grid trend_overlay_w=0.20), kept "
        "ONLY if it cuts maxDD without diluting standalone Sharpe — never 50/50.\n"
        "KILL: standalone fails if holdout Sharpe<=0 or DSR not clearly >0 or PBO high or |search Sharpe|<0.3. "
        "If standalone clears, the overlay is kept only on a DD-cut-at-no-Sharpe-cost basis.\n"
        "CAVEATS: SAMPLE is short (2018-03+, ~8yr) and restricted to the consistent -0.5x SVXY spec — only one "
        "major stress (Mar-2020) is in-sample; 2022 sits in the holdout. The STANDALONE book is single-name "
        "(sector=Volatility), so deployment-sanity concentration is EXPECTED at this stage; the diversified "
        "deployable artifact is a follow-up VRP + small-trend-overlay book (deploy_max_positions ~22)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},  # primary = STANDALONE VRP (thr=0.95, no trend overlay)
    grid={
        "default": {},                          # standalone, frozen params
        "thr_0.93": {"thr": 0.93},              # stricter contango entry
        "thr_0.97": {"thr": 0.97},              # looser contango entry
        "vol_lb_40": {"vol_lb": 40},            # slower vol estimate
        "trend_overlay_0.20": {"trend_overlay_w": 0.20},  # SMALL trend tail-overlay (not 50/50)
    },
    holdout_start="2022-01-01",
    deploy_max_positions=1,  # standalone is one instrument (SVXY); overlay book would raise this
)