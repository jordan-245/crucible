"""
Boreas — Cross-asset VALUE (long-run 5y reversal) across the 21-market futures basket,
with the validated TREND leg available as a SIZED tail-overlay (not a reflexive 50/50).

FROZEN / PRE-REGISTERED. Primary (default_params={}) = VALUE STANDALONE.
The trend overlay is a SECONDARY, gated, sized variant (per the 2026-06-08 over-blend lesson:
never sink a real premium with a ~0-Sharpe 50/50 hedge — size to minimise drag).

Construction is RETURN-based (back-adjustment-robust): value = -1 x cumulative LOG return
over t-60m..t-12m (skip last 12m so it does NOT overlap the trend/momentum window).
Cross-sectional rank -> demean -> inverse-vol size -> ex-ante vol target -> monthly rebalance
-> 8bps cost -> lag 1 day. No price-vs-MA levels (roll back-adjustment corrupts absolute price).
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, trend_returns
import numpy as np
import pandas as pd

# ---- the existing 21-market Boreas basket, 4 sectors (yfinance continuous futures) ----
TICKERS = {
    # Equity indices
    "ES=F": "Equity", "NQ=F": "Equity", "YM=F": "Equity", "RTY=F": "Equity",
    # Government bonds
    "ZB=F": "Bond", "ZN=F": "Bond", "ZF=F": "Bond", "ZT=F": "Bond",
    # Commodities
    "CL=F": "Commodity", "NG=F": "Commodity", "GC=F": "Commodity", "SI=F": "Commodity",
    "HG=F": "Commodity", "ZC=F": "Commodity", "ZS=F": "Commodity", "ZW=F": "Commodity",
    # FX
    "6E=F": "FX", "6J=F": "FX", "6B=F": "FX", "6A=F": "FX", "6C=F": "FX",
}
START = "2000-01-01"


def load_data() -> pd.DataFrame:
    """Close panel for the 21 Boreas futures (FREE yfinance continuous contracts)."""
    panel = yf_panel(list(TICKERS.keys()), START)
    panel = panel.sort_index()
    keep = [c for c in TICKERS if c in panel.columns]
    return panel[keep]


def _value_leg(panel, long_m, skip_m, vol_lb, target_vol, min_breadth, cost_bps):
    """Standalone cross-asset value (long-run reversal) daily net returns + held weights."""
    prices = panel.sort_index().astype(float)
    cols = list(prices.columns)
    logp = np.log(prices)
    rets = prices.pct_change()

    long_d = int(round(long_m * 21))
    skip_d = int(round(skip_m * 21))

    # value = -1 x cumulative LOG return over [t-60m, t-12m] (back-adjustment robust)
    cum = logp.shift(skip_d) - logp.shift(long_d)
    value = -cum

    # trailing annualised vol for inverse-vol sizing
    vol = rets.rolling(vol_lb).std() * np.sqrt(252.0)

    # month-end rebalance dates (last trading day in each calendar month)
    idx = prices.index
    per = pd.Series(idx, index=idx).dt.to_period("M")
    is_rebal = per.values != np.append(per.values[1:], None)
    rebal_dates = idx[is_rebal]

    wmap = {}
    last_w = pd.Series(0.0, index=cols)
    for d in rebal_dates:
        v = value.loc[d]
        vv = vol.loc[d]
        live = v.notna() & vv.notna() & (vv > 0)
        if int(live.sum()) >= min_breadth:
            sv = v[live]
            r = sv.rank()
            r = r - r.mean()                       # cross-sectional demean (dollar-neutral L/S)
            sd = r.std(ddof=0)
            if sd > 0:
                s = r / sd                          # standardise for stable vol-targeting
                denom = np.sqrt((s ** 2).sum())     # ex-ante port vol ~ target * (s/vol)*vol
                if denom > 0:
                    k = target_vol / denom
                    w = k * (s / vv[live])
                    last_w = pd.Series(0.0, index=cols)
                    last_w[live] = w
        wmap[d] = last_w.copy()

    W = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for d, w in wmap.items():
        W.loc[d] = w
    W = W.ffill().fillna(0.0)
    W = W.shift(1).fillna(0.0)                      # lag 1 day -> no look-ahead

    gross = (W * rets).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost)

    # trim leading all-zero (pre-signal) rows
    active = W.abs().sum(axis=1)
    nz = active[active > 0].index
    if len(nz):
        net = net.loc[nz[0]:]
        W = W.loc[nz[0]:]
    net = net.astype(float)
    return net, W, rets


def _extract_trades(W, rets, capital=100000.0):
    """One trade per contiguous same-sign held run, per market (factor-book convention)."""
    trades = []
    for col in W.columns:
        w = W[col]
        sgn = np.sign(w).fillna(0.0)
        grp = (sgn != sgn.shift(1)).cumsum()
        for _, sub in w.groupby(grp):
            s = float(sgn.loc[sub.index[0]])
            if s == 0.0:
                continue
            entry = sub.index[0]
            exit_ = sub.index[-1]
            r = rets[col].reindex(sub.index)
            pnl = float((w.loc[sub.index] * r).sum(skipna=True) * capital)
            trades.append({
                "ticker": col,
                "sector": TICKERS.get(col, "Other"),
                "entry_date": entry.strftime("%Y-%m-%d"),
                "exit_date": exit_.strftime("%Y-%m-%d"),
                "hold_days": int(len(sub)),
                "position_value": float(sub.abs().mean() * capital),
                "pnl": pnl,
            })
    return trades


def signal(panel, **params):
    long_m = float(params.get("long_m", 60))        # 60-month lookback start
    skip_m = float(params.get("skip_m", 12))        # skip last 12m (no trend overlap)
    vol_lb = int(params.get("vol_lb", 60))          # inverse-vol lookback (days)
    target_vol = float(params.get("target_vol", 0.10))
    min_breadth = int(params.get("min_breadth", 10))
    cost_bps = float(params.get("cost_bps", 8.0))
    trend_weight = float(params.get("trend_weight", 0.0))   # 0 = STANDALONE primary

    net, W, rets = _value_leg(panel, long_m, skip_m, vol_lb,
                              target_vol, min_breadth, cost_bps)

    trades = _extract_trades(W, rets)

    out = net
    if trend_weight > 0.0:
        try:
            tr = trend_returns()
            tr = tr[0] if isinstance(tr, (tuple, list)) else tr
            tr = pd.Series(tr).dropna()
            df = pd.concat([net.rename("v"), tr.rename("t")], axis=1).dropna()
            if len(df) > 60 and df["t"].std() > 0 and df["v"].std() > 0:
                tsc = df["v"].std() / df["t"].std()         # vol-match trend to value leg
                blend = (1.0 - trend_weight) * df["v"] + trend_weight * (df["t"] * tsc)
                if blend.std() > 0:                         # renorm to value vol (Sharpe-invariant)
                    blend = blend * (df["v"].std() / blend.std())
                out = blend
            # append trend trades (guarded) for combination deployment-sanity
            try:
                ttr = tr  # placeholder; trend trades requested separately if needed
                tt = trend_returns()
                if isinstance(tt, (tuple, list)) and len(tt) > 1 and isinstance(tt[1], list):
                    for t in tt[1]:
                        if isinstance(t, dict) and "ticker" in t:
                            trades.append(t)
            except Exception:
                pass
        except Exception:
            out = net

    out = out.astype(float)
    out.name = "xasset_value" + (f"_trend{int(round(trend_weight*100))}" if trend_weight > 0 else "")
    return out, trades


SPEC = StrategySpec(
    id="boreas_xasset_value_trend",
    family="cross_asset_value",
    title="Cross-asset VALUE (5y long-run reversal) x 21-market futures + sized TREND hedge "
          "(canonical AQR value+momentum book, value tested STANDALONE first)",
    markets=list(TICKERS.keys()),
    data_desc="21 yfinance continuous futures (4 sectors: equity indices, govt bonds, "
              "commodities, FX), daily Close 2000+, return-based value signal "
              "(t-60m..t-12m), + the validated trend_returns() CTA leg for overlay.",
    pre_registration=(
        "FROZEN. PRIMARY = VALUE STANDALONE (trend_weight=0). For each of 21 futures, "
        "value = -1 x cumulative LOG return over t-60m..t-12m (skip last 12m -> no overlap "
        "with the 12m trend window; long-run cross-asset reversal = AQR value proxy). "
        "Monthly: cross-sectionally rank -> demean (dollar-neutral L/S) -> inverse-vol size "
        "(60d trailing annualised vol) -> ex-ante vol target 10% -> hold to next month-end -> "
        "8bps cost on turnover -> lag 1 day. Require >=10 live markets/rebalance. "
        "Built from RETURNS only (back-adjustment robust); NO price-vs-MA levels. "
        "Decision rule: pass value's standalone search/holdout/DSR/FDR/deployment-sanity bar "
        "FIRST. ONLY THEN consider the TREND overlay (grid trend_weight in {0.25,0.40}); the "
        "headline test is corr, drawdown-cut and MAR of value+trend at NO Sharpe dilution. "
        "Per the 2026-06-08 over-blend lesson: do NOT force a 50/50; size trend to minimise drag. "
        "Holdout 2022-01-01 write-once."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                              # value standalone (PRIMARY)
        "lb_long_48": {"long_m": 48},               # shorter reversal window
        "lb_long_72": {"long_m": 72},               # longer reversal window
        "trend_overlay_25": {"trend_weight": 0.25}, # secondary sized trend overlay
        "trend_overlay_40": {"trend_weight": 0.40},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=21,
)