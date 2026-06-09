"""Low-risk / lottery cross-sectional equity factor (the "low-risk lottery" anomaly).

Hypothesis (pre-registered): lottery-like stocks -- high short-term idiosyncratic
volatility AND high positive return skewness -- are systematically over-priced and
under-perform; low-risk / non-lottery names out-perform. A universal mispricing
mechanism (=> scope 'broad'), strongest in small/illiquid names where limits-to-
arbitrage bind, so tested in Sharadar SEP small-cap survivorship-clean universe.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe

START = "2004-01-01"
SECTORS = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Communication Services', 'Consumer Defensive', 'Energy',
           'Basic Materials', 'Real Estate', 'Utilities']

_SECTOR_MAP = {}


def _build_universe_and_sectors(marketcap='Small', per_sector=120):
    """Build a bounded, sector-labelled small-cap universe (every name -> a sector)."""
    global _SECTOR_MAP
    tickers = []
    for s in SECTORS:
        try:
            ts = us_universe(sector=s, category='Domestic Common Stock',
                             marketcap=marketcap, include_delisted=True, top_n=per_sector)
        except Exception:
            ts = []
        for t in ts:
            if t not in _SECTOR_MAP:
                _SECTOR_MAP[t] = s
            tickers.append(t)
    seen, uni = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uni.append(t)
    if not uni:  # fallback if per-sector queries unsupported
        try:
            uni = list(us_universe(category='Domestic Common Stock', marketcap='Small',
                                   include_delisted=True, top_n=1200))
        except Exception:
            uni = list(us_universe(top_n=1200))
    return uni


def _sector_of(t):
    return _SECTOR_MAP.get(t, 'Unknown')


def load_data() -> pd.DataFrame:
    uni = _build_universe_and_sectors(marketcap='Small', per_sector=120)
    px = sep_panel(uni, START, field='closeadj')
    px = px.sort_index().dropna(axis=1, how='all').dropna(axis=0, how='all')
    return px


def _zc(df):
    """Cross-sectional (per-date) z-score."""
    m = df.mean(axis=1)
    s = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(m, axis=0).div(s, axis=0)


def signal(panel, **params):
    p = {'lookback': 21, 'vol_lb': 63, 'quantile': 0.2,
         'target_vol': 0.10, 'max_pos': 100, 'rebalance': 5}
    p.update(params or {})
    lookback = int(p['lookback'])
    vol_lb = int(p['vol_lb'])
    q = float(p['quantile'])
    target_vol = float(p['target_vol'])
    max_pos = int(p['max_pos'])
    rebal = max(1, int(p['rebalance']))
    cost = 8e-4  # ~8 bps per unit of |turnover|
    notional = 1_000_000.0

    px = panel.sort_index().astype(float)
    cols = px.columns
    dates = px.index
    rets = px.pct_change()

    mp = max(5, lookback // 2)
    rvol = rets.rolling(lookback, min_periods=mp).std()      # short-term risk
    skew = rets.rolling(lookback, min_periods=mp).skew()     # lottery (positive skew)
    sizevol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std()  # inverse-vol sizing

    # Lottery score: high => lottery-like (short); low => low-risk / non-lottery (long)
    lott = _zc(rvol) + _zc(skew)

    rebal_idx = dates[::rebal]
    w_rows, w_dates = [], []
    for d in rebal_idx:
        sc = lott.loc[d]
        sv = sizevol.loc[d]
        ok = sc.notna() & sv.notna() & (sv > 0)
        sc, sv = sc[ok], sv[ok]
        n = len(sc)
        w = pd.Series(0.0, index=cols)
        if n >= 40:
            k = max(1, min(int(np.floor(n * q)), max_pos))
            order = sc.sort_values()                 # ascending: low lott first
            longs = order.index[:k]
            shorts = order.index[-k:]
            lw = (1.0 / sv[longs])
            lw = lw / lw.sum()
            sw = (1.0 / sv[shorts])
            sw = sw / sw.sum()
            w.loc[longs] = lw
            w.loc[shorts] = -sw                      # dollar-neutral L/S, gross ~2
        w_rows.append(w)
        w_dates.append(d)

    W = pd.DataFrame(w_rows, index=w_dates).reindex(dates).ffill().fillna(0.0)
    Wlag = W.shift(1).fillna(0.0)                    # lag signals 1 day (no look-ahead)

    # Vol-target the book using TRAILING realized vol (lagged -> no look-ahead)
    base = (Wlag * rets).sum(axis=1)
    realized = base.rolling(63, min_periods=20).std() * np.sqrt(252)
    lev = (target_vol / realized).shift(1).clip(upper=3.0).fillna(1.0)
    Wpos = Wlag.mul(lev, axis=0)

    contrib = Wpos * rets
    port = contrib.sum(axis=1)
    turnover = Wpos.diff().abs().sum(axis=1).fillna(0.0)
    net = port - turnover * cost

    first = Wpos.ne(0).any(axis=1)
    if first.any():
        net = net.loc[first.idxmax():]
    net = net.fillna(0.0)
    net.name = "auto_crypto_cross_sectional_low_risk_lottery__smith3_97363"

    # ---- Trades: one per continuous held-position run ----
    trades = []
    held_cols = Wpos.columns[(Wpos != 0).any(axis=0)]
    for t in held_cols:
        wt = Wpos[t]
        sgn = np.sign(wt)
        grp = (sgn != sgn.shift(1)).cumsum()
        for _, idx in wt.groupby(grp).groups.items():
            seg = wt.loc[idx]
            if seg.iloc[0] == 0:
                continue
            entry, exitd = idx[0], idx[-1]
            pos_val = float(seg.abs().mean() * notional)
            pnl = float(contrib[t].loc[idx].sum() * notional)
            trades.append({
                "ticker": str(t),
                "sector": _sector_of(t),
                "entry_date": entry.strftime("%Y-%m-%d"),
                "exit_date": exitd.strftime("%Y-%m-%d"),
                "hold_days": int(len(idx)),
                "position_value": pos_val,
                "pnl": pnl,
            })

    return net, trades


SPEC = StrategySpec(
    id="auto_crypto_cross_sectional_low_risk_lottery__smith3_97363",
    family="low_risk_lottery_anomaly",
    title="Low-risk / lottery cross-sectional equity factor (small-cap)",
    markets=["us_equities"],
    data_desc=("Sharadar SEP survivorship-clean daily split/div-adjusted closes; "
               "bounded small-cap Domestic Common Stock universe (~120 most-liquid "
               "names per GICS sector, delisted included); sectors from Sharadar TICKERS."),
    pre_registration=(
        "Lottery-like stocks (high short-term idiosyncratic vol + high positive "
        "return skewness) are over-priced and under-perform; low-risk / non-lottery "
        "names out-perform. Long bottom-quintile, short top-quintile of a combined "
        "z(rvol)+z(skew) lottery score; inverse-vol sized, dollar-neutral, weekly "
        "rebalance, signals lagged 1 day, ~8bps turnover cost, 10% vol target. "
        "Universal mispricing => must generalise to untouched mid/large/sector slices."),
    load_data=load_data,
    signal=signal,
    default_params={'lookback': 21, 'vol_lb': 63, 'quantile': 0.2,
                    'target_vol': 0.10, 'max_pos': 100, 'rebalance': 5},
    grid={
        'default': {},
        'lb10': {'lookback': 10},
        'lb42': {'lookback': 42},
        'q10': {'quantile': 0.10},
        'q30': {'quantile': 0.30},
    },
    scope='broad',
    generalization_universes=['large', 'mid', 'sectors'],
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)