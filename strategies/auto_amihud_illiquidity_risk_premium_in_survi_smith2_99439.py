"""
Amihud (2002) illiquidity risk premium — survivorship-clean US small-caps.

Hypothesis: less-liquid stocks earn higher expected returns as compensation for
illiquidity (Amihud ILLIQ = avg |daily return| / daily dollar volume). We go LONG
the most-illiquid names and SHORT the most-liquid names, cross-sectionally, within
a liquidity/price-filtered small-cap book. Inverse-vol sized, weekly rebalanced,
1-day signal lag (no look-ahead), ~8bps cost on turnover.

This is a UNIVERSAL risk-premium theory -> scope='broad' -> must later GENERALISE
to untouched mid/large/sector slices.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe

START = "2001-01-01"

# Sharadar TICKERS sector labels (GICS-ish). Used to build a balanced, sectored
# small-cap universe so trades are spread across sectors for deployment-sanity.
SECTORS = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Consumer Defensive', 'Energy', 'Basic Materials',
           'Real Estate', 'Communication Services', 'Utilities']

_SECTOR_MAP = {}


def _build_universe(per_sector=120):
    """Union of the most-liquid small-caps per sector -> {ticker: sector}."""
    smap = {}
    for s in SECTORS:
        try:
            ts = us_universe(sector=s, category='Domestic Common Stock',
                             marketcap='Small', include_delisted=True, top_n=per_sector)
        except Exception:
            ts = []
        for t in ts:
            smap[t] = s
    return smap


def load_data() -> pd.DataFrame:
    """Survivorship-clean small-cap price + volume panel (MultiIndex columns)."""
    global _SECTOR_MAP
    smap = _build_universe()
    if not smap:  # fallback if sector filtering unavailable
        tickers = us_universe(category='Domestic Common Stock', marketcap='Small',
                              include_delisted=True, top_n=1300)
        smap = {t: 'Unknown' for t in tickers}
    _SECTOR_MAP = smap

    tickers = sorted(smap.keys())
    px = sep_panel(tickers, START, field='closeadj').sort_index()
    vol = sep_panel(tickers, START, field='volume')
    vol = vol.reindex(index=px.index, columns=px.columns)

    panel = pd.concat({'price': px, 'volume': vol}, axis=1)
    panel.attrs['sector_map'] = smap
    return panel


def signal(panel, lookback=21, vol_lb=63, top_q=0.2, target_vol=0.10,
           rebalance=5, min_price=2.0, cost_bps=8.0, long_short=True, **params):
    smap = panel.attrs.get('sector_map', _SECTOR_MAP)

    px = panel['price'].astype(float).sort_index()
    vol = panel['volume'].astype(float).reindex_like(px)
    rets = px.pct_change()

    # --- Amihud illiquidity: mean(|ret| / dollar-volume) over lookback ---
    dollar_vol = (px * vol)
    dollar_vol = dollar_vol.where(dollar_vol > 0)
    daily_illiq = rets.abs() / dollar_vol
    illiq = daily_illiq.rolling(lookback, min_periods=max(5, lookback // 2)).mean()

    avg_dv = dollar_vol.rolling(lookback, min_periods=5).mean()
    vlt = rets.rolling(vol_lb, min_periods=20).std()
    eligible = (px > min_price) & illiq.notna() & (avg_dv > 0) & vlt.notna()

    dates = px.index
    rebal_dates = dates[::max(1, int(rebalance))]
    weights = pd.DataFrame(np.nan, index=dates, columns=px.columns)

    for d in rebal_dates:
        il = illiq.loc[d].where(eligible.loc[d]).dropna()
        if len(il) < 20:
            continue
        n_sel = max(5, int(len(il) * top_q))
        longs = il.nlargest(n_sel).index          # most illiquid -> high premium
        shorts = il.nsmallest(n_sel).index if long_short else pd.Index([])

        iv = (1.0 / vlt.loc[d]).replace([np.inf, -np.inf], np.nan)
        w = pd.Series(0.0, index=px.columns)

        lw = iv.reindex(longs).fillna(0.0)
        if lw.sum() > 0:
            w.loc[longs] = (lw / lw.sum()).values
        if len(shorts) > 0:
            sw = iv.reindex(shorts).fillna(0.0)
            if sw.sum() > 0:
                w.loc[shorts] = -(sw / sw.sum()).values
        weights.loc[d] = w

    weights = weights.ffill().fillna(0.0)

    # --- positions lagged 1 day (no look-ahead), costs on turnover ---
    pos = weights.shift(1).fillna(0.0)
    asset_ret = rets.reindex_like(pos).fillna(0.0)
    gross = (pos * asset_ret).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    raw = gross - cost

    # --- vol targeting (trailing, lagged estimate) ---
    realized = raw.rolling(63, min_periods=20).std() * np.sqrt(252.0)
    lev = (target_vol / realized).replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    lev = lev.shift(1).fillna(1.0)
    net = (raw * lev).rename('amihud_illiquidity')

    # --- trades: one record per held position run (long & short legs) ---
    notional = 1_000_000.0
    pos_lev = pos.mul(lev, axis=0)
    idx = px.index
    trades = []
    held = (pos != 0.0)
    for tk in px.columns:
        h = held[tk].values
        if not h.any():
            continue
        w_arr = pos_lev[tk].fillna(0.0).values
        r_arr = asset_ret[tk].values
        hi = h.astype(int)
        diff = np.diff(np.concatenate([[0], hi, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            seg_w = w_arr[s:e]
            seg_r = r_arr[s:e]
            pv = float(np.mean(np.abs(seg_w)) * notional)
            pnl = float(np.sum(seg_w * seg_r) * notional)
            trades.append({
                "ticker": tk,
                "sector": smap.get(tk, "Unknown"),
                "entry_date": idx[s].strftime("%Y-%m-%d"),
                "exit_date": idx[e - 1].strftime("%Y-%m-%d"),
                "hold_days": int(e - s),
                "position_value": pv,
                "pnl": pnl,
            })

    return net, trades


SPEC = StrategySpec(
    id="amihud_illiquidity_smallcap",
    family="liquidity",
    title="Amihud Illiquidity Risk Premium (Survivorship-Clean Small-Cap, Long/Short)",
    markets=["us_equities"],
    data_desc=("Sharadar SEP survivorship-clean small-cap closeadj + volume (delisted "
               "included). Amihud ILLIQ = mean(|daily ret| / daily dollar volume) over "
               "lookback; long most-illiquid / short most-liquid quintile, inverse-vol, "
               "weekly rebalance, 8bps turnover cost, signals lagged 1 day."),
    pre_registration=("Amihud (2002) illiquidity premium: less-liquid stocks earn higher "
                      "expected returns as compensation for illiquidity. Pre-registered: "
                      "long top-20% ILLIQ, short bottom-20% ILLIQ within liquidity/price-"
                      "filtered US small-caps; expect +ve net-of-cost expectancy. Universal "
                      "premium -> must generalise to mid/large/sector slices."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb10": {"lookback": 10},
        "lb63": {"lookback": 63},
        "q10": {"top_q": 0.10},
        "long_only": {"long_short": False},
    },
    scope="broad",
    generalization_universes=["large", "mid", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)