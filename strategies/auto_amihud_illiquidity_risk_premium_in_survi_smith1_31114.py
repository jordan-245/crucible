# Amihud (2002) illiquidity risk premium — survivorship-clean US small-caps.
# Hypothesis (pre-registered): expected return rises with average daily
# |return| / dollar-volume (ILLIQ). We go long the MOST-illiquid but still
# tradeable small-caps, inverse-vol weighted, weekly rebalanced, 8bps costs.
# Illiquidity is a UNIVERSAL premium -> scope='broad' -> must generalise to
# untouched mid-cap and sector slices.

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np, pandas as pd

START = "2000-01-01"
NOTIONAL = 1_000_000.0
COST_BPS = 0.0008  # 8bps on turnover

# Sharadar sectors — used to tag trades for the deployment-sanity check.
_SECTORS = ['Basic Materials', 'Consumer Cyclical', 'Financial Services',
            'Real Estate', 'Consumer Defensive', 'Healthcare', 'Utilities',
            'Communication Services', 'Energy', 'Industrials', 'Technology']
_SECTOR_MAP = {}  # ticker -> sector, populated by load_data()


def load_data() -> pd.DataFrame:
    """Survivorship-clean small-cap panel: adjusted close + dollar volume.

    Cross-sectional anomalies live in small/illiquid names, so we bound the
    universe to the most-liquid SMALL caps per sector (delisted included)."""
    global _SECTOR_MAP
    smap = {}
    tickers = []
    for sec in _SECTORS:
        try:
            ts = us_universe(sector=sec, category='Domestic Common Stock',
                             marketcap='Small', include_delisted=True, top_n=120)
        except Exception:
            ts = []
        for t in ts:
            if t not in smap:
                smap[t] = sec
                tickers.append(t)
    _SECTOR_MAP = smap

    price = sep_panel(tickers, START, field='closeadj')   # split+div adjusted
    vol = sep_panel(tickers, START, field='volume')
    common = price.columns.intersection(vol.columns)
    price = price[common].sort_index()
    vol = vol.reindex(index=price.index, columns=common)
    dvol = price * vol  # consistent (split-adjusted) dollar volume

    panel = pd.concat({'price': price, 'dvol': dvol}, axis=1)
    panel.attrs['sectors'] = {k: v for k, v in smap.items() if k in common}
    return panel


def signal(panel, lookback=126, vol_lb=63, select_n=40, target_vol=0.10,
           max_pos=0.08, min_dvol=2.5e5, **params):
    smap = panel.attrs.get('sectors', _SECTOR_MAP)
    price = panel['price']
    dvol = panel['dvol']
    rets = price.pct_change()

    # Amihud ILLIQ = trailing mean of |ret| / dollar-volume.
    illiq_d = rets.abs() / dvol.replace(0.0, np.nan)
    illiq = illiq_d.rolling(lookback, min_periods=int(lookback * 0.6)).mean()

    # Tradeability floor: require sufficient average daily dollar volume.
    avg_dvol = dvol.rolling(lookback, min_periods=int(lookback * 0.6)).mean()
    illiq = illiq.where(avg_dvol >= min_dvol)

    # Inverse-vol sizing input.
    vol = rets.rolling(vol_lb, min_periods=int(vol_lb * 0.6)).std()

    idx = price.index
    rebal = idx[::5]  # weekly (every 5 trading days)

    weights = pd.DataFrame(np.nan, index=idx, columns=price.columns)
    for d in rebal:
        s = illiq.loc[d].dropna()
        if len(s) < select_n:
            continue
        sel = s.sort_values(ascending=False).index[:select_n]  # most illiquid
        v = vol.loc[d, sel].replace(0.0, np.nan).dropna()
        if len(v) == 0:
            continue
        w = 1.0 / v
        w = w / w.sum()
        w = w.clip(upper=max_pos)
        w = w / w.sum()
        row = pd.Series(0.0, index=price.columns)
        row[w.index] = w.values
        weights.loc[d] = row

    weights = weights.ffill().fillna(0.0)
    # Drop phantom positions in delisted/missing-price names (realistic exit).
    weights = weights.where(price.notna(), 0.0)

    # Lag 1 day -> position established the day after the signal (no look-ahead).
    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)

    # Turnover cost.
    turnover = w_lag.diff().abs().sum(axis=1)
    cost = turnover * COST_BPS

    # Gentle portfolio-level vol targeting (lagged -> no look-ahead).
    ann = np.sqrt(252.0)
    realized = gross.rolling(63, min_periods=30).std() * ann
    lev = (target_vol / realized).replace([np.inf, -np.inf], np.nan)
    lev = lev.clip(upper=3.0).shift(1).fillna(1.0)

    net = lev * gross - lev * cost
    net.name = 'amihud_illiquidity'

    # Trim leading flat period.
    active = w_lag.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]

    # ---- Trades: one per held position run ----
    eff = w_lag.mul(lev, axis=0)                # leverage-adjusted weights
    held = (w_lag > 1e-8)
    trades = []
    cols = held.columns[held.any().values]
    arr_idx = idx
    for t in cols:
        h = held[t].values.astype(int)
        changes = np.diff(np.concatenate(([0], h, [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1
        rt = rets[t].values
        wt = eff[t].values
        for a, b in zip(starts, ends):
            entry, exit_ = arr_idx[a], arr_idx[b]
            run_r = np.nan_to_num(rt[a:b + 1])
            cumret = float(np.prod(1.0 + run_r) - 1.0)
            avg_w = float(np.nanmean(wt[a:b + 1]))
            pv = avg_w * NOTIONAL
            hd = int((exit_ - entry).days) or 1
            trades.append({
                "ticker": t,
                "sector": smap.get(t, "Unknown"),
                "entry_date": pd.Timestamp(entry).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(exit_).strftime("%Y-%m-%d"),
                "hold_days": hd,
                "position_value": float(pv),
                "pnl": float(pv * cumret),
            })

    return net.dropna(), trades


SPEC = StrategySpec(
    id="amihud_illiquidity_smallcap",
    family="liquidity_premium",
    title="Amihud Illiquidity Premium (Survivorship-Clean US Small-Caps)",
    markets=["us_equity"],
    data_desc=("Sharadar SEP closeadj + volume (delisted incl, split/div adj). "
               "ILLIQ = trailing-126d mean of |daily ret| / daily dollar-volume; "
               "long the most-illiquid tradeable (>=$250k ADV) small-caps, "
               "inverse-vol weighted, weekly rebalance, 8bps turnover cost."),
    pre_registration=(
        "Amihud (2002): expected return increases with average |return|/dollar-volume "
        "(ILLIQ). We pre-declare a long-only book of the top-N most-illiquid but still "
        "tradeable survivorship-clean US small-caps, inverse-vol sized, weekly "
        "rebalanced, costed at 8bps on turnover, with a lagged vol target. As a "
        "universal liquidity-premium mechanism it must generalise out-of-sample to "
        "untouched mid-cap and sector slices. Holdout begins 2022-01-01."),
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 126, "vol_lb": 63, "select_n": 40,
                    "target_vol": 0.10, "max_pos": 0.08, "min_dvol": 2.5e5},
    grid={
        "default": {},
        "lb63": {"lookback": 63},
        "lb252": {"lookback": 252},
        "top25": {"select_n": 25},
        "top60": {"select_n": 60},
    },
    scope="broad",
    generalization_universes=["mid", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)