Looking at this: the `=== CODE ===` section is empty and the error is simply that no `SPEC` was ever defined. Given the module name (idiosyncratic-skewness / lottery), and that there's no cross-sectional crypto panel in the inventory, I'll implement the well-documented **lottery / idiosyncratic-skewness anomaly** in the survivorship-clean US **small-cap** universe (where it lives), standalone (no reflexive trend pairing), with the full contract satisfied.

```python
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np
import pandas as pd

START = "2004-01-01"
NOTIONAL = 1_000_000.0

# Sharadar sector strings (for trade tagging / deployment-sanity sector spread)
_SECTORS = ["Healthcare", "Basic Materials", "Financial Services", "Technology",
            "Consumer Cyclical", "Consumer Defensive", "Industrials", "Real Estate",
            "Energy", "Utilities", "Communication Services"]

_SECTOR_CACHE = {}


def _sector_map(tickers):
    key = frozenset(tickers)
    if key in _SECTOR_CACHE:
        return _SECTOR_CACHE[key]
    tset = set(tickers)
    smap = {}
    for s in _SECTORS:
        try:
            members = us_universe(sector=s, category='Domestic Common Stock',
                                  include_delisted=True)
        except Exception:
            members = []
        for t in members:
            if t in tset:
                smap[t] = s
    _SECTOR_CACHE[key] = smap
    return smap


def load_data() -> pd.DataFrame:
    # Lottery / idiosyncratic-skewness anomaly lives in SMALL, illiquid names.
    # Bound the universe (~1000 most-liquid small caps) for CPCV tractability.
    tickers = us_universe(category='Domestic Common Stock', marketcap='Small',
                          include_delisted=True, top_n=1000)
    panel = sep_panel(tickers, start=START, field='closeadj')
    panel = panel.sort_index().dropna(how='all', axis=1).dropna(how='all', axis=0)
    return panel


def signal(panel, **params):
    p = dict(lookback=126, n_side=30, target_vol=0.10, vol_lb=63,
             cost_bps=8.0, min_price=3.0, vol_floor=0.005, long_only=False)
    p.update(params)
    lookback = int(p['lookback'])
    n_side = int(p['n_side'])

    px = panel.sort_index().astype(float)
    rets = px.pct_change().clip(-0.5, 0.5)

    # idiosyncratic return ~ stock minus equal-weight market (beta=1 proxy)
    mkt = rets.mean(axis=1)
    resid = rets.sub(mkt, axis=0)

    mp = max(int(lookback * 0.6), 20)
    skew = resid.rolling(lookback, min_periods=mp).skew()

    vol = rets.rolling(int(p['vol_lb']), min_periods=20).std().clip(lower=float(p['vol_floor']))

    # lag everything by 1 trading day -> NO look-ahead
    skew_l = skew.shift(1)
    vol_l = vol.shift(1)
    price_l = px.shift(1)

    valid = skew_l.notna() & vol_l.notna() & (price_l > float(p['min_price']))
    skew_v = skew_l.where(valid)

    # cross-sectional selection: LONG lowest idio-skew, SHORT highest (lottery)
    asc = skew_v.rank(axis=1, method='first')
    desc = skew_v.rank(axis=1, method='first', ascending=False)
    long_mask = asc <= n_side
    short_mask = desc <= n_side

    iv = (1.0 / vol_l).where(valid)          # inverse-vol sizing
    long_w = iv.where(long_mask)
    long_w = long_w.div(long_w.sum(axis=1), axis=0).fillna(0.0)

    if bool(p['long_only']):
        w = long_w
    else:
        short_w = iv.where(short_mask)
        short_w = short_w.div(short_w.sum(axis=1), axis=0).fillna(0.0)
        w = long_w - short_w                  # dollar-neutral

    # weekly rebalance: hold target weights set on last trading day of each week
    idx = px.index
    wk_last = idx.to_series().groupby(idx.to_period('W')).max().values
    is_rebal = pd.Series(False, index=idx)
    is_rebal.loc[wk_last] = True
    mask_rows = is_rebal.values

    w_rebal = w.copy()
    w_rebal.loc[~mask_rows, :] = np.nan
    w_rebal = w_rebal.ffill().fillna(0.0)

    # ex-ante vol targeting (lagged) on the unlevered book
    unlev = (w_rebal * rets).sum(axis=1)
    roll_vol = unlev.rolling(63, min_periods=20).std() * np.sqrt(252.0)
    lev = (float(p['target_vol']) / roll_vol).replace([np.inf, -np.inf], np.nan)
    lev = lev.clip(0.0, 4.0).shift(1).ffill().fillna(1.0)

    pos = w_rebal.mul(lev, axis=0)

    gross = (pos * rets).sum(axis=1)
    turnover = (pos - pos.shift(1)).abs().sum(axis=1)
    cost = turnover * (float(p['cost_bps']) / 1e4)
    net = (gross - cost)

    # trim warm-up
    active = pos.abs().sum(axis=1) > 0
    if active.any():
        first = active.idxmax()
        net = net.loc[first:]
        pos = pos.loc[first:]
    net = net.fillna(0.0).astype(float)
    net.name = 'idio_skew_lottery'

    # ----- trades: one per continuous held run per name -----
    smap = _sector_map(list(px.columns))
    trades = []
    dates = pos.index
    held = pos.abs() > 1e-12
    cols = pos.columns[held.any(axis=0)]
    rr = rets.reindex(index=dates)
    for col in cols:
        h = held[col].values
        if not h.any():
            continue
        pv = pos[col].values
        rv = np.nan_to_num(rr[col].values)
        active_i = np.where(h)[0]
        splits = np.where(np.diff(active_i) > 1)[0]
        for g in np.split(active_i, splits + 1):
            if g.size == 0:
                continue
            seg_p = pv[g]
            seg_r = rv[g]
            trades.append({
                "ticker": str(col),
                "sector": smap.get(col, "Unknown"),
                "entry_date": str(pd.Timestamp(dates[g[0]]).date()),
                "exit_date": str(pd.Timestamp(dates[g[-1]]).date()),
                "hold_days": int(g.size),
                "position_value": float(np.mean(np.abs(seg_p)) * NOTIONAL),
                "pnl": float(np.sum(seg_p * seg_r) * NOTIONAL),
            })

    return net, trades


SPEC = StrategySpec(
    id="idio_skew_lottery_smallcap",
    family="skewness_lottery",
    title="Idiosyncratic-skewness (lottery) anomaly — long low-skew / short high-skew US small caps",
    markets=["US equities (Sharadar SEP, small-cap, survivorship-clean)"],
    data_desc=("Daily split/div-adjusted closes (sep_panel) for the ~1000 most-liquid US small-cap "
               "common stocks incl. delisted. Idiosyncratic return = stock minus equal-weight market; "
               "rolling skewness over `lookback` days is the lottery signal. NB: the project name "
               "references crypto, but no cross-sectional crypto panel exists in the data inventory; "
               "we test the SAME universal lottery/idio-skew mechanism in survivorship-clean US small "
               "caps where it is documented (Bali-Cakici-Whitelaw MAX; Boyer-Mitton-Vorkink idio skew)."),
    pre_registration=(
        "HYPOTHESIS: lottery-like stocks with high idiosyncratic return skewness are overpriced and "
        "earn LOWER subsequent returns. PREDICTION: a weekly-rebalanced, inverse-vol, dollar-neutral "
        "book LONG the lowest-idio-skew and SHORT the highest-idio-skew small caps earns a positive "
        "risk-adjusted return NET of 8bps/turnover. DESIGN frozen in advance: small-cap universe "
        "(anomaly arbitraged away in large liquid names), 6-month idio-skew signal, n_side=30 names/leg, "
        "1-day signal lag, ex-ante 10% vol target. Tested STANDALONE (no reflexive trend pairing). "
        "Broad equity factor -> a stage-1 pass MUST generalize to untouched large/mid slices or it is "
        "an overfit outlier. HOLDOUT 2022-01-01+ untouched."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 126, "n_side": 30, "target_vol": 0.10, "vol_lb": 63,
                    "cost_bps": 8.0, "min_price": 3.0, "long_only": False},
    grid={
        "default": {},
        "skew_1m": {"lookback":