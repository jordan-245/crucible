"""
Small/Mid-cap Value + 12-1 Momentum composite, sector-neutral, long-only.

THESIS
------
Two universal, well-documented equity premia (value = high book-to-market,
momentum = 12-1 total return skipping the most recent month) are strong in
small/mid-cap names where arbitrage capital is thin. We combine them as an
equal-weight sector-neutral composite z-score (so the bet is the *factor*, not
a sector tilt), go long the top tercile, size inverse-vol with per-name and
per-sector caps, and scale total net exposure with a 10-month moving-average
trend overlay on the broad market (100% risk-on / 50% risk-off) to cut the
left tail. Point-in-time fundamentals (bvps as of the SEC filing `datekey`,
ARQ) and a $1M ADV floor avoid look-ahead and untradeable micro-caps.

scope = 'broad' : value & momentum are universal mechanisms -> a stage-1 pass
must GENERALISE to untouched slices (large / small / sectors). To make that
honest, signal() derives its own fundamentals/sector map for WHATEVER panel
it is handed (cache hit on the home universe, recompute on a generalization
slice) so the same mechanism is genuinely re-tested out-of-universe.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (
    sep_panel, us_universe, sf1, yf_panel, fred_series,
    trend_returns, carry_returns, inv_vol_position,
)
import numpy as np
import pandas as pd

# Morningstar/Sharadar sector buckets used by the TICKERS table
_SECTORS = [
    'Basic Materials', 'Communication Services', 'Consumer Cyclical',
    'Consumer Defensive', 'Energy', 'Financial Services', 'Healthcare',
    'Industrials', 'Real Estate', 'Technology', 'Utilities',
]

_START = "2003-06-01"
_CAPITAL = 1_000_000.0          # notional for trade-ledger sizing only
_ID = 'value_mom_smallmid_sectorneutral'

_AUX = {}                       # per-panel aux cache (module global; no I/O side effects)
_SECTOR_CACHE = {}              # ticker -> sector (universe-independent, built once)


# --------------------------------------------------------------------------- #
# aux builders (self-contained so generalization universes work)
# --------------------------------------------------------------------------- #
def _sector_map():
    if _SECTOR_CACHE:
        return _SECTOR_CACHE
    for s in _SECTORS:
        try:
            for t in us_universe(sector=s, category='Domestic Common Stock',
                                 include_delisted=True):
                _SECTOR_CACHE[t] = s
        except Exception:
            pass
    return _SECTOR_CACHE


def _build_aux(price):
    cols = list(price.columns)

    # ADV ($) — share volume * price, 21d mean (fallback: no floor)
    try:
        vol = sep_panel(cols, _START, field='volume').reindex(
            index=price.index, columns=cols)
        adv = (price * vol).rolling(21, min_periods=10).mean()
    except Exception:
        adv = pd.DataFrame(np.inf, index=price.index, columns=cols)

    # point-in-time book value per share (ARQ, as-of FILING datekey)
    try:
        fund = sf1(cols, ['bvps'], dimension='ARQ').reset_index()
        lc = {c.lower(): c for c in fund.columns}
        tk, dk, bc = lc.get('ticker'), lc.get('datekey'), lc.get('bvps')
        bv = fund.pivot_table(index=dk, columns=tk, values=bc, aggfunc='last')
        bv.index = pd.to_datetime(bv.index)
        bv = bv.sort_index().reindex(columns=cols)
        idx = price.index.union(bv.index)
        bvps = bv.reindex(idx).ffill().reindex(price.index)   # PIT, no peek
    except Exception:
        bvps = pd.DataFrame(np.nan, index=price.index, columns=cols)

    # broad-market trend-overlay series (free ETF)
    try:
        spy = yf_panel(['SPY'], _START)
        market = spy['SPY'] if 'SPY' in spy.columns else spy.iloc[:, 0]
    except Exception:
        market = price.mean(axis=1)
    market = market.reindex(price.index).ffill()

    return adv, bvps, _sector_map(), market


def _get_aux(panel):
    """Return (adv, bvps, sector_map, market) for THIS panel; recompute if the
    panel's columns differ from the cached universe (generalization slice)."""
    key = tuple(panel.columns)
    if _AUX.get('cols') == key and 'adv' in _AUX:
        return _AUX['adv'], _AUX['bvps'], _AUX['sector_map'], _AUX['market']
    adv, bvps, smap, market = _build_aux(panel)
    _AUX.clear()
    _AUX.update(cols=key, adv=adv, bvps=bvps, sector_map=smap, market=market)
    return adv, bvps, smap, market


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    # survivorship-clean Small/Mid liquid universe (delisted included)
    mid = us_universe(marketcap='Mid', category='Domestic Common Stock',
                      include_delisted=True, top_n=600)
    small = us_universe(marketcap='Small', category='Domestic Common Stock',
                        include_delisted=True, top_n=900)

    smap = _sector_map()
    tickers = [t for t in dict.fromkeys(list(mid) + list(small)) if t in smap]
    tickers = tickers[:1500]

    price = sep_panel(tickers, _START, field='closeadj')
    price = price.dropna(how='all', axis=1).sort_index()

    _get_aux(price)                       # warm cache for the home universe
    return price


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _apply_caps(w, sectors, max_name, max_sector, iters=8):
    """Iterative water-fill: enforce per-name and per-sector caps; sum->1."""
    w = w.astype(float).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(iters):
        changed = False
        over = w > max_name + 1e-12
        if over.any():
            changed = True
            excess = float((w[over] - max_name).sum())
            w[over] = max_name
            free = w[(~over) & (w > 0)]
            if len(free) and free.sum() > 0:
                w[free.index] = free + excess * (free / free.sum())
            else:
                break
        sec_sum = w.groupby(sectors).sum()
        over_sec = sec_sum[sec_sum > max_sector + 1e-12]
        if len(over_sec):
            changed = True
            for s, tot in over_sec.items():
                members = sectors.index[sectors == s]
                w[members] = w[members] * (max_sector / tot)
            tot = w.sum()
            if tot > 0:
                w = w / tot
        if not changed:
            break
    return w


# --------------------------------------------------------------------------- #
# signal
# --------------------------------------------------------------------------- #
def signal(panel, **params):
    p = params
    mom_lb     = int(p.get('mom_lookback', 252))
    mom_gap    = int(p.get('mom_gap', 21))
    vol_lb     = int(p.get('vol_lb', 63))
    top_frac   = float(p.get('top_frac', 1.0 / 3.0))
    w_val      = float(p.get('w_value', 0.5))
    w_mom      = float(p.get('w_mom', 0.5))
    max_name   = float(p.get('max_name', 0.05))
    max_sector = float(p.get('max_sector', 0.25))
    min_sec_n  = int(p.get('min_sector_n', 3))
    cost_bps   = float(p.get('cost_bps', 8.0))
    adv_floor  = float(p.get('adv_floor', 1.0e6))

    price = panel
    adv, bvps, sector_map, market = _get_aux(panel)

    rets = price.pct_change().clip(-0.75, 0.75)

    # --- factor inputs (all as-of date t, no look-ahead) ---
    btm   = (bvps / price)                                    # high = value
    value = np.log(btm.where(btm > 0))                        # log book-to-market
    mom   = price.shift(mom_gap) / price.shift(mom_lb) - 1.0  # 12-1 momentum
    volat = rets.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std() \
                * np.sqrt(252.0)

    use_value = bool(value.notna().any().any())              # degrade gracefully

    # --- trend overlay: 10-month (~210 trading day) MA on broad market ---
    ma10 = market.rolling(210, min_periods=120).mean()
    exposure = pd.Series(np.where(market > ma10, 1.0, 0.5), index=market.index)
    exposure = exposure.reindex(price.index).ffill().fillna(1.0)

    # --- weekly rebalance dates (last trading day each week) ---
    ds = pd.Series(price.index, index=price.index)
    rebal_dates = pd.DatetimeIndex(ds.resample('W-FRI').last().dropna().values)
    rebal_dates = rebal_dates[rebal_dates.isin(price.index)]

    W = pd.DataFrame(0.0, index=rebal_dates, columns=price.columns)

    for d in rebal_dates:
        pr, vl, mo, vo, ad = (price.loc[d], value.loc[d], mom.loc[d],
                              volat.loc[d], adv.loc[d])
        valid = (pr.notna() & mo.notna() & vo.notna()
                 & (vo > 0) & (ad >= adv_floor))
        names = [t for t in valid[valid].index if t in sector_map]
        if len(names) < 20:
            continue

        df = pd.DataFrame({
            'value': vl.reindex(names), 'mom': mo.reindex(names),
            'vol': vo.reindex(names),
            'sector': [sector_map[t] for t in names],
        }, index=names)
        need = ['mom', 'vol'] + (['value'] if use_value else [])
        df = df.dropna(subset=need)

        # thin-sector guard: need >= min_sec_n names to z-score within sector
        cnt = df.groupby('sector')['mom'].transform('count')
        df = df[cnt >= min_sec_n]
        if len(df) < 15:
            continue

        g = df.groupby('sector')
        zm = (df['mom'] - g['mom'].transform('mean')) / g['mom'].transform('std')
        if use_value:
            zv = (df['value'] - g['value'].transform('mean')) \
                 / g['value'].transform('std')
            comp = (w_val * zv + w_mom * zm)
        else:
            comp = zm
        comp = comp.replace([np.inf, -np.inf], np.nan).dropna()
        if len(comp) < 10:
            continue

        # long-only top tercile of the composite
        thr = comp.quantile(1.0 - top_frac)
        sel = comp[comp >= thr].index
        if len(sel) < 5:
            continue

        iv = 1.0 / df.loc[sel, 'vol']                # inverse-vol size
        w = iv / iv.sum()
        w = _apply_caps(w, df.loc[sel, 'sector'], max_name, max_sector)
        W.loc[d, w.index] = w.values

    # --- apply trend exposure, turnover costs, 1-day execution lag ---
    exp_rebal = exposure.reindex(W.index).fillna(1.0)
    eff = W.mul(exp_rebal.values, axis=0)

    turn = eff.diff().abs().sum(axis=1)
    if len(eff):
        turn.iloc[0] = float(eff.iloc[0].abs().sum())
    cost = turn * (cost_bps / 1.0e4)

    eff_daily = eff.reindex(price.index).ffill().fillna(0.0)
    eff_exec  = eff_daily.shift(1).fillna(0.0)           # lag signals 1 day
    cost_exec = cost.reindex(price.index).fillna(0.0).shift(1).fillna(0.0)

    gross = (eff_exec * rets).sum(axis=1)
    daily = gross - cost_exec

    held_mass = eff_exec.abs().sum(axis=1)
    if (held_mass > 0).any():
        daily = daily.loc[held_mass[held_mass > 0].index.min():]
    daily = daily.fillna(0.0)
    daily.name = _ID

    # --- trade ledger: one trade per held-position run ---
    trades = []
    active = eff_exec.columns[(eff_exec.abs().sum(axis=0) > 0)]
    for tkr in active:
        wt = eff_exec[tkr]
        mask = wt > 1e-9
        if not mask.any():
            continue
        block = (mask != mask.shift(fill_value=False)).cumsum()
        for _, sub in mask[mask].groupby(block[mask]):
            run = sub.index
            seg_w = wt.loc[run]
            seg_r = rets[tkr].reindex(run).fillna(0.0)
            trades.append({
                'ticker': tkr,
                'sector': sector_map.get(tkr, 'Unknown'),
                'entry_date': run[0].strftime('%Y-%m-%d'),
                'exit_date': run[-1].strftime('%Y-%m-%d'),
                'hold_days': int(len(run)),
                'position_value': float(seg_w.mean() * _CAPITAL),
                'pnl': float((seg_w * seg_r * _CAPITAL).sum()),
            })

    return daily, trades


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id=_ID,
    family='equity_value_momentum',
    title='Small/Mid Value + 12-1 Momentum composite (sector-neutral, trend-overlaid)',
    markets=['us_equity_smallmid'],
    data_desc=('Sharadar SEP closeadj+volume (survivorship-clean, delisted incl) '
               'for Small/Mid liquid names; SF1 ARQ bvps as-of filing datekey '
               '(point-in-time book-to-market); SPY for the 10-month trend overlay.'),
    pre_registration=(
        'Long-only top-tercile of an equal-weight (0.5/0.5) sector-neutral '
        'z-score composite of value (log book-to-market, PIT via datekey) and '
        '12-1 momentum, restricted to Small/Mid names with >=$1M ADV. '
        'Inverse-vol sized, 5% per-name / 25% per-sector caps, thin-sector '
        'guard (>=3 names/sector), weekly rebalance, 8bps turnover cost, '
        'signals lagged 1 day. Net exposure scaled 100%/50% by a 10-month MA '
        'trend filter on the broad market. Both legs are universal premia -> '
        'must generalise out-of-universe.'),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        'default':    {},
        'value_tilt': {'w_value': 0.7, 'w_mom': 0.3},
        'mom_tilt':   {'w_value': 0.3, 'w_mom': 0.7},
        'mom_6m':     {'mom_lookback': 126},
        'cap_8pct':   {'max_name': 0.08},
    },
    scope='broad',
    generalization_universes=['large', 'small', 'sectors'],
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)