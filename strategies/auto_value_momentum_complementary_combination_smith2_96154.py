"""
Small-cap Value + Momentum composite — long-only top-tercile factor book.

Economic thesis (pre-registration):
  Two of the most robust, theory-grounded cross-sectional equity premia are VALUE
  (high book-to-market names are compensated for distress/duration risk and persistent
  mispricing) and MOMENTUM (12-1 month relative strength persists via under-reaction).
  They are NEGATIVELY correlated, so a composite of the two is a smoother, deeper edge
  than either alone. These anomalies are STRONGEST in small/illiquid names (less
  arbitraged), so we test in the Small-cap, survivorship-clean Sharadar slice.

  Construction: point-in-time book value per share (bvps as of its FILING DATE 'datekey',
  never calendardate) divided by adjusted price -> B/M value score; 12-1 momentum (skip the
  last month) -> momentum score; each winsorised + cross-sectionally z-scored each day;
  weighted composite. LONG-ONLY TOP TERCILE, inverse-vol sized, weekly rebalance, a
  no-trade hysteresis BAND to suppress turnover, and a market TREND moving-average overlay
  (risk-off to cash when the equal-weight index is below its MA) to cut the left tail.
  Signals lagged 1 day; ~8bps cost on turnover. Holdout reserved from 2022-01-01.

  Scope = BROAD: value & momentum are universal premia, so a stage-1 pass must GENERALISE
  to untouched slices (large / mid / sectors), else it is an overfit outlier.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
START = "2004-01-01"
SECTORS = ['Healthcare', 'Technology', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Consumer Defensive', 'Energy', 'Basic Materials',
           'Real Estate', 'Utilities', 'Communication Services']

DEFAULTS = dict(
    mom_lb=252,        # 12-month momentum lookback
    mom_skip=21,       # skip most-recent month (12-1)
    vol_lb=63,         # inverse-vol estimation window
    trend_ma=200,      # market trend moving-average length
    w_value=0.5,       # composite weight on value
    w_mom=0.5,         # composite weight on momentum
    band=0.15,         # no-trade hysteresis band (quantile units)
    cost_bps=8.0,      # per-unit-turnover cost
    max_pos=50,        # cap on simultaneous holdings
    use_trend=1,       # market MA overlay on/off
    book=1_000_000.0,  # notional book (for trade dollar figures only)
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Survivorship-clean small-cap US panel + point-in-time bvps + sector map."""
    sector_map = {}
    tickers = []
    for sec in SECTORS:
        try:
            ts = us_universe(sector=sec, category='Domestic Common Stock',
                             marketcap='Small', include_delisted=True, top_n=130)
        except Exception:
            ts = []
        for t in ts:
            sector_map[t] = sec
        tickers.extend(ts)
    tickers = sorted(set(tickers))

    px = sep_panel(tickers, START, field='closeadj').sort_index()
    px = px.loc[:, px.columns.isin(tickers)]

    bv = sf1(tickers, ['bvps'], dimension='ARQ')
    px.attrs['bvps'] = bv
    px.attrs['sector_map'] = sector_map
    return px


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pit_bvps(bv, dates, tickers) -> pd.DataFrame:
    """Point-in-time bvps panel: each value known only from its FILING DATE (datekey)."""
    bv = bv.reset_index()
    cols = {c.lower(): c for c in bv.columns}
    tcol, dcol, vcol = cols.get('ticker'), cols.get('datekey'), cols.get('bvps')
    if tcol is None or dcol is None or vcol is None:
        return pd.DataFrame(index=dates, columns=tickers, dtype=float)
    bv = bv[[tcol, dcol, vcol]].dropna()
    bv[dcol] = pd.to_datetime(bv[dcol])
    piv = bv.pivot_table(index=dcol, columns=tcol, values=vcol, aggfunc='last').sort_index()
    piv = piv[~piv.index.duplicated(keep='last')]
    allidx = piv.index.union(dates)
    piv = piv.reindex(allidx).ffill().reindex(dates)
    return piv.reindex(columns=tickers)


def _zscore(df) -> pd.DataFrame:
    """Cross-sectional (per-date) winsorised z-score."""
    lo = df.quantile(0.05, axis=1)
    hi = df.quantile(0.95, axis=1)
    dfc = df.clip(lower=lo, upper=hi, axis=0)
    m = dfc.mean(axis=1)
    s = dfc.std(axis=1).replace(0, np.nan)
    return dfc.sub(m, axis=0).div(s, axis=0)


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def signal(panel, **params):
    p = dict(DEFAULTS); p.update(params)
    px = panel.sort_index()
    rets = px.pct_change()
    dates = px.index
    tickers = list(px.columns)
    sector_map = panel.attrs.get('sector_map', {})

    # --- raw factor scores ---
    bvps = _pit_bvps(panel.attrs['bvps'], dates, tickers)
    bm = bvps / px                                              # book-to-market (value)
    mom = px.shift(p['mom_skip']) / px.shift(p['mom_lb']) - 1.0  # 12-1 momentum

    zv = _zscore(bm)
    zm = _zscore(mom)

    terms = []
    if p['w_value'] != 0:
        terms.append(zv * p['w_value'])
    if p['w_mom'] != 0:
        terms.append(zm * p['w_mom'])
    composite = terms[0] if len(terms) == 1 else terms[0].add(terms[1])  # require both when blended

    # --- weekly rebalance dates (last trading day of each ISO week) ---
    dser = pd.Series(dates, index=dates)
    rebal_dates = pd.DatetimeIndex(dser.groupby(dates.to_period('W')).last().values)

    vol = rets.rolling(p['vol_lb']).std()
    target = pd.DataFrame(index=rebal_dates, columns=tickers, dtype=float)  # NaN = carry-forward

    held = set()
    for rd in rebal_dates:
        if rd not in composite.index:
            continue
        sc = composite.loc[rd].dropna()
        if len(sc) < 20:
            continue
        q_hi = sc.quantile(2.0 / 3.0)
        q_lo = sc.quantile(max(0.0, 2.0 / 3.0 - p['band']))
        new_sel = set(sc.index[sc >= q_hi])
        keep = {t for t in held if t in sc.index and sc[t] >= q_lo}   # hysteresis / no-trade band
        sel = new_sel | keep
        sel = sorted(sel, key=lambda t: sc[t], reverse=True)[:int(p['max_pos'])]
        held = set(sel)

        v = vol.loc[rd, sel].replace(0, np.nan)
        iv = (1.0 / v).fillna(0.0)
        if iv.sum() <= 0:
            w = pd.Series(1.0 / len(sel), index=sel)
        else:
            w = iv / iv.sum()

        target.loc[rd, :] = 0.0
        target.loc[rd, sel] = w.values

    # --- daily holdings: weekly target held + 1-day lag (no look-ahead) ---
    W = target.reindex(dates).ffill().fillna(0.0).shift(1).fillna(0.0)

    # --- trend MA overlay (risk-off to cash below the market MA) ---
    if p['use_trend']:
        ew = rets.mean(axis=1).fillna(0.0)
        idx = (1.0 + ew).cumprod()
        ma = idx.rolling(int(p['trend_ma'])).mean()
        risk_on = (idx > ma).astype(float)
    else:
        risk_on = pd.Series(1.0, index=dates)
    W = W.mul(risk_on.shift(1).fillna(0.0), axis=0)

    # --- net-of-cost daily returns ---
    gross = (W * rets).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1)
    net = (gross - turnover * p['cost_bps'] * 1e-4).fillna(0.0)
    net.name = "val_mom_trend_smallcap"

    active = W.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]

    # --- trades: one record per held position run ---
    trades = []
    book = float(p['book'])
    W_arr = W.values
    R_arr = rets.fillna(0.0).values
    dstr = [d.strftime('%Y-%m-%d') for d in dates]
    n = len(dates)
    for cj, t in enumerate(tickers):
        col = W_arr[:, cj]
        mask = col > 1e-9
        if not mask.any():
            continue
        i = 0
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                seg_w = col[i:j + 1]
                seg_r = R_arr[i:j + 1, cj]
                trades.append({
                    "ticker": t,
                    "sector": sector_map.get(t, "Unknown"),
                    "entry_date": dstr[i],
                    "exit_date": dstr[j],
                    "hold_days": int(j - i + 1),
                    "position_value": float(np.nanmean(seg_w) * book),
                    "pnl": float(np.nansum(seg_w * seg_r) * book),
                })
                i = j + 1
            else:
                i += 1

    return net, trades


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id="val_mom_trend_smallcap",
    family="value_momentum",
    title="Small-cap Value+Momentum composite (long-only top tercile, trend MA overlay)",
    markets=["us_equity"],
    data_desc=("Sharadar SEP closeadj (survivorship-clean, delisted incl.) for ~1400 "
               "small-cap Domestic Common Stocks across 11 sectors; Sharadar SF1 ARQ bvps "
               "lagged to its filing datekey for point-in-time book-to-market."),
    pre_registration=(
        "Universal cross-sectional premia. Value (point-in-time B/M = bvps@datekey / "
        "adj price) and Momentum (12-1, skip last month) are winsorised, daily "
        "cross-sectionally z-scored, and blended 50/50 into a composite. Long-only top "
        "tercile, inverse-vol sized, weekly rebalance with a no-trade hysteresis band, "
        "market trend MA overlay to cut the tail. Signals lagged 1 day, ~8bps turnover "
        "cost. Strongest in small/illiquid names; BROAD edge -> must generalise to "
        "large/mid/sectors. Holdout from 2022-01-01."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":    {},
        "value_only": {"w_value": 1.0, "w_mom": 0.0},
        "mom_only":   {"w_value": 0.0, "w_mom": 1.0},
        "no_trend":   {"use_trend": 0},
        "tight_band": {"band": 0.0},
    },
    scope="broad",
    generalization_universes=["large", "mid", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)