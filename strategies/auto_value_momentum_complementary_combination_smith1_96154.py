"""
Small/Mid-cap VALUE + MOMENTUM long-only tilt (survivorship-clean US equities).

THESIS (pre-registered, broad/universal premia):
  Value (book-to-price) and Momentum (12-1 total return, skip last month) are the two
  canonical, theory-backed cross-sectional equity premia. They are concentrated in
  small/mid-cap names (arbitraged away in the largest liquid names -> false nulls there),
  and they are low-correlated, so an equal-RISK z-score blend diversifies their drawdowns.
  We build a survivorship-clean small/mid universe (Sharadar TICKERS, delisted included),
  impose a point-in-time marketcap tradability floor, compute VALUE (book-to-price using
  ARQ bvps lagged to its filing 'datekey' over the split+div-adjusted close) and 12-1
  MOMENTUM, winsorized cross-sectional z-score each, equal-risk blend, and hold the top
  tercile long-only as an inverse-vol-weighted tilt. A hysteresis no-trade band (enter top
  tercile, only exit when ranking drops below the median) cuts turnover/costs.
  Being a UNIVERSAL premium, a stage-1 pass MUST generalise to untouched slices (large, sectors).

NOTE: sep_panel only serves the split+div-adjusted 'closeadj' field, so book-to-price uses
  the adjusted close (a standard quick-research approximation) and tradability uses a
  point-in-time marketcap floor from SF1 instead of a dollar-volume floor.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np, pandas as pd

START = "2002-01-01"

# Morningstar/Sharadar TICKERS sector vocabulary
_SECTORS = ['Healthcare', 'Technology', 'Financial Services', 'Consumer Cyclical',
            'Industrials', 'Communication Services', 'Consumer Defensive', 'Energy',
            'Basic Materials', 'Real Estate', 'Utilities']

_SECTOR_MAP = {}  # ticker -> sector (populated by load_data)


def _pit_fundamental(raw, field, price_index, tickers):
    """Pivot Sharadar fundamentals to a daily panel, forward-filled from the
    filing date ('datekey') -> strictly point-in-time, no look-ahead."""
    df = raw.copy()
    lc = {c.lower(): c for c in df.columns}
    if 'datekey' in lc:  # long format (ticker, datekey, field...)
        tcol, dcol = lc.get('ticker', 'ticker'), lc['datekey']
        fcol = lc.get(field.lower(), field)
        df = df[[tcol, dcol, fcol]].dropna()
        df[dcol] = pd.to_datetime(df[dcol])
        df = df.sort_values(dcol)
        piv = df.pivot_table(index=dcol, columns=tcol, values=fcol, aggfunc='last')
    else:  # already a (date x ticker) panel
        piv = df.copy()
        piv.index = pd.to_datetime(piv.index)
    full = price_index.union(piv.index)
    piv = piv.reindex(full).sort_index().ffill().reindex(price_index)
    return piv.reindex(columns=tickers)


def load_data() -> pd.DataFrame:
    global _SECTOR_MAP

    # ---- survivorship-clean small/mid universe + sector tags (one pass) ----
    tickers, smap = set(), {}
    for s in _SECTORS:
        for mc in ('Small', 'Mid'):
            try:
                names = us_universe(sector=s, category='Domestic Common Stock',
                                    marketcap=mc, include_delisted=True, top_n=60)
            except Exception:
                names = []
            for t in names:
                smap.setdefault(t, s)
                tickers.add(t)
    tickers = sorted(tickers)

    # ---- panels (sep_panel only serves the adjusted 'closeadj' field) ----
    px = sep_panel(tickers, START, field='closeadj')   # split+div adjusted (returns, momentum, B/P)
    tk = list(px.columns)

    raw = sf1(tk, ['bvps', 'marketcap'], dimension='ARQ')   # filing-date PIT fundamentals
    bvps = _pit_fundamental(raw, 'bvps', px.index, tk)
    mcap = _pit_fundamental(raw, 'marketcap', px.index, tk)

    _SECTOR_MAP = {t: smap.get(t, 'Unknown') for t in tk}

    panel = pd.concat({'px': px, 'bvps': bvps, 'mcap': mcap}, axis=1)
    panel.attrs['sector_map'] = _SECTOR_MAP
    return panel


def _cs_zscore(df):
    """Winsorized (1/99 pct) cross-sectional z-score, row-wise."""
    lo, hi = df.quantile(0.01, axis=1), df.quantile(0.99, axis=1)
    w = df.clip(lower=lo, upper=hi, axis=0)
    m, s = w.mean(axis=1), w.std(axis=1)
    return w.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)


def signal(panel, **params):
    p = dict(value_weight=1.0, mom_weight=1.0,
             top_q=0.33, exit_q=0.50,                 # enter top tercile, hysteresis exit at median
             mom_lb=252, mom_skip=21,                 # 12-1 total return
             mcap_floor=50.0,                         # PIT marketcap tradability floor
             vol_lb=60, cost_bps=8.0)
    p.update(params)

    px = panel['px'].sort_index()
    bvps = panel['bvps'].reindex_like(px)
    mcap = panel['mcap'].reindex_like(px)
    sector_map = panel.attrs.get('sector_map', _SECTOR_MAP)

    rets = px.pct_change()

    # ---- tradability: small/mid universe already liquidity-bounded; require live PIT size ----
    tradable = mcap.notna() & (mcap >= p['mcap_floor'])

    # ---- factors (masked to the investable universe before z-scoring) ----
    value = (bvps / px.where(px > 0)).where(tradable)                             # book-to-price
    mom = (px.shift(p['mom_skip']) / px.shift(p['mom_lb']) - 1.0).where(tradable)  # 12-1
    val_z, mom_z = _cs_zscore(value), _cs_zscore(mom)

    parts = []
    if p['value_weight'] != 0:
        parts.append(p['value_weight'] * val_z)
    if p['mom_weight'] != 0:
        parts.append(p['mom_weight'] * mom_z)
    if not parts:                                     # degenerate guard (both weights 0)
        parts = [val_z * 0.0]
    combined = parts[0]
    for x in parts[1:]:
        combined = combined + x                       # both present -> equal-risk blend
    combined = combined.where(tradable)

    # ---- inverse-vol sizing input ----
    vol_est = rets.rolling(p['vol_lb'], min_periods=20).std()

    # ---- weekly rebalance days (last trading day of each week) ----
    wk = pd.Series(px.index.to_period('W-FRI'), index=px.index)
    rb_days = px.index[wk.ne(wk.shift(-1)).values]

    target = pd.DataFrame(0.0, index=rb_days, columns=px.columns)
    prev = set()
    for d in rb_days:
        sc = combined.loc[d].dropna()
        if len(sc) < 20:
            prev = set()
            continue
        rk = sc.rank(pct=True)                                  # 1.0 = best score
        enter = set(rk[rk >= (1.0 - p['top_q'])].index)         # top tercile
        keep = set(rk[rk >= (1.0 - p['exit_q'])].index)         # hysteresis band (top half)
        new = (prev & keep) | enter                             # no-trade band: hold until below median
        new = {t for t in new if t in sc.index}
        prev = new
        if not new:
            continue
        iv = (1.0 / vol_est.loc[d, list(new)].replace(0, np.nan)).dropna()
        if iv.empty:
            prev = set()
            continue
        w = iv / iv.sum()
        target.loc[d, w.index] = w.values

    # ---- daily held weights, lagged 1 day (no look-ahead), costed ----
    hw = target.reindex(px.index).ffill().fillna(0.0)
    pos_lag = hw.shift(1).fillna(0.0)
    r = rets.reindex(columns=hw.columns).fillna(0.0)
    gross = (pos_lag * r).sum(axis=1)
    turn = (hw - hw.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (turn * (p['cost_bps'] / 1e4)).shift(1).fillna(0.0)   # charged when trade takes effect
    daily = (gross - cost).fillna(0.0)
    daily.index = pd.DatetimeIndex(daily.index)
    daily.name = 'smid_value_momentum'

    # ---- trades: one per held-position run ----
    book, rt_cost = 1_000_000.0, 2.0 * p['cost_bps'] / 1e4
    held = hw > 1e-6
    idx = hw.index
    trades = []
    for t in hw.columns:
        hcol = held[t].values
        if not hcol.any():
            continue
        wcol = hw[t].values
        rcol = r[t].values if t in r.columns else np.zeros(len(idx))
        in_pos = False
        start_i, entry_w = 0, 0.0
        n = len(idx)
        for i in range(n):
            if hcol[i] and not in_pos:
                in_pos = True
                start_i = i
                entry_w = wcol[i]
            if in_pos and (i == n - 1 or not hcol[i + 1]):
                seg_ret = rcol[start_i + 1: i + 1]              # returns realised while held (1-day lag)
                cum = float(np.prod(1.0 + seg_ret) - 1.0) if len(seg_ret) else 0.0
                pos_val = float(entry_w * book)
                pnl = pos_val * cum - pos_val * rt_cost          # net of round-trip cost
                trades.append({
                    "ticker": t,
                    "sector": sector_map.get(t, "Unknown"),
                    "entry_date": idx[start_i].strftime("%Y-%m-%d"),
                    "exit_date": idx[i].strftime("%Y-%m-%d"),
                    "hold_days": int(i - start_i + 1),
                    "position_value": pos_val,
                    "pnl": float(pnl),
                })
                in_pos = False

    return daily, trades


SPEC = StrategySpec(
    id="smid_value_momentum",
    family="equity_factor",
    title="Small/Mid-cap Value + Momentum long-only tilt",
    markets=["us_equities_smid"],
    data_desc=("Survivorship-clean Sharadar SEP small/mid-cap US equities (delisted incl, "
               "split+div adjusted closeadj); ARQ bvps + marketcap lagged to filing 'datekey'. "
               "Book-to-price VALUE (bvps/closeadj) + 12-1 MOMENTUM, point-in-time marketcap "
               "tradability floor."),
    pre_registration=("Value (book-to-price) and 12-1 momentum are canonical, theory-backed "
                      "cross-sectional equity premia concentrated in small/mid caps (arbitraged "
                      "away in the largest liquid names). Winsorized cross-sectional z-score each, "
                      "equal-risk blend, top-tercile long-only inverse-vol tilt, weekly rebalance, "
                      "hysteresis no-trade band (exit below median). Signals lagged 1 day, ~8bps "
                      "turnover cost. UNIVERSAL premia -> a stage-1 pass MUST generalise to the "
                      "untouched large-cap and sector slices."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "value_only": {"mom_weight": 0.0},
        "mom_only": {"value_weight": 0.0},
        "top_quintile": {"top_q": 0.20, "exit_q": 0.40},
        "mom_6_1": {"mom_lb": 126},
    },
    scope="broad",
    generalization_universes=["large", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=30,
)