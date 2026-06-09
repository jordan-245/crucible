import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1

# ----------------------------------------------------------------------------
# EARNINGS ANNOUNCEMENT PREMIUM
# Thesis (Frazzini-Lamont style): stocks earn an abnormal positive return in the
# (predictable) window around their scheduled quarterly earnings announcement.
# We PREDICT the next announcement point-in-time from the last KNOWN SF1 filing
# 'datekey' + ~91 days (strictly lagged, no calendardate, no look-ahead), go LONG
# the predicted announcers and SHORT matched NON-announcers in the SAME sector,
# inverse-vol weighted, then strip out market/beta exposure (ex-ante 60d beta) and
# sector exposure (us_universe tags). Survivorship-clean mid-cap universe with a
# trailing-60d SEP dollar-ADV liquidity floor. Monthly rebalance, 1-day signal lag.
# ----------------------------------------------------------------------------

START = "2004-01-01"
TOP_N = 1500
SECTORS = ['Healthcare', 'Technology', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Communication Services', 'Consumer Defensive', 'Energy',
           'Basic Materials', 'Real Estate', 'Utilities']
NOTIONAL = 1_000_000.0


def _extract_datekeys(sf):
    """dict ticker -> sorted np.datetime64[D] array of SF1 filing dates (datekey)."""
    out = {}
    if sf is None or len(sf) == 0:
        return out
    df = sf
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name in ('datekey', 'ticker', 'date'):
        df = df.reset_index()
    cols = {str(c).lower(): c for c in df.columns}
    tcol = cols.get('ticker')
    dcol = cols.get('datekey')
    if dcol is None:  # datekey may have been the (named) index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.assign(_dk=df.index)
            dcol = '_dk'
    if tcol is None or dcol is None:
        return out
    sub = df[[tcol, dcol]].dropna()
    sub[dcol] = pd.to_datetime(sub[dcol], errors='coerce')
    sub = sub.dropna()
    for tk, g in sub.groupby(tcol):
        d = np.sort(g[dcol].values.astype('datetime64[D]'))
        if len(d):
            out[str(tk)] = np.unique(d)
    return out


def _build_announce(reb_dates, datekeys, columns, horizon, win_lo, win_hi):
    """[reb_date x ticker] indicator: predicted to announce in the window ahead.
       predicted = last KNOWN datekey (<= reb_date) + horizon days (point-in-time)."""
    rv = reb_dates.values.astype('datetime64[D]')
    A = pd.DataFrame(0.0, index=reb_dates, columns=columns)
    for tk in columns:
        d = datekeys.get(tk)
        if d is None or len(d) == 0:
            continue
        pos = np.searchsorted(d, rv, side='right') - 1
        valid = pos >= 0
        last = d[np.clip(pos, 0, len(d) - 1)]
        pred = last + np.timedelta64(int(horizon), 'D')
        du = (pred - rv) / np.timedelta64(1, 'D')
        flag = valid & (du >= win_lo) & (du <= win_hi)
        A[tk] = flag.astype(float)
    return A


def _residualize(w, beta):
    """Remove market (dollar) + beta exposure: project off span{1, beta}."""
    b = beta.values.astype(float)
    X = np.column_stack([np.ones(len(b)), b])
    try:
        coef, *_ = np.linalg.lstsq(X, w.values.astype(float), rcond=None)
        res = w.values.astype(float) - X @ coef
    except Exception:
        res = w.values.astype(float)
    return pd.Series(res, index=w.index)


def _emit_trades(held, rets, sector_map, notional=NOTIONAL):
    """One trade per held-position run (factor-book convention)."""
    trades = []
    idx = held.index
    r_all = rets.reindex(idx).fillna(0.0)
    for tk in held.columns:
        w = held[tk].values
        active = w != 0
        if not active.any():
            continue
        r = r_all[tk].values
        starts = np.where(active & ~np.r_[False, active[:-1]])[0]
        ends = np.where(active & ~np.r_[active[1:], False])[0]
        sec = sector_map.get(tk, 'Unknown')
        for s, e in zip(starts, ends):
            wseg = w[s:e + 1]
            rseg = r[s:e + 1]
            trades.append({
                "ticker": str(tk),
                "sector": str(sec),
                "entry_date": pd.Timestamp(idx[s]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(idx[e]).strftime("%Y-%m-%d"),
                "hold_days": int(e - s + 1),
                "position_value": float(np.mean(np.abs(wseg)) * notional),
                "pnl": float(np.sum(wseg * rseg) * notional),
            })
    return trades


# ----------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    # Survivorship-clean mid-cap universe (delisted included), liquidity-bounded.
    tickers = us_universe(category='Domestic Common Stock', marketcap='Mid',
                          include_delisted=True, top_n=TOP_N)
    tickers = sorted(set(tickers))

    px = sep_panel(tickers, START, field='closeadj')        # split+div adj -> returns/vol
    px = px.dropna(axis=1, how='all').sort_index()
    cols = list(px.columns)

    # Trailing-60d dollar ADV liquidity series (close * volume).
    try:
        vol = sep_panel(cols, START, field='volume').reindex(index=px.index, columns=cols)
        clo = sep_panel(cols, START, field='close').reindex(index=px.index, columns=cols)
        adv60 = (clo * vol).rolling(60, min_periods=20).mean()
    except Exception:
        adv60 = pd.DataFrame(np.inf, index=px.index, columns=cols)  # floor disabled if no vol

    # Sector tags from us_universe (for sector-neutralization & matched shorts).
    sector_map = {}
    for s in SECTORS:
        try:
            names_s = us_universe(sector=s, category='Domestic Common Stock',
                                  marketcap='Mid', include_delisted=True, top_n=4000)
        except Exception:
            names_s = []
        for nm in names_s:
            sector_map[nm] = s

    # Point-in-time announcement schedule inputs.
    datekeys = {}
    try:
        sf = sf1(cols, fields=['eps'], dimension='ARQ')
        datekeys = _extract_datekeys(sf)
    except Exception:
        datekeys = {}
    datekeys = {k: v for k, v in datekeys.items() if k in set(cols)}

    px.attrs['adv60'] = adv60
    px.attrs['sector'] = sector_map
    px.attrs['datekeys'] = datekeys
    return px


def signal(panel, **params):
    horizon = int(params.get('horizon', 91))
    win_lo = float(params.get('win_lo', -15))
    win_hi = float(params.get('win_hi', 30))
    vol_lb = int(params.get('vol_lb', 60))
    beta_lb = int(params.get('beta_lb', 60))
    adv_floor = float(params.get('adv_floor', 2_000_000.0))
    target_vol = float(params.get('target_vol', 0.10))
    cost_bps = float(params.get('cost_bps', 8.0))
    min_names = int(params.get('min_names', 20))

    name = 'earnings_announcement_premium'
    empty = (pd.Series(dtype=float, name=name), [])
    px = panel
    if px is None or px.shape[1] < min_names:
        return empty

    adv60 = panel.attrs.get('adv60')
    sector = panel.attrs.get('sector', {})
    datekeys = panel.attrs.get('datekeys', {})
    cols = list(px.columns)

    rets = px.pct_change()
    mkt = rets.mean(axis=1)

    # ex-ante rolling vol (inverse-vol sizing) and ex-ante 60d beta (neutralization)
    vol = rets.rolling(vol_lb, min_periods=20).std()
    mp = 20
    mean_i = rets.rolling(beta_lb, min_periods=mp).mean()
    mean_m = mkt.rolling(beta_lb, min_periods=mp).mean()
    mean_p = rets.mul(mkt, axis=0).rolling(beta_lb, min_periods=mp).mean()
    var_m = mkt.rolling(beta_lb, min_periods=mp).var()
    cov = mean_p.sub(mean_i.mul(mean_m, axis=0))
    beta = cov.div(var_m.replace(0.0, np.nan), axis=0)

    # monthly rebalance dates = first trading day of each month
    reb = px.index.to_series().groupby([px.index.year, px.index.month]).first()
    reb_dates = pd.DatetimeIndex(reb.values)

    announce = _build_announce(reb_dates, datekeys, cols, horizon, win_lo, win_hi)
    adv_reb = adv60.reindex(reb_dates) if adv60 is not None else None
    vol_reb = vol.reindex(reb_dates)
    beta_reb = beta.reindex(reb_dates)
    px_reb = px.reindex(reb_dates)

    target_daily = target_vol / np.sqrt(252.0)
    sector_ok = pd.Series([sector.get(c) in SECTORS for c in cols], index=cols)

    W = pd.DataFrame(0.0, index=reb_dates, columns=cols)
    for t in reb_dates:
        a_t = announce.loc[t]
        v_t = vol_reb.loc[t]
        b_t = beta_reb.loc[t]
        p_t = px_reb.loc[t]
        elig = (adv_reb.loc[t] >= adv_floor) if adv_reb is not None else pd.Series(True, index=cols)

        valid = (p_t.notna() & v_t.notna() & (v_t > 0) & b_t.notna()
                 & elig.fillna(False) & sector_ok)
        names = [c for c in cols if bool(valid.get(c, False))]
        if len(names) < min_names:
            continue

        iv = 1.0 / v_t[names]
        w = pd.Series(0.0, index=names)
        active_sectors = 0
        for s in SECTORS:
            g = [c for c in names if sector.get(c) == s]
            if len(g) < 2:
                continue
            ag = a_t[g]
            longs = [c for c in g if ag[c] > 0]
            shorts = [c for c in g if ag[c] == 0]
            if not longs or not shorts:
                continue  # need both a predicted announcer and a matched non-announcer
            wl = iv[longs] / iv[longs].sum()
            ws = iv[shorts] / iv[shorts].sum()
            w[longs] = wl.values
            w[shorts] = -ws.values
            active_sectors += 1
        if active_sectors == 0 or w.abs().sum() == 0:
            continue

        # beta + market (dollar) neutralization, gross normalize, vol target
        w = _residualize(w, b_t[names])
        gl = w.abs().sum()
        if gl <= 0:
            continue
        w = w / gl
        pred_daily = np.sqrt(((w * v_t[names]) ** 2).sum())
        if pred_daily > 0:
            w = w * (target_daily / pred_daily)
        W.loc[t, names] = w.values

    if (W.abs().sum().sum() == 0):
        return empty

    # expand to daily, lag 1 day (no look-ahead), apply costs on turnover
    daily_w = W.reindex(px.index, method='ffill').fillna(0.0)
    held = daily_w.shift(1).fillna(0.0)
    port = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    cost = (cost_bps / 1e4) * turnover
    net = (port - cost).fillna(0.0)

    nz = net.ne(0).cumsum() > 0
    net = net.loc[nz]
    if net.empty:
        return empty
    net.name = name

    held_trim = held.loc[net.index]
    trades = _emit_trades(held_trim, rets, sector, notional=NOTIONAL)
    return net, trades


SPEC = StrategySpec(
    id="earnings_announcement_premium",
    family="equity_event_premium",
    title="Earnings Announcement Premium (sector-matched, beta-neutral L/S)",
    markets=["US equities"],
    data_desc=("Sharadar SEP (survivorship-clean, split/div-adjusted, delisted incl.) "
               "mid-cap US equities + Sharadar SF1 ARQ 'datekey' (filing date) used "
               "point-in-time to PREDICT the next quarterly announcement window. "
               "Trailing-60d SEP dollar-ADV liquidity floor; us_universe sector tags."),
    pre_registration=(
        "H: Stocks earn an abnormal positive return over the predictable window "
        "surrounding their scheduled quarterly earnings announcement (Frazzini-Lamont). "
        "Predict the next announcement strictly from the last KNOWN SF1 datekey + ~91d "
        "(no calendardate, no look-ahead). Go LONG predicted announcers vs SHORT matched "
        "NON-announcers in the SAME sector, inverse-vol weighted, then residualize ex-ante "
        "60d beta + market (dollar-neutral). Monthly rebalance, signals lagged 1 day, "
        "~8bps costs on turnover. Edge must persist net-of-cost and generalise across "
        "untouched size/sector slices."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "win_tight": {"win_lo": -5, "win_hi": 15},
        "horizon_90": {"horizon": 90},
        "tv_15": {"target_vol": 0.15},
    },
    scope='broad',
    generalization_universes=['large', 'small', 'sectors'],
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)