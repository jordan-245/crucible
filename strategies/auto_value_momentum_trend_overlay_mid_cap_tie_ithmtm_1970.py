# mid_valmom_trend_overlay.py
# Value (point-in-time B/M) x Momentum (12-1) composite, long-only top tercile,
# inverse-vol sized, weekly rebalance with no-trade hysteresis, broad-market
# trend overlay to cash. MID-cap TIER-NATIVE re-tune (grid searched on Mid so
# PBO/DSR are measured on the tier's OWN data). Survivorship-clean Sharadar.
#
# No external side effects: pure compute over OWNED Sharadar data via adapters.

from sdk.harness import StrategySpec
from sdk.adapters import (sep_panel, us_universe, sf1, yf_panel, fred_series,
                          trend_returns, carry_returns, inv_vol_position)
import numpy as np, pandas as pd

# Sharadar/Morningstar sector labels (the 11 GICS-style buckets in TICKERS.sector)
_SECTORS = ['Healthcare', 'Basic Materials', 'Financial Services',
            'Consumer Cyclical', 'Technology', 'Consumer Defensive',
            'Industrials', 'Real Estate', 'Energy',
            'Communication Services', 'Utilities']

# ticker -> sector map, populated by load_data() (used for trade-level sector tags)
_SECTOR_MAP = {}

START = "2004-01-01"   # warmup for 252d momentum + 200d trend MA before ~2006
SECTOR_TOP_N = 120     # ~11 * 120 -> ~1.3k liquid mid-cap names (bounded universe)

DEFAULT_PARAMS = dict(
    w_value=0.5, w_mom=0.5,        # 50/50 composite
    vol_lb=63,                     # inverse-vol lookback (~quarter)
    mom_lb=252, mom_skip=21,       # 12-1 momentum
    top_q=2.0 / 3.0,              # long top tercile (score >= 2/3 quantile)
    hysteresis_band=0.10,          # keep held names whose score >= (2/3 - band) quantile
    winsor_lo=5, winsor_hi=95,     # cross-sectional winsorisation
    use_trend=True, trend_ma=200,  # broad-market trend overlay (risk-off to cash)
    cost_bps=8.0,                  # ~8bps on turnover
    min_names=30,                  # min valid cross-section to act on a rebalance
    notional=1_000_000.0,          # book notional for trade-level accounting
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _winsor_z(df, lo, hi):
    """Per-date (row) winsorise at lo/hi pct then cross-sectional z-score."""
    ql = df.quantile(lo / 100.0, axis=1)
    qh = df.quantile(hi / 100.0, axis=1)
    c = df.clip(lower=ql, upper=qh, axis=0)
    mu = c.mean(axis=1)
    sd = c.std(axis=1).replace(0.0, np.nan)
    return c.sub(mu, axis=0).div(sd, axis=0)


def _weekly_rebalance_dates(idx):
    """Last trading day of each ISO week present in idx."""
    s = pd.Series(idx)
    last = s.groupby(idx.to_period('W')).max()
    return pd.DatetimeIndex(sorted(last.values))


def _build_bvps(tickers, price_index):
    """Point-in-time bvps panel: pivot SF1 ARQ on datekey (filing date), ffill
    onto the daily price index. NEVER calendardate -> no look-ahead."""
    raw = sf1(tickers, ['bvps'], dimension='ARQ')
    if raw is None or len(raw) == 0:
        return pd.DataFrame(index=price_index, columns=tickers, dtype=float)
    raw = raw.copy()
    lc = {str(c).lower(): c for c in raw.columns}
    if 'datekey' not in lc or 'ticker' not in lc:
        raw = raw.reset_index()
        lc = {str(c).lower(): c for c in raw.columns}
    tcol, dcol = lc['ticker'], lc['datekey']
    bcol = lc.get('bvps', 'bvps')
    sub = raw[[tcol, dcol, bcol]].dropna(subset=[dcol]).copy()
    sub[dcol] = pd.to_datetime(sub[dcol])
    piv = sub.pivot_table(index=dcol, columns=tcol, values=bcol,
                          aggfunc='last').sort_index()
    full = piv.index.union(price_index)
    return piv.reindex(full).ffill().reindex(price_index)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data():
    """Survivorship-clean MID-cap panel: SEP closeadj prices + SF1 ARQ bvps.
    Universe built per-sector (diversified) and bounded to ~1.3k liquid names."""
    _SECTOR_MAP.clear()
    tickers = []
    for sec in _SECTORS:
        try:
            ts = us_universe(sector=sec, category='Domestic Common Stock',
                             marketcap='Mid', include_delisted=True,
                             top_n=SECTOR_TOP_N)
        except Exception:
            ts = []
        for t in ts:
            _SECTOR_MAP[t] = sec
        tickers.extend(ts)
    tickers = sorted(set(tickers))

    px = sep_panel(tickers, START, field='closeadj')
    px = px.sort_index()
    px = px.loc[:, px.notna().sum() > 252]          # drop names with no usable history
    bvps = _build_bvps(list(px.columns), px.index)
    bvps = bvps.reindex(columns=px.columns)

    panel = pd.concat({'price': px, 'bvps': bvps}, axis=1)
    panel.attrs['sectors'] = dict(_SECTOR_MAP)
    return panel


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------
def signal(panel, **params):
    p = {**DEFAULT_PARAMS, **params}
    px = panel['price'].astype(float)
    bvps = panel['bvps'].astype(float)
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)

    # --- factors ---------------------------------------------------------
    # Value: point-in-time book-to-market (bvps already datekey-ffilled).
    bm = bvps / px
    bm = bm.replace([np.inf, -np.inf], np.nan)
    # Momentum: 12-1 (skip most recent month).
    mom = px.shift(p['mom_skip']) / px.shift(p['mom_lb']) - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)

    zv = _winsor_z(bm, p['winsor_lo'], p['winsor_hi'])
    zm = _winsor_z(mom, p['winsor_lo'], p['winsor_hi'])

    wv, wm = float(p['w_value']), float(p['w_mom'])
    comp = wv * zv.fillna(0.0) + wm * zm.fillna(0.0)
    if wv > 0 and wm > 0:
        valid = zv.notna() & zm.notna()
    elif wv > 0:
        valid = zv.notna()
    else:
        valid = zm.notna()
    score = comp.where(valid)

    # --- inverse-vol weights --------------------------------------------
    vol = rets.rolling(p['vol_lb']).std()
    invvol = (1.0 / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # --- weekly selection with hysteresis -------------------------------
    reb_dates = _weekly_rebalance_dates(px.index)
    top_q = float(p['top_q'])
    hold_q = max(0.0, top_q - float(p['hysteresis_band']))

    weights = pd.DataFrame(index=reb_dates, columns=px.columns, dtype=float)  # NaN init
    held = set()
    for t in reb_dates:
        s = score.loc[t].dropna()
        if len(s) < p['min_names']:
            continue
        enter_thr = s.quantile(top_q)
        hold_thr = s.quantile(hold_q)
        new_enter = set(s.index[s >= enter_thr])
        kept = {n for n in held if (n in s.index) and (s[n] >= hold_thr)}
        names = new_enter | kept
        if not names:
            weights.loc[t] = 0.0
            held = set()
            continue
        iv = invvol.loc[t, list(names)].replace([np.inf, -np.inf], np.nan).dropna()
        if iv.empty or iv.sum() <= 0:
            weights.loc[t] = 0.0
            held = set()
            continue
        w = iv / iv.sum()
        row = pd.Series(0.0, index=px.columns)
        row[w.index] = w.values
        weights.loc[t] = row
        held = set(w.index)

    # daily-held, then LAG 1 day (no same-day execution look-ahead)
    W = weights.reindex(px.index).ffill().fillna(0.0)
    W = W.shift(1).fillna(0.0)

    # --- broad-market trend overlay -------------------------------------
    if p['use_trend']:
        eqw = (1.0 + rets.mean(axis=1).fillna(0.0)).cumprod()   # equal-weight univ index
        ma = eqw.rolling(int(p['trend_ma'])).mean()
        trend_on = (eqw > ma).astype(float)
        trend_on = trend_on.shift(1).reindex(px.index).fillna(0.0)  # lag 1 day
    else:
        trend_on = pd.Series(1.0, index=px.index)
    W_eff = W.mul(trend_on, axis=0)

    # --- returns net of cost --------------------------------------------
    gross = (W_eff * rets).sum(axis=1)
    turnover = W_eff.diff().abs().sum(axis=1)
    turnover.iloc[0] = W_eff.iloc[0].abs().sum()
    cost = turnover.fillna(0.0) * (p['cost_bps'] / 1e4)
    net = (gross - cost)

    exposure = W_eff.abs().sum(axis=1)
    live = exposure[exposure > 0]
    if not live.empty:
        net = net.loc[live.index.min():]
    net = net.fillna(0.0)
    net.name = "mid_valmom_trend"

    # --- trades: one per held position run ------------------------------
    notional = float(p['notional'])
    contrib = (W_eff * rets)
    dates = W_eff.index
    sectors = panel.attrs.get('sectors', _SECTOR_MAP)
    trades = []
    for tk in W_eff.columns:
        w = W_eff[tk].values
        active = w > 1e-9
        if not active.any():
            continue
        padded = np.concatenate([[False], active, [False]]).astype(int)
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        cser = contrib[tk]
        wser = W_eff[tk]
        for s_i, e_i in zip(starts, ends):
            run_w = wser.iloc[s_i:e_i + 1]
            run_pnl = float(cser.iloc[s_i:e_i + 1].sum()) * notional
            avg_w = float(run_w.mean())
            trades.append({
                "ticker": tk,
                "sector": sectors.get(tk, "Unknown"),
                "entry_date": dates[s_i].strftime("%Y-%m-%d"),
                "exit_date": dates[e_i].strftime("%Y-%m-%d"),
                "hold_days": int(e_i - s_i + 1),
                "position_value": float(avg_w * notional),
                "pnl": float(run_pnl),
            })

    return net, trades


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------
SPEC = StrategySpec(
    id="mid_valmom_trend_overlay",
    family="value_momentum",
    title="Value x Momentum + trend-overlay — MID-cap TIER-NATIVE re-tune",
    markets=["US Mid-cap equities (Sharadar SEP/SF1, survivorship-clean)"],
    data_desc=("Sharadar SEP closeadj prices + SF1 ARQ bvps (datekey-lagged, "
               "point-in-time B/M). Universe = us_universe(marketcap='Mid', "
               "Domestic Common Stock, include_delisted=True) built per-sector "
               "(11 GICS-style sectors, top_n~120 each) -> ~1.3k liquid names."),
    pre_registration=(
        "H: a 50/50 point-in-time Value (B/M) x Momentum (12-1) composite, "
        "long-only top tercile, inverse-vol sized, weekly-rebalanced with a "
        "no-trade hysteresis band and a broad-market 200d trend overlay to cash, "
        "delivers positive net-of-cost expectancy on US MID-caps. The V+M "
        "MECHANISM is already CPCV-proven to generalise across cap tiers, but a "
        "config tuned off-tier does NOT transfer (PBO 0.71-0.97). This book is "
        "grid-searched ON MID so PBO/DSR are measured on the tier's OWN OOS; the "
        "goal is a default that ranks ROBUST (low own-tier PBO), completing the "
        "missing leg of the multi-tier V+M portfolio (Small overlay PBO 0.078)."),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULT_PARAMS,
    grid={
        "default": {},                                   # primary
        "value_only": {"w_value": 1.0, "w_mom": 0.0},
        "mom_only": {"w_value": 0.0, "w_mom": 1.0},
        "no_trend": {"use_trend": False},
        "tight_band": {"hysteresis_band": 0.0},
        "trend_ma_150": {"trend_ma": 150},
    },
    scope='broad',                                       # V+M is a universal premium
    generalization_universes=['large', 'small', 'sectors'],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)