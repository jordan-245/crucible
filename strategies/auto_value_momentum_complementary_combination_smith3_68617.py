"""
value_momentum_complementary_combination
----------------------------------------
Cross-sectional EQUITY factor BOOK combining two standalone, complementary premia
("Value and Momentum Everywhere", AQR): 12-1 price MOMENTUM and book-to-price VALUE.
The two legs are individually real but negatively correlated, so the COMBINATION is the
edge (lower drawdown / smoother book than either leg alone). Tested in the SMALL/MID-cap,
survivorship-clean Sharadar universe where these anomalies actually live (they are
arbitraged away in the largest liquid names -> false nulls there).

Standalone legs are pre-declared in the grid (mom_only / value_only) so the DSR
effective-N pays the honest search burden, and the combination is NOT a reflexive 50/50
hedge bolt-on: both legs carry their own premium.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np, pandas as pd

START = "2003-01-01"
CAPITAL = 100_000.0

DEFAULTS = dict(mom_lb=252, mom_skip=21, top_n=25, vol_lb=63, w_value=0.5, cost_bps=8.0)

# Sharadar/Morningstar sector vocabulary (used to build a sectored universe + a real
# ticker->sector map for the deployment-sanity trade ledger).
SECTORS = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Consumer Defensive', 'Energy', 'Basic Materials',
           'Real Estate', 'Communication Services', 'Utilities']

_SECTOR_MAP = {}  # module-level fallback (process-persistent across load_data->signal)


# ----------------------------------------------------------------------------- helpers
def _build_universe(per_sector_n=120):
    """Bounded, survivorship-clean mid-cap universe built per-sector so we also get a
    real ticker->sector map for free."""
    tic_sector = {}
    for s in SECTORS:
        try:
            ts = us_universe(sector=s, category='Domestic Common Stock',
                             marketcap='Mid', include_delisted=True, top_n=per_sector_n)
        except Exception:
            ts = []
        for t in (ts or []):
            tic_sector.setdefault(t, s)
    if len(tic_sector) < 100:
        # fallback: unsectored bounded universe, deterministic sector buckets
        try:
            ts = us_universe(category='Domestic Common Stock', marketcap='Mid',
                             include_delisted=True, top_n=1200)
        except Exception:
            ts = []
        for t in (ts or []):
            tic_sector.setdefault(t, SECTORS[hash(t) % len(SECTORS)])
    return tic_sector


def _pit_panel(fund, field, dates, cols):
    """Point-in-time daily panel of a fundamental field, forward-filled by 'datekey'
    (filing date) to avoid look-ahead."""
    if fund is None or len(fund) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    f = fund.copy()
    if ('ticker' not in f.columns) or ('datekey' not in f.columns):
        f = f.reset_index()
    if field not in f.columns:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    f = f[['ticker', 'datekey', field]].dropna()
    f['datekey'] = pd.to_datetime(f['datekey'])
    f = f.sort_values('datekey')
    out = {}
    for tic, g in f.groupby('ticker'):
        s = pd.Series(g[field].values, index=pd.DatetimeIndex(g['datekey'].values))
        s = s[~s.index.duplicated(keep='last')]
        s = s.reindex(s.index.union(dates)).sort_index().ffill().reindex(dates)
        out[tic] = s
    return pd.DataFrame(out).reindex(columns=cols)


def _xs_z(df):
    """Cross-sectional (per-day) z-score, clipped to tame outliers."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-3.0, 3.0)


# ------------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    global _SECTOR_MAP
    tic_sector = _build_universe()
    _SECTOR_MAP = tic_sector
    tickers = sorted(tic_sector.keys())

    px = sep_panel(tickers, START, field='closeadj').sort_index()
    px = px.reindex(columns=[c for c in tickers if c in px.columns])

    try:
        fund = sf1(list(px.columns), ['bvps'], dimension='ARQ')
        bvps = _pit_panel(fund, 'bvps', px.index, px.columns)
    except Exception:
        bvps = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)

    panel = pd.concat({'px': px, 'bvps': bvps}, axis=1)
    try:
        panel.attrs['sector'] = tic_sector
    except Exception:
        pass
    return panel


# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    mom_lb, mom_skip = int(p['mom_lb']), int(p['mom_skip'])
    top_n, vol_lb = int(p['top_n']), int(p['vol_lb'])
    w_val, cost_bps = float(p['w_value']), float(p['cost_bps'])

    sector_map = (panel.attrs.get('sector') if hasattr(panel, 'attrs') else None) or _SECTOR_MAP

    px = panel['px'].sort_index()
    bvps = panel['bvps'].reindex_like(px) if 'bvps' in panel.columns.get_level_values(0) \
        else pd.DataFrame(index=px.index, columns=px.columns, dtype=float)

    idx = px.index
    rets = px.pct_change()
    vol = rets.rolling(vol_lb).std()

    # --- two complementary legs -------------------------------------------------
    mom = px.shift(mom_skip) / px.shift(mom_lb) - 1.0          # 12-1 momentum
    bp = (bvps / px).where(lambda x: x > 0)                    # book-to-price (>0 only)
    val = np.log(bp)

    mom_z = _xs_z(mom)
    val_z = _xs_z(val)

    have_mom = mom_z.notna()
    have_val = val_z.notna()
    # neutralise a missing leg to its cross-sectional mean (0 after z); require momentum.
    combo = (w_val * val_z.where(have_val, 0.0)) + ((1.0 - w_val) * mom_z.where(have_mom, 0.0))
    combo = combo.where(have_mom)

    # --- weekly rebalance: last trading day of each ISO week --------------------
    iso = idx.isocalendar()
    week_id = pd.Series((iso['year'].astype(int) * 100 + iso['week'].astype(int)).values, index=idx)
    is_last = week_id.values != np.r_[week_id.values[1:], -1]
    rebal_dates = idx[is_last]

    # --- build target weights (inverse-vol, top-N) held to next rebalance -------
    weights = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    for rd in rebal_dates:
        scores = combo.loc[rd].dropna()
        if len(scores) < top_n:
            continue
        picks = scores.nlargest(top_n).index
        iv = (1.0 / vol.loc[rd, picks]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(iv) < 2:
            continue
        w = iv / iv.sum()
        row = pd.Series(0.0, index=px.columns)
        row[w.index] = w.values
        weights.loc[rd] = row.values
    weights = weights.ffill().fillna(0.0)

    # lag 1 day (no look-ahead): positions effective the day AFTER signal/rebalance
    pos = weights.shift(1).fillna(0.0)

    # --- net-of-cost daily portfolio returns ------------------------------------
    gross = (pos * rets).sum(axis=1)
    turnover = (pos - pos.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost)

    active = pos.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    net = net.fillna(0.0)
    net.name = 'value_momentum_complementary_combination'

    # --- trade ledger (one trade per held position run) -------------------------
    trades = []
    held_cols = pos.columns[(pos > 0).any().values]
    for tic in held_cols:
        a = (pos[tic] > 0).values
        if not a.any():
            continue
        starts = np.where(a & ~np.r_[False, a[:-1]])[0]
        ends = np.where(a & ~np.r_[a[1:], False])[0]
        wv = pos[tic].values
        rv = np.nan_to_num(rets[tic].values)
        for s_i, e_i in zip(starts, ends):
            sl = slice(s_i, e_i + 1)
            avg_w = float(np.nanmean(wv[sl]))
            pnl = float(CAPITAL * np.nansum(wv[sl] * rv[sl]))
            trades.append({
                'ticker': str(tic),
                'sector': str(sector_map.get(tic, 'Unknown')),
                'entry_date': idx[s_i].strftime('%Y-%m-%d'),
                'exit_date': idx[e_i].strftime('%Y-%m-%d'),
                'hold_days': int(e_i - s_i + 1),
                'position_value': float(avg_w * CAPITAL),
                'pnl': pnl,
            })

    return net, trades


# -------------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id='value_momentum_complementary_combination',
    family='equity_factor_combination',
    title='Value + Momentum complementary cross-sectional equity book (mid-cap)',
    markets=['US equities (Sharadar SEP, mid-cap, survivorship-clean incl. delisted)'],
    data_desc=('Survivorship-clean Sharadar SEP split/div-adjusted daily closes + Sharadar '
               'SF1 ARQ fundamentals (bvps, as-of datekey). Bounded mid-cap universe built '
               'per-sector (~1.0-1.3k liquid names). 12-1 momentum + book-to-price value, '
               'cross-sectional z-scores, inverse-vol, top-N long book, weekly rebalance, '
               '8bps turnover cost, signals lagged 1 day.'),
    pre_registration=(
        'HYPOTHESIS: 12-1 momentum and book-to-price value are two real, standalone, '
        'NEGATIVELY-correlated equity premia; combining them (the AQR "everywhere" result) '
        'delivers a smoother book (lower drawdown) than either leg alone at comparable '
        'Sharpe. We test the standalone legs (grid: mom_only, value_only) AND the 50/50 '
        'combination; the COMBINATION is judged on tail/DD reduction without diluting the '
        'better standalone Sharpe. Universe = mid-cap (anomalies are arbitraged out of the '
        'largest liquid names). BROAD factor claim -> must generalise to untouched '
        'large/small/sector slices in stage-2, else treat as overfit outlier. '
        'Holdout 2022-01-01 onward is untouched.'
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid={
        'default': {},
        'mom_only': {'w_value': 0.0},
        'value_only': {'w_value': 1.0},
        'mom_6m': {'mom_lb': 126},
        'top_40': {'top_n': 40},
    },
    scope='broad',
    generalization_universes=['large', 'small', 'sectors'],
    holdout_start="2022-01-01",
    deploy_max_positions=25,
)