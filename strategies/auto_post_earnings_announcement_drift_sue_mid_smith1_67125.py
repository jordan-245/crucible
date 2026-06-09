from sdk.harness import StrategySpec
from sdk.adapters import (sep_panel, us_universe, sf1, yf_panel, fred_series,
                          trend_returns, carry_returns, inv_vol_position)
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------
# Post-Earnings-Announcement-Drift via Standardized Unexpected Earnings (SUE)
# Thesis: stocks with the largest positive seasonal earnings surprise drift up
# (and the largest negatives drift down) for ~60 trading days post-filing. The
# anomaly is strongest in mid-caps (less-arbitraged than mega-caps, more
# liquid/clean than micro-caps). Standalone cross-sectional premium first; a
# SMALL vol-matched trend tail-overlay only to clip the crash tail.
# ----------------------------------------------------------------------------

START = "2004-01-01"
UNIVERSE_N = 1500
NOTIONAL = 10000.0  # nominal per-name notional for trade-log diagnostics only

DEFAULTS = dict(
    hold_days=60,     # 60 trading-day overlapping post-announcement cohort
    vol_lb=60,        # inverse-vol lookback
    target_vol=0.10,  # 10% annualized vol target
    max_pos=200,      # broad quintile book
    rebalance='ME',   # monthly rebalance (pandas 'ME' = month-end)
    cost_bps=25,      # central cost (15/25/50 tested in grid)
    trend_w=0.15,     # small trend tail-overlay (sized to minimise drag)
    top_q=0.80,       # long top quintile
    bot_q=0.20,       # short bottom quintile
)

SECTORS = ['Healthcare', 'Technology', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Consumer Defensive', 'Energy', 'Basic Materials',
           'Real Estate', 'Utilities', 'Communication Services']


def load_data() -> pd.DataFrame:
    # Survivorship-clean mid-cap universe (delisted INCLUDED)
    tickers = us_universe(marketcap='Mid', category='Domestic Common Stock',
                          include_delisted=True, top_n=UNIVERSE_N)
    tickers = list(dict.fromkeys(tickers))
    # SEP split+div adjusted closes (survivorship-clean)
    px = sep_panel(tickers, start=START, field='closeadj')
    # Point-in-time fundamentals: ARQ EPS, gated on 'datekey' (filing date)
    eps = sf1(list(px.columns), ['eps'], dimension='ARQ')
    # Sector map (diagnostics for the trade log)
    sec_map = {}
    for s in SECTORS:
        try:
            for t in us_universe(sector=s, category='Domestic Common Stock',
                                 include_delisted=True, marketcap='Mid'):
                sec_map[t] = s
        except Exception:
            pass
    px.attrs['eps'] = eps
    px.attrs['sec_map'] = sec_map
    return px


def _sue_events(eps):
    """Return list of (datekey, ticker, SUE) point-in-time surprise events.
    SUE_q = (EPS_q - EPS_{q-4}) / std(prior 8 seasonal EPS changes)."""
    eps = eps.copy()
    eps.columns = [str(c).lower() for c in eps.columns]
    if 'ticker' not in eps.columns or 'datekey' not in eps.columns:
        eps = eps.reset_index()
        eps.columns = [str(c).lower() for c in eps.columns]
    eps['datekey'] = pd.to_datetime(eps['datekey'], errors='coerce')
    eps = eps.dropna(subset=['datekey', 'eps'])
    events = []
    for tkr, g in eps.groupby('ticker'):
        g = g.sort_values('datekey').drop_duplicates('datekey', keep='last')
        e = g['eps'].astype(float).reset_index(drop=True)
        dk = g['datekey'].reset_index(drop=True)
        sc = e - e.shift(4)                    # seasonal (YoY) quarterly EPS change
        denom = sc.shift(1).rolling(8).std()   # std of PRIOR 8 seasonal changes
        sue = sc / denom
        for i in range(len(sue)):
            v = sue.iloc[i]
            if np.isfinite(v):
                events.append((dk.iloc[i], tkr, float(v)))
    return events


def _extract_trades(pos, rets, sec_map):
    """One trade dict per contiguous held-position run (constant sign)."""
    trades = []
    dates = pos.index
    n = len(dates)
    for tkr in pos.columns:
        w = pos[tkr].values
        r = rets[tkr].values
        i = 0
        while i < n:
            if not np.isfinite(w[i]) or w[i] == 0.0:
                i += 1
                continue
            sgn = np.sign(w[i])
            j, pnl, wsum, cnt = i, 0.0, 0.0, 0
            while j < n and np.isfinite(w[j]) and w[j] != 0.0 and np.sign(w[j]) == sgn:
                pnl += w[j] * r[j] * NOTIONAL
                wsum += abs(w[j])
                cnt += 1
                j += 1
            if cnt > 0:
                trades.append({
                    'ticker': tkr,
                    'sector': sec_map.get(tkr, 'Unknown'),
                    'entry_date': dates[i].strftime('%Y-%m-%d'),
                    'exit_date': dates[min(j, n - 1)].strftime('%Y-%m-%d'),
                    'hold_days': int(cnt),
                    'position_value': float(wsum / cnt * NOTIONAL),
                    'pnl': float(pnl),
                })
            i = j
    return trades


def signal(panel, **params):
    p = dict(DEFAULTS); p.update(params)
    px = panel
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)

    eps = px.attrs.get('eps')
    if eps is None:
        eps = sf1(list(px.columns), ['eps'], dimension='ARQ')
    sec_map = px.attrs.get('sec_map', {})

    dates = px.index
    cols = list(px.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    hold = int(p['hold_days'])

    # Daily SUE matrix via OVERLAPPING 60-day post-announcement cohorts.
    # Signal appears on the filing date (datekey); inv_vol_position applies the
    # 1-day execution lag -> no look-ahead. Newer filings overwrite older ones.
    arr = np.full((len(dates), len(cols)), np.nan)
    for dk, tkr, v in sorted(_sue_events(eps), key=lambda x: x[0]):
        j = col_idx.get(tkr)
        if j is None:
            continue
        s = int(dates.searchsorted(pd.Timestamp(dk), side='left'))
        if s >= len(dates):
            continue
        e = min(s + hold, len(dates))
        arr[s:e, j] = v
    sue_df = pd.DataFrame(arr, index=dates, columns=cols)

    # Cross-sectional quintile L/S among currently-active announcers.
    rk = sue_df.rank(axis=1, pct=True)
    sig = pd.DataFrame(0.0, index=dates, columns=cols)
    sig = sig.mask(rk >= p['top_q'], 1.0)
    sig = sig.mask(rk <= p['bot_q'], -1.0)

    # Inverse-vol sizing, monthly rebalance, 10% vol target, 1-day lag inside.
    pos = inv_vol_position(sig, rets, target_vol=p['target_vol'],
                           vol_lb=int(p['vol_lb']), max_pos=int(p['max_pos']),
                           rebalance=p['rebalance'])
    pos = pos.reindex(index=dates, columns=cols).fillna(0.0)

    r = rets.reindex(index=dates, columns=cols).fillna(0.0)
    gross = (pos * r).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * (p['cost_bps'] / 1e4)          # 15/25/50 bps cost model
    pead = (gross - cost)

    # Optional SMALL vol-matched trend tail-overlay (clip the crash tail only).
    out = pead
    if p['trend_w'] and p['trend_w'] > 0:
        try:
            tr, _ = trend_returns()
            comb = pd.concat([pead.rename('p'), tr.rename('t')], axis=1).dropna()
            if len(comb) > 50 and comb['t'].std() > 0:
                scl = comb['p'].std() / comb['t'].std()
                w = float(p['trend_w'])
                out = (1 - w) * comb['p'] + w * (comb['t'] * scl)
        except Exception:
            out = pead

    out = out.dropna()
    out.name = 'pead_sue_book'

    trades = _extract_trades(pos, r, sec_map)
    return out, trades


SPEC = StrategySpec(
    id="pead_sue_midcap_v1",
    family="post_earnings_drift",
    title="PEAD / SUE earnings-surprise drift (mid-cap L/S, monthly, optional trend overlay)",
    markets=["us_equities_midcap"],
    data_desc=("Sharadar SEP closeadj (survivorship-clean, delisted incl) + SF1 ARQ EPS "
               "gated on filing 'datekey'; mid-cap Domestic Common Stock, top 1500 liquid; "
               "trend overlay from validated 21-market CTA leg."),
    pre_registration=(
        "HYPOTHESIS: Post-Earnings-Announcement-Drift. Stocks in the top quintile of "
        "Standardized Unexpected Earnings drift positively (bottom quintile negatively) for "
        "~60 trading days after the filing. SUE_q = (EPS_q - EPS_{q-4}) / std(prior 8 seasonal "
        "EPS changes), computed from SF1 ARQ EPS, made tradeable only as-of 'datekey' (no "
        "look-ahead, 1-day exec lag). Cross-sectional quintile rank among currently-active "
        "announcers; long top minus short bottom; 60-day overlapping cohort hold; monthly "
        "rebalance; inverse-vol sizing to 10% vol target; net of 15/25/50 bps turnover cost. "
        "Universe = mid-cap (less arbitraged than mega-caps, cleaner than micro-caps). "
        "PRIMARY = standalone premium; a SMALL (~15%) vol-matched trend overlay is added ONLY "
        "to clip the crash tail without diluting the standalone edge. PASS REQUIRES the "
        "standalone leg to carry the Sharpe and to GENERALISE to untouched large/small/sector "
        "slices (it is a universal cross-sectional anomaly, not a universe-specific quirk)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "cost15": {"cost_bps": 15},
        "cost50": {"cost_bps": 50},
        "hold40": {"hold_days": 40},
        "hold90": {"hold_days": 90},
        "standalone": {"trend_w": 0.0},
        "overlay30": {"trend_w": 0.30},
    },
    scope='broad',
    generalization_universes=['large', 'small', 'sectors'],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)