Looking at the error: the fix is that **book-to-market must use the actual (unadjusted) market price**, not `closeadj` (which back-adjusts for *future* dividends → look-ahead + high-dividend bias). Returns/momentum keep `closeadj` (correct for total return). I also completed the truncated function.

```python
"""
Value × Momentum complementary combination — US SMALL/MID cap, LONG-ONLY turnover-buffered TILT.

Thesis (Asness/Moskowitz/Pedersen 'Value & Momentum Everywhere'):
value and cross-sectional momentum carry opposite tails (value = pro-cyclical mean-reversion,
momentum = trend that crashes on post-crash reversals) and are ~-0.4/-0.5 correlated -> the
DIVERSIFICATION of the pair is the edge. Parent (atlas-equity-factors) tested each STANDALONE in
LARGE-liquid caps -> FAIL/closed. Mutations here: (1) move into the small/mid limits-to-arbitrage
corner, (2) capture the pair as a DEPLOYABLE long-only active tilt vs the EW tradable universe
(no small-cap borrow -> fixes anti-pattern #8), (3) turnover-buffer (hysteresis band + monthly cap)
and conservative ~40bps round-trip cost so a PROMOTE means a genuinely tradable book.

Pre-registered verdict: the LONG-ONLY COMBINED tilt beats BOTH single-factor long-only tilts on
risk-adjusted active terms (higher Sharpe/IR, lower maxDD) net of cost with the band ON. Holdout
(2022+) is the only arbiter. The mom_only / value_only grid variants ARE the single-factor baselines.

Daily series = ACTIVE return (inverse-vol top-tercile tilt − equal-weight eligible benchmark),
net of cost. Equity beta is not the edge; the active component isolates the diversification premium.

FIX: book-to-market is computed on the UNADJUSTED market price (closeunadj), NOT closeadj.
closeadj at date d embeds the cumulative dividend/split factor for events AFTER d (look-ahead) and
biases high-dividend payers to look 'cheap'. B/M = bvps(point-in-time, datekey) / actual price.
Momentum & returns correctly use closeadj (total-return; the future factor cancels in the ratio).

OWNED/FREE data only: Sharadar SEP (closeadj, closeunadj, volume), SF1 (bvps via datekey), TICKERS.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np, pandas as pd

START = "2001-01-01"
NOTIONAL = 1_000_000.0
_SECTORS = ['Healthcare', 'Financial Services', 'Consumer Cyclical', 'Technology',
            'Consumer Defensive', 'Industrials', 'Real Estate', 'Energy',
            'Communication Services', 'Utilities', 'Basic Materials']


# ----------------------------------------------------------------------------- data
def _universe_and_sectors():
    """Survivorship-clean small+mid universe (delisted incl.) + ticker->sector map.
    Bounded per (cap,sector) so the full panel stays ~1300 liquid names (rails-safe)."""
    sector_map, tickers = {}, []
    for cap in ('Small', 'Mid'):
        for sec in _SECTORS:
            try:
                names = us_universe(sector=sec, marketcap=cap,
                                    category='Domestic Common Stock',
                                    include_delisted=True, top_n=60)
            except Exception:
                names = []
            for t in names:
                sector_map.setdefault(t, sec)
                tickers.append(t)
    return sorted(set(tickers)), sector_map


def _bvps_panel(tickers, price_index):
    """Point-in-time book value per share, forward-filled from FILING date (datekey)."""
    raw = sf1(tickers, ['bvps'], dimension='ARQ')
    cols = {c.lower(): c for c in raw.columns}
    if 'datekey' in cols and 'ticker' in cols and 'bvps' in cols:
        tcol, dcol, vcol = cols['ticker'], cols['datekey'], cols['bvps']
        df = raw[[tcol, dcol, vcol]].copy()
        df[dcol] = pd.to_datetime(df[dcol])
        df = df.dropna(subset=[vcol])
        p = df.pivot_table(index=dcol, columns=tcol, values=vcol, aggfunc='last').sort_index()
    else:  # already a (date x ticker) panel
        p = raw.copy()
        p.index = pd.to_datetime(p.index)
        p = p.sort_index()
    full = price_index.union(p.index)
    p = p.reindex(full).ffill().reindex(price_index)
    return p.reindex(columns=tickers)


def load_data():
    tickers, sector_map = _universe_and_sectors()
    price = sep_panel(tickers, START, field='closeadj').sort_index()   # total-return adjusted
    mktpx = sep_panel(tickers, START, field='closeunadj')              # ACTUAL market price
    volume = sep_panel(tickers, START, field='volume')
    mktpx = mktpx.reindex(index=price.index, columns=price.columns)
    volume = volume.reindex(index=price.index, columns=price.columns)
    bvps = _bvps_panel(list(price.columns), price.index)
    # book-to-market on the ACTUAL unadjusted price -> no dividend look-ahead, no payer bias
    btm = bvps / mktpx.replace(0.0, np.nan)
    panel = pd.concat({'price': price, 'mktpx': mktpx, 'volume': volume, 'btm': btm}, axis=1)
    panel.attrs['sector_map'] = sector_map
    return panel


# --------------------------------------------------------------------------- signal
def _zscore(x, mask):
    v = x[mask].astype(float)
    sd = v.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=v.index)
    return (v - v.mean()) / sd


def signal(panel, **params):
    p = dict(exit_pct=0.50, enter_pct=0.70, max_turnover=0.15,
             w_value=0.5, w_mom=0.5, cost_bps=20.0,   # cost_bps = PER-SIDE bps (=40bps round trip)
             vol_lb=60, adv_floor=1.0e6)
    p.update(params)
    exit_pct, enter_pct = float(p['exit_pct']), float(p['enter_pct'])
    max_turn = float(p['max_turnover'])
    wv, wm = float(p['w_value']), float(p['w_mom'])
    cps = float(p['cost_bps']) / 1e4
    vol_lb, adv_floor = int(p['vol_lb']), float(p['adv_floor'])

    price = panel['price']      # closeadj (total return) -> returns + momentum
    mktpx = panel['mktpx']      # unadjusted market price -> liquidity floor (value lives in btm)
    volume = panel['volume']
    btm = panel['btm']
    sector_map = panel.attrs.get('sector_map', {})
    cols = price.columns

    rets = price.pct_change()
    vol_panel = rets.rolling(vol_lb, min_periods=20).std()
    mom_panel = price.shift(21) / price.shift(252) - 1.0                 # 12-1 total-return momentum
    adv_panel = (mktpx * volume).rolling(60, min_periods=30).median()    # $-volume tradability floor
    pxff = price.ffill()

    # monthly rebalance dates (actual last trading day per month)
    rebal = pd.Series(price.index, index=price.index).resample('M').last().dropna()
    rebal_dates = pd.DatetimeIndex(rebal.values)
    rebal_dates = rebal_dates[rebal_dates.isin(price.index)]

    need_v, need_m = (wv != 0.0), (wm != 0.0)
    W_rows, B_rows = {}, {}
    trades, open_pos = [], {}
    prev_holdings = set()

    def _close(t, d):
        ed, ep, wt = open_pos.pop(t)
        xp = float(pxff.at[d, t]) if t in pxff.columns else np.nan
        if np.isfinite(xp) and np.isfinite(ep) and ep > 0:
            pv = NOTIONAL * wt
            pnl = pv * (xp / ep - 1.0)
            hd = max(int((d - ed).days), 1)
            trades.append({'ticker': t, 'sector': sector_map.get(t, 'Unknown'),
                           'entry_date': ed.strftime('%Y-%m-%d'),
                           'exit_date': d.strftime('%Y-%m-%d'),
                           'hold_days': hd, 'position_value': float(pv),
                           'pnl': float(pnl)})

    def _flush(d):
        for t in list(open_pos):
            _close(t, d)

    for d in rebal_dates:
        adv_d = adv_panel.loc[d]
        valid = adv_d > adv_floor
        val_d = btm.loc[d]
        mom_d = mom_panel.loc[d]
        if need_v:
            valid &= val_d.notna() & (val_d > 0)
        if need_m:
            valid &= mom_d.notna()
        elig = valid[valid].index
        if len(elig) < 30:
            _flush(d); prev_holdings = set()
            W_rows[d] = pd.Series(dtype=float); B_rows[d] = pd.Series(dtype=float)
            continue

        comp = pd.Series(0.0, index=elig)
        if need_v:
            comp = comp.add(wv * _zscore(val_d, valid), fill_value=0.0)
        if need_m:
            comp = comp.add(wm * _zscore(mom_d, valid), fill_value=0.0)
        comp = comp.reindex(elig).dropna()
        if comp.empty:
            _flush(d); prev_holdings = set()
            W_rows[d] = pd.Series(dtype=float); B_rows[d] = pd.Series(dtype=float)
            continue
        ranks = comp.rank(pct=True)

        # hysteresis band: exit < exit_pct, enter > enter_pct
        exit_set = {t for t in prev_holdings if (t not in ranks.index) or (ranks[t] < exit_pct)}
        keep = prev_holdings - exit_set
        cand = [t for t in ranks.index[ranks >= enter_pct] if t not in keep]
        cand.sort(key=lambda t: -ranks[t])
        target_n = int((ranks >= enter_pct).sum())
        if len(prev_holdings) == 0:
            n_add_cap = len(cand)                                       # initial build
        else:
            n_add_cap = int(np.ceil(max_turn * max(len(prev_holdings), 1)))
        n_add = max(0, min(target_n - len(keep), len(cand), n_add_cap))
        new_holdings = set(keep) | set(cand[:n_add])

        # inverse-vol weights within the long book
        vol_d = vol_panel.loc[d]
        hl = [t for t in new_holdings if t in vol_d.index
              and np.isfinite(vol_d.get(t, np.nan)) and vol_d.get(t, 0.0) > 0]
        if hl:
            iv = 1.0 / vol_d[hl]
            w = iv / iv.sum()
        else:
            w = pd.Series(dtype=float)
        new_holdings = set(w.index)
        W_rows[d] = w
        B_rows[d] = pd.Series(1.0 / len(elig), index=elig)             # EW eligible benchmark

        # trade ledger (one trade per held position run)
        for t in new_holdings - prev_holdings:
            open_pos[t] = (d, float(pxff.at[d, t]) if t in pxff.columns else np.nan,
                           float(w.get(t, 0.0)))
        for t in prev_holdings - new_holdings:
            if t in open_pos:
                _close(t, d)
        prev_holdings = new_holdings

    _flush(price.index[-1])  # close residual book at last date

    # ----------------------------------------------------- daily net-of-cost ACTIVE returns
    W = pd.DataFrame(index=price.index, columns=cols, dtype=float)
    B = pd.DataFrame(index=price.index, columns=cols, dtype=float)
    for d in rebal_dates:
        W.loc[d] = 0.0
        w = W_rows.get(d)
        if w is not None and len(w):
            W.loc[d, w