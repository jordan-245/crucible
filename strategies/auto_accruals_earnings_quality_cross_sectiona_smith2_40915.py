# Accruals / Earnings-Quality cross-sectional anomaly (Sloan 1996; Hribar-Collins CFO definition).
# Hypothesis: firms with HIGH accruals (Net Income >> Cash Flow from Ops -> low earnings quality)
# subsequently UNDER-perform; LOW-accruals (high-quality cash earnings) names OUT-perform.
# This is a classic cross-sectional anomaly that lives in SMALL caps (arbitraged away in mega-caps),
# so we test it in a survivorship-clean Sharadar small-cap universe, long low / short high accruals.

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, inv_vol_position
import numpy as np, pandas as pd

START = "2004-01-01"
_SECTORS = ["Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
            "Industrials", "Communication Services", "Consumer Defensive", "Energy",
            "Basic Materials", "Real Estate", "Utilities"]
_SECTOR_MAP = {}


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """Survivorship-clean small-cap panel: price (SEP) + daily point-in-time accruals (SF1 ART)."""
    global _SECTOR_MAP
    sector_map, tickers = {}, []
    for s in _SECTORS:
        try:
            ts = us_universe(sector=s, category='Domestic Common Stock',
                             marketcap='Small', include_delisted=True, top_n=120)
        except Exception:
            ts = []
        for t in ts:
            if t not in sector_map:
                sector_map[t] = s
                tickers.append(t)
    _SECTOR_MAP = sector_map

    px = sep_panel(tickers, start=START, field='closeadj')
    px = px.sort_index().dropna(how='all', axis=1)
    px = px.loc[:, px.notna().sum() > 252]          # need >~1yr of history per name

    # --- fundamentals -> Hribar-Collins accruals = (NetIncome - CFO) / TotalAssets  (TTM = ART)
    fund = sf1(list(px.columns), ['netinc', 'ncfo', 'assets'], dimension='ART')
    f = fund.copy()
    if isinstance(f.index, pd.MultiIndex) or 'ticker' not in f.columns or 'datekey' not in f.columns:
        f = f.reset_index()
    f = f[['ticker', 'datekey', 'netinc', 'ncfo', 'assets']].copy()
    f['datekey'] = pd.to_datetime(f['datekey'])
    f = f.dropna(subset=['datekey', 'netinc', 'ncfo', 'assets'])
    f = f[f['assets'].abs() > 0]
    f['accruals'] = (f['netinc'] - f['ncfo']) / f['assets']
    f = f.dropna(subset=['accruals']).sort_values('datekey')

    # wide accruals, as-of FILING date (datekey) -> ffill to daily -> NO look-ahead
    acc_w = f.pivot_table(index='datekey', columns='ticker', values='accruals', aggfunc='last').sort_index()
    full = px.index.union(acc_w.index).sort_values()
    acc = acc_w.reindex(full).ffill().reindex(px.index).reindex(columns=px.columns)

    panel = pd.concat({'price': px, 'accruals': acc}, axis=1)
    panel.attrs['sector_map'] = sector_map
    return panel


# ----------------------------------------------------------------------------- helpers
def _extract_trades(pos, rets, sector_map, notional=100000.0):
    trades = []
    rr = rets.reindex(pos.index).fillna(0.0)
    for tk in pos.columns:
        w = pos[tk].fillna(0.0)
        active = w.abs() > 1e-8
        if not active.any():
            continue
        r_tk = rr[tk] if tk in rr.columns else pd.Series(0.0, index=pos.index)
        blocks = (active != active.shift(fill_value=False)).cumsum()
        for _, sub in w.groupby(blocks):
            if not (sub.abs() > 1e-8).iloc[0]:
                continue
            dts = sub.index
            rseg = r_tk.reindex(dts).fillna(0.0)
            avg_w = float(sub.abs().mean())
            trades.append({
                "ticker": str(tk),
                "sector": sector_map.get(tk, "Unknown"),
                "entry_date": dts[0].strftime('%Y-%m-%d'),
                "exit_date": dts[-1].strftime('%Y-%m-%d'),
                "hold_days": int(len(dts)),
                "position_value": float(avg_w * notional),
                "pnl": float((sub * rseg).sum() * notional),
            })
    return trades


# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = dict(N=25, target_vol=0.10, vol_lb=63, cost_bps=8.0, long_short=True, min_names=60)
    p.update(params)

    px = panel['price']
    acc = panel['accruals']
    sector_map = panel.attrs.get('sector_map') or _SECTOR_MAP

    rets = px.pct_change().clip(lower=-0.95, upper=5.0)

    # cross-sectional rank: rank 1 = LOWEST accruals (best quality) -> long; highest -> short
    ranked = acc.rank(axis=1, ascending=True, method='first')
    nvalid = acc.notna().sum(axis=1)

    long_mask = ranked.le(p['N'])
    sig = long_mask.astype(float)
    if p['long_short']:
        short_mask = ranked.gt((nvalid - p['N']), axis=0) & acc.notna()
        sig = sig - short_mask.astype(float)
    sig = sig.where(nvalid >= p['min_names'], 0.0).reindex(columns=px.columns).fillna(0.0)

    # inverse-vol sized, weekly-held, lagged positions (handles no-look-ahead lag)
    pos = inv_vol_position(sig, rets, target_vol=p['target_vol'], vol_lb=int(p['vol_lb']),
                           max_pos=1000, rebalance='W')
    pos = pos.reindex(index=px.index, columns=px.columns).fillna(0.0)

    gross = (pos * rets.reindex_like(pos)).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (p['cost_bps'] / 1e4)
    daily = (gross - cost)

    active_days = pos.abs().sum(axis=1) > 0
    if active_days.any():
        daily = daily.loc[active_days.idxmax():]
    daily = daily.dropna()
    daily.name = "accruals_earnings_quality_xs"

    trades = _extract_trades(pos.loc[daily.index] if len(daily) else pos, rets, sector_map)
    return daily, trades


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="accruals_earnings_quality_xs",
    family="cross_sectional_equity_anomaly",
    title="Accruals / Earnings-Quality Cross-Sectional (Sloan) Long-Short",
    markets=["US_EQUITY"],
    data_desc=("Sharadar SEP survivorship-clean adjusted closes (delisted incl.) + SF1 ART "
               "fundamentals (netinc, ncfo, assets); ~1300 small-cap names across 11 sectors, "
               "2004-present daily. Accruals = (NetIncome - CFO)/TotalAssets, point-in-time via datekey."),
    pre_registration=(
        "PRE-REG. Premium: accruals/earnings-quality anomaly (Sloan 1996, Hribar-Collins). "
        "Predicted sign: LONG low-accruals (high cash-earnings quality) MINUS SHORT high-accruals; "
        "expect positive long-short return concentrated in small caps. Decision rule: confirm only if "
        "the STANDALONE long-short Sharpe survives holdout (>=2022-01-01) net of 8bps turnover cost with "
        "DSR clearing the rails. Accruals lagged to filing date (datekey) + positions lagged 1 day -> "
        "no look-ahead. Hedge (trend) NOT added pre-emptively; only consider a small tail-overlay if the "
        "standalone leg is real and a tail-cut adds MAR without diluting Sharpe."),
    load_data=load_data,
    signal=signal,
    default_params={"N": 25, "target_vol": 0.10, "vol_lb": 63, "cost_bps": 8.0,
                    "long_short": True, "min_names": 60},
    grid={
        "default": {},
        "concentrated": {"N": 15},
        "broad": {"N": 40},
        "long_only": {"long_short": False},
        "vol_fast": {"vol_lb": 21},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)