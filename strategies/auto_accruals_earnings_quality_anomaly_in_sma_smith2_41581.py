# Accruals (Earnings-Quality) Anomaly — Sloan (1996), small-cap cross-sectional long/short.
# Hypothesis: firms whose accounting earnings exceed operating cash flow (high accruals,
# scaled by assets) have LOW earnings quality and subsequently UNDERPERFORM; low-accruals
# firms outperform. Edge is strongest in small/illiquid names (least arbitraged), so we test
# a dollar-neutral L/S in the survivorship-clean small-cap universe. Inverse-vol, weekly
# rebalance, 1-day signal lag, 8bps/turnover costs.

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np, pandas as pd

START = "2004-01-01"
NOTIONAL = 100_000.0
SECTORS = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
           'Industrials', 'Consumer Defensive', 'Energy', 'Basic Materials',
           'Real Estate', 'Utilities', 'Communication Services']


def load_data():
    # Survivorship-clean small-cap domestic common stock universe, balanced across sectors
    # (keeps the panel bounded ~1500 names so the CPCV rails don't OOM).
    sec_map, tickers = {}, []
    per_sector = max(60, 1500 // len(SECTORS))
    for s in SECTORS:
        try:
            ts = us_universe(sector=s, category='Domestic Common Stock',
                             marketcap='Small', include_delisted=True, top_n=per_sector)
        except Exception:
            ts = []
        for t in ts:
            sec_map[t] = s
            tickers.append(t)
    if not tickers:
        tickers = us_universe(category='Domestic Common Stock', marketcap='Small', top_n=1500)
    tickers = sorted(set(tickers))

    # Split/div-adjusted daily prices, delisted included.
    px = sep_panel(tickers, start=START, field='closeadj')

    # Fundamentals for accruals: net income, operating cash flow, total assets.
    fund = sf1(tickers, ['netinc', 'ncfo', 'assets'], dimension='ARQ')
    if not {'ticker', 'datekey'}.issubset(fund.columns):
        fund = fund.reset_index()
    fund = fund[['ticker', 'datekey', 'netinc', 'ncfo', 'assets']].copy()
    fund['datekey'] = pd.to_datetime(fund['datekey'], errors='coerce')
    fund = fund.dropna(subset=['datekey']).sort_values('datekey')

    # Balance-sheet accruals proxy = (NI - operating cash flow) / total assets.
    fund['accr'] = (fund['netinc'] - fund['ncfo']) / fund['assets'].replace(0, np.nan)
    fund = fund.dropna(subset=['accr'])

    acc_wide = fund.pivot_table(index='datekey', columns='ticker',
                                values='accr', aggfunc='last')

    # As-of alignment: forward-fill from FILING date (datekey) only -> no look-ahead.
    all_idx = px.index.union(acc_wide.index).sort_values()
    accr = acc_wide.reindex(all_idx).ffill().reindex(px.index)
    accr = accr.reindex(columns=px.columns)

    return {'px': px, 'accr': accr, 'sectors': sec_map}


def _extract_trades(pos, rets, sec_map):
    trades = []
    for tk in pos.columns:
        w = pos[tk]
        nz = w != 0
        if not nz.any():
            continue
        state = np.sign(w).where(nz, 0.0)
        grp = (state != state.shift(1)).cumsum()
        r = rets[tk]
        for _, sub in state.groupby(grp):
            sv = sub.iloc[0]
            if sv == 0:
                continue
            d = sub.index
            entry, exit_ = d[0], d[-1]
            pv = abs(float(w.loc[entry])) * NOTIONAL
            rr = r.reindex(d).fillna(0.0)
            cum = float((1.0 + rr).prod() - 1.0)
            trades.append({
                "ticker": str(tk),
                "sector": sec_map.get(tk, "Unknown"),
                "entry_date": entry.strftime('%Y-%m-%d'),
                "exit_date": exit_.strftime('%Y-%m-%d'),
                "hold_days": int(len(d)),
                "position_value": float(pv),
                "pnl": float(sv * pv * cum),
            })
    return trades


def signal(panel, **params):
    px = panel['px']
    accr = panel['accr']
    sec_map = panel.get('sectors', {})

    p = {'quantile': 0.20, 'vol_lb': 63, 'max_pos': 40,
         'cost_bps': 8.0, 'min_price': 5.0, 'long_short': True}
    p.update(params)
    quantile = float(p['quantile']); vol_lb = int(p['vol_lb'])
    max_pos = int(p['max_pos']); long_short = bool(p['long_short'])
    cost_bps = float(p['cost_bps']); min_price = float(p['min_price'])

    cols = px.columns
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.9, 1.0)

    vol = rets.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std() * np.sqrt(252)
    vol_c = vol.clip(lower=1e-3)

    eligible = accr.notna() & vol.notna() & (px > min_price)

    # Weekly rebalance dates = last trading day of each ISO week.
    s = pd.Series(px.index, index=px.index)
    rebal_dates = pd.DatetimeIndex(s.groupby(px.index.to_period('W')).last().values)

    per_side_cap = (max_pos // 2) if long_short else max_pos
    long_scale = 0.5 if long_short else 1.0

    rebal_w = {}
    for dt in rebal_dates:
        row = accr.loc[dt].where(eligible.loc[dt])
        valid = row.dropna()
        if len(valid) < 10:
            continue
        cap = len(valid) // 2 if long_short else len(valid)
        n_side = int(np.clip(int(len(valid) * quantile), 1, min(per_side_cap, cap)))
        if n_side < 1:
            continue

        longs = valid.nsmallest(n_side).index      # low accruals -> long
        w = pd.Series(0.0, index=cols)
        lv = 1.0 / vol_c.loc[dt, longs]
        w.loc[longs] = (lv / lv.sum()) * long_scale
        if long_short:
            shorts = valid.nlargest(n_side).index   # high accruals -> short
            sv = 1.0 / vol_c.loc[dt, shorts]
            w.loc[shorts] = -(sv / sv.sum()) * 0.5
        rebal_w[dt] = w

    if not rebal_w:
        empty = pd.Series(dtype=float, name="accruals_eq_smallcap")
        return empty, []

    W = pd.DataFrame(rebal_w).T.reindex(px.index).ffill().fillna(0.0)

    # Lag positions 1 day (signal known at rebal close, traded next day) -> no look-ahead.
    pos = W.shift(1).fillna(0.0)
    gross = (pos * rets).sum(axis=1)
    turnover = (pos - pos.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost)
    net.name = "accruals_eq_smallcap"

    trades = _extract_trades(pos, rets, sec_map)
    return net, trades


SPEC = StrategySpec(
    id="accruals_earnings_quality_smallcap",
    family="cross_sectional_equity_anomaly",
    title="Accruals (Earnings-Quality) Anomaly — Small-Cap Long/Short",
    markets=["US_EQUITY"],
    data_desc=("Sharadar SEP survivorship-clean closeadj prices (delisted incl, split/div-adj) "
               "+ SF1 ARQ fundamentals (netinc, ncfo, assets) for ~1500 small-cap domestic "
               "common stocks; accruals=(NI-CFO)/assets, datekey as-of (no look-ahead)."),
    pre_registration=(
        "Sloan (1996) accruals anomaly. Accruals = (net income - operating cash flow) / total "
        "assets measures earnings quality: high accruals => earnings not backed by cash => "
        "subsequent UNDERPERFORMANCE; low accruals => OUTPERFORMANCE. Predict a positive "
        "net-of-cost Sharpe for a dollar-neutral, inverse-vol, top/bottom-quintile small-cap "
        "long/short book (long low accruals, short high accruals), weekly rebalanced with 1-day "
        "signal lag and 8bps/turnover costs, over the pre-2022 search window. Tested standalone "
        "(no trend hedge). Expectation is decay post-publication; holdout (2022+) persistence is "
        "the binding uncertainty. Falsified if search Sharpe <= 0 or DSR fails effective-N."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "q15": {"quantile": 0.15},
        "q25": {"quantile": 0.25},
        "vol126": {"vol_lb": 126},
        "long_only": {"long_short": False},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)