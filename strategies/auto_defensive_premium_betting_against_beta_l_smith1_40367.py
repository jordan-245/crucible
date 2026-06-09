"""
Defensive premium — Betting-Against-Beta (BAB), standalone.

Economic thesis (Frazzini-Pedersen / AQR defensive premium): leverage-constrained
investors bid up high-beta stocks, so low-beta names earn higher RISK-ADJUSTED
returns. A beta-neutral book that levers LOW-beta names and de-levers HIGH-beta
names harvests this. We test it STANDALONE (no trend pairing) in mid-cap names
where the anomaly is less arbitraged than in the largest liquid mega-caps.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np, pandas as pd

START = "2004-01-01"

_SECTORS = ['Healthcare', 'Financial Services', 'Technology', 'Consumer Cyclical',
            'Consumer Defensive', 'Industrials', 'Energy', 'Basic Materials',
            'Real Estate', 'Communication Services', 'Utilities']


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """Survivorship-clean mid-cap US panel (Sharadar SEP) + per-ticker sector map."""
    sector_map = {}
    for s in _SECTORS:
        try:
            ts = us_universe(sector=s, category='Domestic Common Stock',
                             marketcap='Mid', include_delisted=True, top_n=140)
        except Exception:
            ts = []
        for t in ts:
            sector_map.setdefault(t, s)

    tickers = list(sector_map.keys())
    panel = sep_panel(tickers, start=START, field='closeadj')
    panel = panel.sort_index()
    panel = panel.loc[:, panel.notna().sum() > 252]   # need >=1yr history
    panel.attrs['sectors'] = sector_map
    return panel


# --------------------------------------------------------------------------- signal
def signal(panel, **params):
    beta_lb    = int(params.get('beta_lb', 252))
    vol_lb     = int(params.get('vol_lb', 63))
    n_side     = int(params.get('n_side', 40))
    target_vol = float(params.get('target_vol', 0.10))
    cost_bps   = float(params.get('cost_bps', 8.0))
    NOTIONAL   = 10000.0

    px   = panel.sort_index()
    idx  = px.index
    rets = px.pct_change()

    # equal-weight market proxy
    mkt = rets.mean(axis=1)

    # rolling beta (consistent population moments)
    mp = max(beta_lb // 2, 60)
    m_i  = rets.rolling(beta_lb, min_periods=mp).mean()
    m_m  = mkt.rolling(beta_lb,  min_periods=mp).mean()
    m_im = rets.mul(mkt, axis=0).rolling(beta_lb, min_periods=mp).mean()
    m_mm = (mkt ** 2).rolling(beta_lb, min_periods=mp).mean()
    cov   = m_im.sub(m_i.mul(m_m, axis=0))
    var_m = (m_mm - m_m ** 2)
    beta  = cov.div(var_m.replace(0.0, np.nan), axis=0)

    # rolling vol for inverse-vol sizing
    vol = rets.rolling(vol_lb, min_periods=max(vol_lb // 2, 20)).std() * np.sqrt(252)

    # weekly rebalance dates = last trading day of each ISO week
    pos = pd.Series(np.arange(len(idx)), index=idx)
    last_pos = pos.groupby(idx.to_period('W')).max().values
    rb_dates = idx[last_pos]

    # build beta-neutral, inverse-vol weighted target weights on rebalance days
    W = pd.DataFrame(index=rb_dates, columns=px.columns, dtype=float)
    for d in rb_dates:
        b = beta.loc[d]
        v = vol.loc[d]
        valid = b.notna() & v.notna() & (v > 0) & (b > -2) & (b < 5)
        b = b[valid]; v = v[valid]
        ns = min(n_side, len(b) // 2)
        if ns < 10:
            continue

        long_names  = b.nsmallest(ns).index
        short_names = b.nlargest(ns).index

        iv = 1.0 / v
        wl = iv[long_names];  wl = wl / wl.sum()
        ws = iv[short_names]; ws = ws / ws.sum()

        # Frazzini-Pedersen beta neutralisation: lever low-beta, de-lever high-beta
        bl = float((wl * b.loc[long_names]).sum())
        bs = float((ws * b.loc[short_names]).sum())
        ls, ss = (1.0 / bl, 1.0 / bs) if (bl > 0 and bs > 0) else (1.0, 1.0)

        w = pd.Series(0.0, index=px.columns)
        w.loc[long_names]  =  wl.values * ls
        w.loc[short_names] = -ws.values * ss
        g = w.abs().sum()
        if g > 0:
            W.loc[d] = (w / g).values   # gross-1 book; vol-target scales it next

    # forward-fill to daily, then LAG 1 day (no look-ahead)
    W = W.reindex(idx).ffill().fillna(0.0)
    W_lag = W.shift(1).fillna(0.0)

    # portfolio-level vol targeting (lagged trailing estimate -> no look-ahead)
    gross_ret = (W_lag * rets).sum(axis=1)
    pv = gross_ret.rolling(63, min_periods=20).std() * np.sqrt(252)
    scale = (target_vol / pv).shift(1).replace([np.inf, -np.inf], np.nan).clip(upper=3.0).fillna(0.0)
    W_eff = W_lag.mul(scale, axis=0).fillna(0.0)

    # net-of-cost daily returns
    gret     = (W_eff * rets).sum(axis=1)
    turnover = W_eff.diff().abs().sum(axis=1).fillna(0.0)
    cost     = turnover * (cost_bps / 1e4)
    daily    = (gret - cost).fillna(0.0)
    daily.name = "bab_defensive"

    # ---- trades: one per contiguous held run (factor book) ----
    sectors = panel.attrs.get('sectors', {})
    n = len(idx)
    trades = []
    held = (W_eff.abs() > 1e-9)
    for col in W_eff.columns:
        h = held[col].values
        if not h.any():
            continue
        d = np.diff(np.concatenate(([0], h.astype(int), [0])))
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0] - 1
        wcol = W_eff[col].values
        rcol = rets[col].values
        for s_i, e_i in zip(starts, ends):
            sl = slice(s_i, e_i + 1)
            pnl = float(np.nansum(wcol[sl] * rcol[sl]) * NOTIONAL)
            pv_ = float(np.nanmean(np.abs(wcol[sl])) * NOTIONAL)
            x_i = min(e_i + 1, n - 1)
            trades.append({
                "ticker":         col,
                "sector":         sectors.get(col, "Unknown"),
                "entry_date":     idx[s_i].strftime("%Y-%m-%d"),
                "exit_date":      idx[x_i].strftime("%Y-%m-%d"),
                "hold_days":      int(e_i - s_i + 1),
                "position_value": pv_,
                "pnl":            pnl,
            })

    return daily, trades


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="defensive_bab",
    family="defensive",
    title="Betting-Against-Beta defensive premium (standalone, mid-cap, beta-neutral)",
    markets=["US equities (Sharadar SEP, mid-cap)"],
    data_desc="Survivorship-clean mid-cap US daily closeadj (Sharadar SEP); "
              "rolling 1y beta vs equal-weight market; rolling 63d vol for inv-vol sizing.",
    pre_registration=(
        "H: leverage-constrained demand makes LOW-beta stocks cheap on a risk-adjusted "
        "basis (Frazzini-Pedersen defensive premium). A beta-neutral book that inverse-vol "
        "weights and levers the lowest-beta tercile against the highest-beta tercile should "
        "earn a positive net-of-cost Sharpe. Tested in MID-CAPS (anomaly less arbitraged "
        "than mega-caps). STANDALONE test — no trend pairing; only consider a tail overlay "
        "later if it cuts DD without diluting standalone Sharpe. Predict modest positive "
        "Sharpe; trend-style crisis convexity NOT expected (this is a calm-premium book)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"beta_lb": 252, "vol_lb": 63, "n_side": 40,
                    "target_vol": 0.10, "cost_bps": 8.0},
    grid={
        "default":      {},
        "beta_lb_126":  {"beta_lb": 126},
        "n_side_25":    {"n_side": 25},
        "n_side_60":    {"n_side": 60},
        "vol_lb_126":   {"vol_lb": 126},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=80,
)