# Strategy: Defensive low-volatility (Betting-Against-Beta) cross-sectional equity book
# Family: defensive risk premium (the "Defensive/Low-Vol" leg of the classic premia set)
# Standalone test FIRST (no reflexive trend pairing). Survivorship-clean Sharadar SEP.
# No external side effects: pure returns + trades for the Hephaestus rails.

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np
import pandas as pd

START = "2003-06-01"

# Sharadar GICS-style sector labels (best-effort; unmatched -> 'Unknown')
_SECTORS = [
    "Healthcare", "Basic Materials", "Financial Services", "Industrials",
    "Consumer Cyclical", "Technology", "Consumer Defensive", "Energy",
    "Real Estate", "Communication Services", "Utilities",
]

DEFAULTS = dict(lookback=252, n_hold=50, cost_bps=8.0, min_price=5.0)


def _sector_map(tickers):
    """Build ticker -> sector map (best-effort, robust to label mismatch)."""
    want = set(tickers)
    m = {}
    for s in _SECTORS:
        try:
            ss = us_universe(sector=s, category="Domestic Common Stock",
                             include_delisted=True)
        except Exception:
            ss = []
        for t in ss:
            if t in want and t not in m:
                m[t] = s
    for t in tickers:
        m.setdefault(t, "Unknown")
    return m


def load_data() -> pd.DataFrame:
    # Cross-sectional defensive anomaly: test in mid-cap liquid (not the largest,
    # arbitraged names). Bound the universe (top_n) so the CPCV does not OOM.
    tickers = us_universe(category="Domestic Common Stock", marketcap="Mid",
                          top_n=1200, include_delisted=True)
    panel = sep_panel(tickers, start=START, field="closeadj")  # split+div adj, delisted incl
    panel = panel.sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    panel.index = pd.to_datetime(panel.index)
    panel = panel.dropna(how="all", axis=1)
    try:
        panel.attrs["sectors"] = _sector_map(list(panel.columns))
    except Exception:
        panel.attrs["sectors"] = {}
    return panel


def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)
    lookback = int(p["lookback"])
    n_hold = int(p["n_hold"])
    cost_bps = float(p["cost_bps"])
    min_price = float(p["min_price"])

    px = panel.sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px.index = pd.to_datetime(px.index)
    px = px.dropna(how="all", axis=1)
    idx = px.index

    rets = px.pct_change()
    # Trailing realized vol (the defensive signal: prefer the LOWEST-vol names)
    minp = max(20, int(lookback * 0.6))
    vol = rets.rolling(lookback, min_periods=minp).std()

    # Weekly rebalance = last trading day of each ISO week
    pos = pd.Series(np.arange(len(idx)), index=idx)
    grp = idx.to_period("W")
    rebal_pos = pos.groupby(grp).last().values
    rebal_days = idx[rebal_pos]

    # Target weights at each rebalance, inverse-vol within the low-vol selection
    w = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    for d in rebal_days:
        v = vol.loc[d]
        pr = px.loc[d]
        elig = v[(v > 0) & v.notna() & pr.notna() & (pr >= min_price)]
        if len(elig) == 0:
            continue
        sel = elig if len(elig) <= n_hold else elig.nsmallest(n_hold)
        iv = 1.0 / sel
        wd = iv / iv.sum()
        w.loc[d] = 0.0
        w.loc[d, wd.index] = wd.values

    w = w.ffill().fillna(0.0)

    # No look-ahead: positions effective the day AFTER the signal
    w_lag = w.shift(1).fillna(0.0)

    # Portfolio gross return, minus realistic turnover cost (~8bps)
    port = (w_lag * rets).sum(axis=1)
    turnover = w_lag.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net = (port - cost)

    # Trim leading no-position warmup
    exposure = w_lag.abs().sum(axis=1)
    has_pos = exposure[exposure > 0]
    if len(has_pos):
        net = net.loc[has_pos.index.min():]
    net = net.dropna()
    net.name = "auto_defensive_bab_lowvol"

    # ---- Trades: one per contiguous holding run per name (deployment sanity) ----
    smap = panel.attrs.get("sectors") if hasattr(panel, "attrs") else None
    if not smap:
        try:
            smap = _sector_map(list(px.columns))
        except Exception:
            smap = {}

    book = 100000.0
    contrib = w_lag * rets
    held = w_lag > 0.0
    trades = []
    cols_held = w_lag.columns[held.any().values]
    didx = w_lag.index
    for col in cols_held:
        h = held[col].values
        if not h.any():
            continue
        ww = w_lag[col].values
        cc = contrib[col].values
        starts = np.where(h & ~np.r_[False, h[:-1]])[0]
        ends = np.where(h & ~np.r_[h[1:], False])[0]
        for s_i, e_i in zip(starts, ends):
            hold_days = int(e_i - s_i + 1)
            avg_w = float(np.nanmean(ww[s_i:e_i + 1]))
            pnl = float(np.nansum(cc[s_i:e_i + 1]) * book)
            trades.append({
                "ticker": str(col),
                "sector": smap.get(col, "Unknown"),
                "entry_date": didx[s_i].strftime("%Y-%m-%d"),
                "exit_date": didx[e_i].strftime("%Y-%m-%d"),
                "hold_days": hold_days,
                "position_value": float(avg_w * book),
                "pnl": pnl,
            })

    return net, trades


SPEC = StrategySpec(
    id="auto_defensive_bab_lowvol",
    family="defensive",
    title="Defensive low-volatility (betting-against-beta) cross-sectional equity book",
    markets=["US_EQUITY"],
    data_desc=("Sharadar SEP survivorship-clean daily closeadj (split+div adj, delisted "
               "included); mid-cap liquid US Domestic Common Stock (top ~1200 by liquidity), "
               "2003+. Sectors from Sharadar TICKERS."),
    pre_registration=(
        "HYPOTHESIS: the Defensive/Low-Volatility premium (Frazzini-Pedersen BAB, "
        "long low-realized-vol names) is one of the few cross-sectional equity premia with "
        "decades of real-money OOS support. We test the LONG low-vol leg STANDALONE first "
        "(no reflexive trend pairing). SIGNAL: trailing 252d realized vol; weekly rebalance "
        "into the n_hold (50) lowest-vol eligible names (price>=$5), inverse-vol weighted, "
        "fully invested long-only, signals lagged 1 day, 8bps cost on turnover. "
        "Tested in MID-cap (not arbitraged mega-caps) to avoid false nulls. "
        "PASS GATES (rails): positive net Shardar SEP search Sharpe AND positive 2022+ holdout "
        "with the drawdown materially smaller than a cap-weighted equity book; DSR-adjusted for "
        "the declared grid effective-N. Only consider a SMALL trend tail-overlay if standalone "
        "passes and the overlay cuts the tail WITHOUT diluting the standalone Sharpe; "
        "never a reflexive 50/50 (that sank credit-carry to ~0). FAIL => bank as negative knowledge."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "vol_lb_126": {"lookback": 126},
        "vol_lb_63": {"lookback": 63},
        "n_hold_30": {"n_hold": 30},
        "n_hold_100": {"n_hold": 100},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)