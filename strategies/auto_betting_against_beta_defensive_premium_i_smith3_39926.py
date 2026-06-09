from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np, pandas as pd

# Sharadar sector labels (variants included so a wrong label just drops, never crashes)
_SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Energy", "Basic Materials",
    "Real Estate", "Realestate", "Utilities", "Communication Services",
]


def _build_universe(top_n_per_sector=120):
    """Bounded, liquid, survivorship-clean universe + ticker->sector map."""
    smap = {}
    tickers = []
    for s in _SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             marketcap="Mid", include_delisted=True,
                             top_n=top_n_per_sector)
        except Exception:
            ts = []
        for t in ts:
            if t not in smap:
                smap[t] = ("Real Estate" if s == "Realestate" else s)
                tickers.append(t)
    return sorted(set(tickers)), smap


def _build_sector_map(tickers):
    """Fallback mapping if panel.attrs got stripped by the harness."""
    tset = set(tickers)
    smap = {}
    for s in _SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             include_delisted=True)
        except Exception:
            ts = []
        for t in ts:
            if t in tset and t not in smap:
                smap[t] = ("Real Estate" if s == "Realestate" else s)
    return smap


def load_data() -> pd.DataFrame:
    tickers, smap = _build_universe(top_n_per_sector=120)
    panel = sep_panel(tickers, start="2004-01-01", field="closeadj")
    panel = panel.sort_index()
    panel.attrs["sector_map"] = smap
    return panel


def signal(panel, **params):
    lookback   = int(params.get("lookback", 252))    # beta estimation window
    vol_lb     = int(params.get("vol_lb", 63))       # inverse-vol window
    per_sector = int(params.get("per_sector", 8))    # low-beta names per sector
    cost_bps   = float(params.get("cost_bps", 8.0))

    panel = panel.sort_index()
    rets = panel.pct_change()

    # sector map (attrs first, reconstruct if missing)
    present = list(panel.columns)
    sector_map = {t: s for t, s in dict(panel.attrs.get("sector_map", {})).items()
                  if t in set(present)}
    if len(sector_map) < 0.5 * len(present):
        sector_map = _build_sector_map(present)
    sector_to_tickers = {}
    for t in present:
        sector_to_tickers.setdefault(sector_map.get(t, "Other"), []).append(t)

    # --- rolling beta vs equal-weight cross-sectional market (fully vectorized) ---
    mkt = rets.mean(axis=1)
    mp = max(63, lookback // 2)
    mean_m  = mkt.rolling(lookback, min_periods=mp).mean()
    var_m   = mkt.rolling(lookback, min_periods=mp).var()
    mean_i  = rets.rolling(lookback, min_periods=mp).mean()
    mean_im = rets.mul(mkt, axis=0).rolling(lookback, min_periods=mp).mean()
    cov = mean_im.sub(mean_i.mul(mean_m, axis=0))
    beta = cov.div(var_m, axis=0)

    vol = rets.rolling(vol_lb, min_periods=max(21, vol_lb // 2)).std()

    # --- weekly rebalance dates (first trading day of each ISO week) ---
    idx = panel.index
    wk = pd.Series(idx.to_period("W"), index=idx)
    rebal_dates = idx[wk.ne(wk.shift()).values]

    # --- sector-neutral low-beta selection, inverse-vol weighted ---
    rows = {}
    for d in rebal_dates:
        b = beta.loc[d]
        v = vol.loc[d]
        sel = []
        for s, members in sector_to_tickers.items():
            bm = b[members].dropna()
            vm = v[members].dropna()
            common = [t for t in bm.index.intersection(vm.index) if v[t] > 0]
            if len(common) < 4:
                continue
            k = min(per_sector, len(common))
            sel.extend(b[common].nsmallest(k).index.tolist())
        if not sel:
            continue
        iv = 1.0 / v[sel]
        w = (iv / iv.sum())
        srow = pd.Series(0.0, index=panel.columns)
        srow[sel] = w.reindex(sel).values
        rows[d] = srow

    target = pd.DataFrame(rows).T.sort_index()
    target = target.reindex(columns=panel.columns).fillna(0.0)
    held = target.reindex(panel.index, method="ffill").fillna(0.0)
    held_lag = held.shift(1).fillna(0.0)              # execute next day -> no look-ahead

    # --- net returns ---
    gross = (held_lag * rets).sum(axis=1, skipna=True)
    turnover = held_lag.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 1e4)
    net = (gross - costs)

    # trim warm-up (pre-first-position) zeros
    live = held_lag.abs().sum(axis=1).gt(0)
    start_mask = live.cumsum().gt(0)
    net = net[start_mask.values]
    net.name = "bab_defensive"

    # --- trades: one per held position run ---
    NOTIONAL = 100000.0
    hl = held_lag.loc[net.index]
    rr = rets.reindex(hl.index)
    dates = list(hl.index)
    n = len(dates)
    trades = []
    for t in hl.columns:
        w = hl[t].values
        if not (w > 1e-9).any():
            continue
        r = np.nan_to_num(rr[t].values, nan=0.0)
        sec = sector_map.get(t, "Other")
        active = w > 1e-9
        i = 0
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            cum_pnl = 0.0
            wsum = 0.0
            while j < n and active[j]:
                cum_pnl += w[j] * r[j]
                wsum += w[j]
                j += 1
            entry, exit_ = dates[i], dates[j - 1]
            ndays = j - i
            trades.append({
                "ticker": t,
                "sector": sec,
                "entry_date": entry.strftime("%Y-%m-%d"),
                "exit_date": exit_.strftime("%Y-%m-%d"),
                "hold_days": int(max((exit_ - entry).days, ndays)),
                "position_value": float((wsum / ndays) * NOTIONAL),
                "pnl": float(cum_pnl * NOTIONAL),
            })
            i = j

    return net, trades


SPEC = StrategySpec(
    id="betting_against_beta_defensive_premium",
    family="defensive",
    title="Betting-Against-Beta / Low-Beta Defensive Premium (sector-neutral, long-only)",
    markets=["US_EQUITY"],
    data_desc=("Sharadar SEP survivorship-clean US equity daily split/div-adjusted closes; "
               "mid-cap liquid universe (~120 names/sector across 11 sectors, delisted included). "
               "Equal-weight cross-sectional market proxy for beta."),
    pre_registration=(
        "HYPOTHESIS: the defensive/low-beta premium (AQR Betting-Against-Beta) — low-beta "
        "stocks earn higher RISK-ADJUSTED returns than high-beta — survives survivorship-clean "
        "data and realistic costs at retail scale. MECHANISM: leverage-constrained investors "
        "overpay for high-beta; low-beta is structurally under-owned. "
        "DESIGN (frozen): rolling 252d beta vs equal-weight market; SECTOR-NEUTRAL select the "
        "bottom-beta names within EACH sector (forces cross-sector spread, removes the trivial "
        "utilities/staples sector tilt); inverse-vol weights; weekly rebalance; signals lagged 1 "
        "day; 8bps turnover cost. Tested STANDALONE (no reflexive trend pairing). "
        "PREDICTIONS: positive net Sharpe in-sample AND in the 2022-01-01+ holdout; DSR>0 after "
        "the declared grid; drawdown materially below cap-weighted equity beta. "
        "KILL: holdout Sharpe<=0 or net premium indistinguishable from market beta => null, do "
        "not deploy. Only consider a SMALL trend tail-overlay AFTER standalone passes and ONLY "
        "if it cuts the tail without diluting Sharpe."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 252, "vol_lb": 63, "per_sector": 8, "cost_bps": 8.0},
    grid={
        "default": {},
        "beta_lb_126": {"lookback": 126},
        "per_sector_5": {"per_sector": 5},
        "per_sector_12": {"per_sector": 12},
        "vol_lb_126": {"vol_lb": 126},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=90,
)