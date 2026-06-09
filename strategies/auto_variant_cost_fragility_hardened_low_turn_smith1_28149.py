# Strategy: Small/Mid-cap Book-to-Market Value — low-turnover, cost-hardened factor book
# No external side effects: builds returns + trades only. OWNED data (Sharadar SEP + SF1).

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, inv_vol_position
import numpy as np
import pandas as pd

# Sharadar (Morningstar-style) sector labels — used to map tickers -> sector for trade spread.
SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive", "Energy",
    "Basic Materials", "Real Estate", "Utilities",
]

START = "2006-01-01"
COST_BPS = 0.0008          # ~8 bps per unit of one-way turnover
NOTIONAL = 1_000_000.0     # book notional used only to denominate trades


def load_data() -> pd.DataFrame:
    """Survivorship-clean small-cap US panel (closeadj) + as-of book-value-per-share.

    Value (book-to-market) is a cross-sectional anomaly that lives in SMALL/illiquid
    names, so we bound the universe to the ~1500 most-liquid small-caps (CPCV-safe).
    The B/M score panel and a ticker->sector map ride along in panel.attrs so signal()
    never recomputes fundamentals across grid variants.
    """
    tickers = us_universe(
        sector=None,
        category="Domestic Common Stock",
        marketcap="Small",
        include_delisted=True,   # survivorship-clean
        top_n=1500,
    )

    px = sep_panel(tickers, START, field="closeadj")   # dates x tickers, split+div adj
    cols = list(px.columns)

    # --- fundamentals: book value per share, dated by FILING date (no look-ahead) ---
    f = sf1(cols, ["bvps"], dimension="ARQ")
    if not isinstance(f, pd.DataFrame):
        f = pd.DataFrame(f)
    if "ticker" not in f.columns or "datekey" not in f.columns:
        f = f.reset_index()
    f = f.copy()
    f["datekey"] = pd.to_datetime(f["datekey"])
    f = f.dropna(subset=["datekey", "bvps"])

    bvps = f.pivot_table(index="datekey", columns="ticker", values="bvps", aggfunc="last").sort_index()
    # As-of join: carry last KNOWN filing forward onto each trading day.
    full = px.index.union(bvps.index)
    bvps = bvps.reindex(full).ffill().reindex(px.index).reindex(columns=cols)

    bm = bvps / px.replace(0.0, np.nan)   # book-to-market; high = cheap = value
    bm = bm.where(bm > 0)                  # drop negative book value

    # --- ticker -> sector map for trade-spread sanity ---
    secmap = {}
    tset = set(cols)
    for s in SECTORS:
        try:
            members = us_universe(sector=s, category="Domestic Common Stock", include_delisted=True)
        except Exception:
            members = []
        for t in members:
            if t in tset:
                secmap[t] = s

    px.attrs["bm"] = bm
    px.attrs["secmap"] = secmap
    return px


def _make_trades(positions: pd.DataFrame, rets: pd.DataFrame, secmap: dict) -> list:
    """One trade per contiguous held-position run (factor-book convention)."""
    trades = []
    for tk in positions.columns:
        w = positions[tk]
        active = w.fillna(0.0).abs() > 1e-9
        if not active.any():
            continue
        run_id = (active != active.shift(fill_value=False)).cumsum()
        r = rets[tk] if tk in rets.columns else None
        for _, grp in active.groupby(run_id):
            idx = grp[grp].index
            if len(idx) == 0:
                continue
            ww = w.reindex(idx).fillna(0.0)
            avg_w = float(ww.mean())
            if abs(avg_w) < 1e-12:
                continue
            if r is not None:
                rr = r.reindex(idx).fillna(0.0)
                pnl = float((ww.values * rr.values).sum() * NOTIONAL)
            else:
                pnl = 0.0
            trades.append({
                "ticker": str(tk),
                "sector": secmap.get(tk, "Unknown"),
                "entry_date": pd.Timestamp(idx[0]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(idx[-1]).strftime("%Y-%m-%d"),
                "hold_days": int(len(idx)),
                "position_value": float(abs(avg_w) * NOTIONAL),
                "pnl": pnl,
            })
    return trades


def signal(panel, n_names=150, vol_lb=63, target_vol=0.10, **params):
    """Long-only top-N book-to-market book, inverse-vol sized, weekly rebalance, 1-day lag.

    Turnover is naturally low (fundamentals update quarterly, B/M ranks are sticky),
    which is what makes this leg cost-robust.
    """
    px = panel
    bm = panel.attrs["bm"]
    secmap = panel.attrs.get("secmap", {})

    rets = px.pct_change(fill_method=None)

    # only rank names that are tradeable (valid price today and yesterday)
    tradeable = px.notna() & px.shift(1).notna()
    score = bm.where(tradeable)

    # pick the top-N highest book-to-market names each day (sampled weekly downstream)
    ranks = score.rank(axis=1, ascending=False, method="first")  # 1 = cheapest
    sel = (ranks <= int(n_names)).astype(float)
    sel = sel.where(score.notna())
    sel = sel.replace(0.0, np.nan)  # NaN where not selected

    # inverse-vol sized, weekly-held, already 1-day lagged by the adapter
    positions = inv_vol_position(
        sel, rets,
        target_vol=target_vol,
        vol_lb=int(vol_lb),
        max_pos=int(n_names),
        rebalance="W",
    )
    positions = positions.reindex(index=rets.index, columns=rets.columns)

    gross = (positions.fillna(0.0) * rets.fillna(0.0)).sum(axis=1)
    turnover = positions.fillna(0.0).diff().abs().sum(axis=1)
    cost = turnover * COST_BPS
    daily = (gross - cost)

    # trim leading flat (pre-position) region, then drop NaNs
    has_pos = positions.fillna(0.0).abs().sum(axis=1) > 0
    if has_pos.any():
        daily = daily.loc[has_pos.idxmax():]
    daily = daily.dropna()
    daily.name = "value_bm_smallcap_lowturn"

    trades = _make_trades(positions, rets, secmap)
    return daily, trades


SPEC = StrategySpec(
    id="value_bm_smallcap_lowturn",
    family="value",
    title="Small-cap Book-to-Market Value (low-turnover, cost-hardened)",
    markets=["us_equity"],
    data_desc=(
        "Sharadar SEP closeadj (split/div-adjusted, delisted incl -> survivorship-clean) for the "
        "~1500 most-liquid US small-cap common stocks, plus SF1 ARQ bvps joined as-of FILING date "
        "(datekey) to avoid look-ahead. Book-to-market = bvps / price."
    ),
    pre_registration=(
        "Book-to-market value premium tested where it is least arbitraged (small caps). "
        "Long-only top-N high-B/M names, inverse-vol weighted, weekly rebalance with naturally "
        "low turnover (quarterly fundamentals + sticky ranks) -> robust to 8bps costs. "
        "Value is a textbook UNIVERSAL premium, so a stage-1 pass must GENERALISE to untouched "
        "large-cap, alternative small slices, and sector-neutral universes (else it is an overfit "
        "outlier). Signals lagged 1 day; standalone leg (no reflexive trend pairing)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "concentrated": {"n_names": 75},
        "broad_book": {"n_names": 250},
        "slow_vol": {"vol_lb": 126},
        "fast_vol": {"vol_lb": 21},
    },
    scope="broad",
    generalization_universes=["large", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=25,
)