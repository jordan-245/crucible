from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np, pandas as pd

# =============================================================================
# Cross-sectional VALUE + MOMENTUM complementary equity book (mid-cap universe)
#
# Economic thesis (Asness/Moskowitz/Pedersen "Value & Momentum Everywhere"):
#   - MOMENTUM: 12-1 month relative strength (recent winners keep winning).
#   - VALUE   : long-horizon reversal proxy (-1 * the t-60m..t-12m return) -
#               long-term laggards are "cheap" and tend to mean-revert. This is
#               the standard price-based value proxy used when one wants a single
#               universal definition (the AMP cross-asset value measure).
#   These two premia are individually positive but NEGATIVELY correlated, so the
#   50/50 z-score blend should deliver a higher Sharpe / shallower drawdown than
#   either standalone leg. The grid below tests each leg standalone (mom_only /
#   val_only) so the harness sees the honest search burden and can confirm the
#   combination is not just one leg in disguise.
#
# Both are UNIVERSAL premia -> scope='broad'; a stage-1 pass MUST generalise to
# the untouched large / small / sector slices, else it is an overfit outlier.
#
# Long-only top-N book, inverse-vol sized, weekly rebalance, signals lagged 1
# trading day, ~8bps cost on traded notional. Holdout from 2022-01-01.
# =============================================================================

STRAT_ID = "value_momentum_complementary_xs_v1"
START = "2000-01-01"
CAPITAL = 1_000_000.0

_DEFAULTS = {
    "mom_lb": 252,    # momentum: return from t-252 ...
    "mom_skip": 21,   # ... to t-21 (12-1 month)
    "val_lb": 1260,   # value/LT-reversal: return from t-1260 (5y) ...
    "val_skip": 252,  # ... to t-252 (1y); signal = -that return
    "w_mom": 0.5,
    "w_val": 0.5,
    "top_n": 50,
    "vol_lb": 63,     # ~3m trailing vol for inverse-vol sizing
    "cost_bps": 8.0,
    "min_names": 100,
}

# sector lookup is populated by load_data() (module-level, no external side effect)
_SECTOR_MAP = {}


def _build_sector_map(tickers):
    """Map universe tickers -> Sharadar sector via the OWNED TICKERS table."""
    global _SECTOR_MAP
    sectors = [
        "Basic Materials", "Communication Services", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
        "Industrials", "Real Estate", "Technology", "Utilities",
    ]
    tset = set(tickers)
    m = {}
    for s in sectors:
        try:
            lst = us_universe(sector=s, category="Domestic Common Stock",
                              include_delisted=True)
        except Exception:
            continue
        for t in lst:
            if t in tset:
                m[t] = s
    _SECTOR_MAP = m


def load_data() -> pd.DataFrame:
    # Bounded, survivorship-clean universe. Cross-sectional value/momentum live in
    # smaller, less-arbitraged names -> mid-cap, ~1500 most-liquid (delisted incl).
    tickers = us_universe(category="Domestic Common Stock", marketcap="Mid",
                          include_delisted=True, top_n=1500)
    _build_sector_map(tickers)
    # split+div adjusted closes from OWNED Sharadar SEP (delisted included)
    px = sep_panel(tickers, start=START, field="closeadj")
    px = px.sort_index()
    px = px.dropna(axis=1, how="all")
    return px


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def signal(panel, **params):
    p = {**_DEFAULTS, **params}
    px = panel.sort_index()

    if not _SECTOR_MAP:
        _build_sector_map(list(px.columns))

    rets = px.pct_change()
    rets_f = rets.fillna(0.0)

    # ---- raw cross-sectional signals (full panels, no look-ahead at scoring) ---
    mom_panel = px.shift(p["mom_skip"]) / px.shift(p["mom_lb"]) - 1.0           # winners
    val_panel = -(px.shift(p["val_skip"]) / px.shift(p["val_lb"]) - 1.0)        # LT losers = "cheap"
    vol_panel = rets.rolling(p["vol_lb"]).std()

    # ---- weekly rebalance dates = last trading day of each ISO week -----------
    iso = px.index.isocalendar()
    yw = iso["year"].astype(int).values * 100 + iso["week"].astype(int).values
    tmp = pd.DataFrame({"yw": yw, "date": px.index})
    rebal_dates = pd.DatetimeIndex(sorted(tmp.groupby("yw")["date"].max().values))

    # ---- build target weight vector (0-filled) on each rebalance date ---------
    cols = px.columns
    weights = {}
    for d in rebal_dates:
        m = mom_panel.loc[d]
        v = val_panel.loc[d]
        vol = vol_panel.loc[d]
        valid = m.notna() & v.notna() & vol.notna() & (vol > 0)
        if int(valid.sum()) < p["min_names"]:
            continue
        zm = _zscore(m[valid])
        zv = _zscore(v[valid])
        combo = p["w_mom"] * zm + p["w_val"] * zv
        n = int(min(p["top_n"], len(combo)))
        if n <= 0:
            continue
        sel = combo.nlargest(n).index
        iv = 1.0 / vol[sel]            # inverse-vol sizing
        if iv.sum() <= 0:
            continue
        w = pd.Series(0.0, index=cols)
        w.loc[sel] = (iv / iv.sum()).values
        weights[d] = w

    if not weights:
        empty = pd.Series(dtype=float, name=STRAT_ID)
        return empty, []

    W_rebal = pd.DataFrame(weights).T.reindex(columns=cols)
    W_daily = W_rebal.reindex(px.index).ffill().fillna(0.0)

    # lag execution 1 trading day -> no look-ahead
    pos = W_daily.shift(1).fillna(0.0)

    gross = (pos * rets_f).sum(axis=1)
    turnover = (W_daily - W_daily.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover.shift(1).fillna(0.0) * (p["cost_bps"] / 1e4)   # cost on day trades execute
    net = (gross - cost)

    # trim leading flat region
    active = pos.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    net.name = STRAT_ID

    # ---- one trade per held-position RUN (deployment sanity) ------------------
    contrib = pos * rets_f
    trades = []
    for tk in cols:
        ser = pos[tk]
        held = ser > 1e-8
        if not held.any():
            continue
        grp = (held != held.shift()).cumsum()
        for _, sub in ser[held].groupby(grp[held]):
            run_idx = sub.index
            entry = run_idx[0]
            exit_ = run_idx[-1]
            hold_days = int(len(run_idx))
            avg_w = float(ser.loc[run_idx].mean())
            position_value = float(avg_w * CAPITAL)
            pnl = float(contrib[tk].loc[run_idx].sum() * CAPITAL)
            trades.append({
                "ticker": tk,
                "sector": _SECTOR_MAP.get(tk, "Unknown"),
                "entry_date": entry.strftime("%Y-%m-%d"),
                "exit_date": exit_.strftime("%Y-%m-%d"),
                "hold_days": hold_days,
                "position_value": position_value,
                "pnl": pnl,
            })

    return net, trades


SPEC = StrategySpec(
    id=STRAT_ID,
    family="value_momentum",
    title="Cross-sectional Value + Momentum complementary book (mid-cap)",
    markets=["US equities (Sharadar SEP, mid-cap ~1500 most-liquid)"],
    data_desc=("Sharadar SEP survivorship-clean daily split+div-adjusted closes "
               "(delisted included); mid-cap Domestic Common Stock universe, "
               "top 1500 by liquidity, since 2000."),
    pre_registration=(
        "12-1 month momentum (t-252..t-21) and 5y-1y long-term price reversal "
        "'value' (-1 * t-1260..t-252 return), each cross-sectionally z-scored "
        "and 50/50 blended. Long top-50 by combined score, inverse-vol sized, "
        "weekly rebalance, signals lagged 1 trading day, 8bps cost on turnover. "
        "Standalone legs (mom_only, val_only) and top_n variants pre-declared in "
        "grid for honest effective-N. Universal premia -> scope broad; a stage-1 "
        "pass must GENERALISE to untouched large/small/sector slices or it is an "
        "overfit outlier."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "mom_only": {"w_mom": 1.0, "w_val": 0.0},
        "val_only": {"w_mom": 0.0, "w_val": 1.0},
        "top30": {"top_n": 30},
        "top75": {"top_n": 75},
    },
    scope="broad",
    generalization_universes=["large", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)