# ============================================================================
# Mid-Cap Value (Book-to-Market) + 12-1 Momentum 50/50 Composite Factor Book
# ----------------------------------------------------------------------------
# THESIS:
#   Value (HML) and momentum (WML) are the two most robust, widely-replicated
#   and *negatively-correlated* cross-sectional equity premia. An equal-weight
#   composite of the two delivers a smoother, higher risk-adjusted return than
#   either leg standalone. We test in US MID-CAPS — premia are stronger / less
#   arbitraged than in mega-caps, but the names are liquid enough to trade
#   (unlike micro-caps where slippage eats the edge).
#
#     - Value    = point-in-time Book-to-Market = bvps(as-of filing datekey) / px
#     - Momentum = 12-1 total return            = px.shift(21)/px.shift(252) - 1
#   Each leg is winsorised 5/95 then cross-sectionally z-scored daily; the two
#   z-scores are blended 50/50. We go long the TOP TERCILE of the composite,
#   inverse-vol weighted, weekly rebalanced with a hysteresis band (a held name
#   only drops when it falls out of the top half) to suppress turnover. Signals
#   are lagged 1 day; ~8 bps cost on traded notional.
#
#   scope='broad': this is a universal premium theory, so a stage-1 pass MUST
#   later generalise to untouched slices (large / small / sectors) — otherwise
#   it is an overfit outlier, not a premium.
# ============================================================================

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np
import pandas as pd

START = "2004-01-01"

# Sharadar sector labels (used to build the universe AND the per-name sector map
# that the trade ledger needs for the deployment-sanity diversification check).
SECTORS = [
    "Healthcare", "Technology", "Financial Services", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Energy", "Basic Materials",
    "Communication Services", "Utilities", "Real Estate",
]

# Populated by load_data(); read by signal() (robust to the harness slicing the
# panel, which can drop DataFrame.attrs).
_SECTOR_MAP = {}

DEFAULTS = dict(
    top_q=1.0 / 3.0,    # enter: composite rank in top tercile
    exit_q=0.5,         # hysteresis: hold while still in top half
    val_w=0.5,          # 50/50 value/momentum blend
    vol_lb=63,          # inverse-vol lookback (~quarter)
    max_pos=50,         # cap held names
    cost_bps=8.0,       # round-trip-ish cost on traded notional
    rebal_step=5,       # weekly (~5 trading days)
)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Survivorship-clean mid-cap price panel + as-of book value per share.

    Returns a wide DataFrame with MultiIndex columns: level0 in {'px','bvps'},
    level1 = ticker. bvps is the point-in-time filed book value (datekey as-of)
    forward-filled to the daily price grid (no look-ahead).
    """
    global _SECTOR_MAP

    # Build the bounded mid-cap universe sector-by-sector so we get the
    # ticker->sector map for free (delisted INCLUDED -> survivorship-clean).
    smap = {}
    tickers = []
    for sec in SECTORS:
        try:
            ts = us_universe(
                sector=sec,
                category="Domestic Common Stock",
                marketcap="Mid",
                include_delisted=True,
                top_n=150,
            )
        except Exception:
            ts = []
        for t in ts:
            if t not in smap:
                smap[t] = sec
                tickers.append(t)
    tickers = tickers[:1500]  # keep CPCV tractable (~few-hundred to ~1500)
    if not tickers:
        raise RuntimeError("Empty mid-cap universe.")

    # Survivorship-clean, split+div-adjusted daily closes from OWNED Sharadar SEP.
    px = sep_panel(tickers, START, field="closeadj").sort_index()
    px = px.replace([np.inf, -np.inf], np.nan)
    avail = [t for t in tickers if t in px.columns]
    px = px[avail]

    # Fundamentals: book value per share, as-of FILING date (datekey) -> no peek.
    fund = sf1(avail, ["bvps"], dimension="ARQ")
    f = fund.copy()
    if f.index.name is not None or isinstance(f.index, pd.MultiIndex):
        f = f.reset_index()
    lc = {str(c).lower(): c for c in f.columns}
    tcol = lc.get("ticker", "ticker")
    dcol = lc.get("datekey", "datekey")
    bcol = lc.get("bvps", "bvps")
    f = f[[tcol, dcol, bcol]].dropna(subset=[dcol, bcol])
    f[dcol] = pd.to_datetime(f[dcol])
    bvps_wide = (
        f.sort_values(dcol)
        .pivot_table(index=dcol, columns=tcol, values=bcol, aggfunc="last")
    )
    # As-of: forward-fill the last FILED bvps onto the daily price grid.
    bvps_daily = bvps_wide.reindex(px.index, method="ffill")

    common = [t for t in px.columns if t in bvps_daily.columns]
    px = px[common]
    bvps_daily = bvps_daily[common]
    _SECTOR_MAP = {t: smap.get(t, "Unknown") for t in common}

    panel = pd.concat({"px": px, "bvps": bvps_daily}, axis=1)
    panel.attrs["sectors"] = dict(_SECTOR_MAP)  # backup channel
    return panel


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def _winsor_zscore(df: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    """Row-wise (cross-sectional, per day) winsorise then z-score."""
    lo_q = df.quantile(lo, axis=1)
    hi_q = df.quantile(hi, axis=1)
    clipped = df.clip(lower=lo_q, upper=hi_q, axis=0)
    mu = clipped.mean(axis=1)
    sd = clipped.std(axis=1).replace(0, np.nan)
    return clipped.sub(mu, axis=0).div(sd, axis=0)


# ----------------------------------------------------------------------------
# SIGNAL
# ----------------------------------------------------------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    top_q = float(p["top_q"])
    exit_q = float(p["exit_q"])
    val_w = float(p["val_w"])
    vol_lb = int(p["vol_lb"])
    max_pos = int(p["max_pos"])
    cost_bps = float(p["cost_bps"])
    step = int(p["rebal_step"])

    name = "midcap_value_momentum"

    px = panel["px"].copy().replace([np.inf, -np.inf], np.nan)
    bvps = panel["bvps"].copy().replace([np.inf, -np.inf], np.nan)
    dates = px.index
    cols = px.columns

    rets = px.pct_change()

    # --- factor signals (long-only ranks) ---
    btm = (bvps / px).replace([np.inf, -np.inf], np.nan)        # value: high = cheap
    mom = (px.shift(21) / px.shift(252) - 1.0)                  # 12-1 momentum
    zv = _winsor_zscore(btm)
    zm = _winsor_zscore(mom)
    # Genuine composite: require BOTH legs present (don't impute a missing leg).
    composite = (val_w * zv + (1.0 - val_w) * zm)
    composite = composite.where(zv.notna() & zm.notna())

    vol = rets.rolling(vol_lb).std()

    # --- weekly rebalance with hysteresis ---
    rebal_dates = dates[::step]
    held = set()
    rebal_w = {}
    for dt in rebal_dates:
        c = composite.loc[dt].dropna()
        if len(c) < 20:
            continue  # too few names -> ffill carries prior book forward
        ranks = c.rank(pct=True)
        enter = set(ranks.index[ranks >= (1.0 - top_q)])       # top tercile
        keep = set(ranks.index[ranks >= (1.0 - exit_q)])       # stay in top half
        new_held = (held & keep) | enter
        if len(new_held) > max_pos:                            # trim by score
            sc = c.reindex(list(new_held)).sort_values(ascending=False)
            new_held = set(sc.index[:max_pos])
        held = new_held
        if not held:
            continue

        # inverse-vol weights, normalised long-only
        v = vol.loc[dt, list(held)]
        iv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
        if iv.sum() > 0:
            w = iv / iv.sum()
        else:  # fallback: equal weight
            w = pd.Series(1.0 / len(held), index=list(held))
        wv = pd.Series(0.0, index=cols)
        wv.loc[w.index] = w.values
        rebal_w[dt] = wv

    if not rebal_w:
        empty = pd.Series(0.0, index=dates, name=name)
        return empty, []

    weights = pd.DataFrame(rebal_w).T.reindex(columns=cols)
    weights = weights.reindex(dates).ffill().fillna(0.0)

    # --- lag 1 day, returns, costs on turnover ---
    w_lag = weights.shift(1).fillna(0.0)                       # held position each day
    gross = (w_lag * rets).sum(axis=1)
    turn = (w_lag - w_lag.shift(1)).abs().sum(axis=1)
    cost = turn * (cost_bps / 1e4)
    net = (gross - cost).fillna(0.0)

    # trim leading zero-position prefix
    active_mask = w_lag.abs().sum(axis=1) > 0
    if active_mask.any():
        first = active_mask.idxmax()
        net = net.loc[first:]
    net.name = name

    # --- trade ledger: one trade per held position run ---
    sectors = panel.attrs.get("sectors") or _SECTOR_MAP
    NOTIONAL = 1_000_000.0
    trades = []
    held_cols = [t for t in cols if (w_lag[t].abs() > 1e-8).any()]
    for tk in held_cols:
        w_arr = w_lag[tk].values
        r_arr = rets[tk].values
        act = np.abs(w_arr) > 1e-8
        n = len(act)
        i = 0
        while i < n:
            if not act[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and act[j + 1]:
                j += 1
            seg_w = w_arr[i:j + 1]
            seg_r = np.nan_to_num(r_arr[i:j + 1])
            avg_w = float(np.mean(seg_w))
            contrib = float(np.sum(seg_w * seg_r))           # portfolio-return contribution
            trades.append({
                "ticker": tk,
                "sector": sectors.get(tk, "Unknown"),
                "entry_date": dates[i].strftime("%Y-%m-%d"),
                "exit_date": dates[j].strftime("%Y-%m-%d"),
                "hold_days": int(j - i + 1),
                "position_value": float(avg_w * NOTIONAL),
                "pnl": float(contrib * NOTIONAL),
            })
            i = j + 1

    return net, trades


# ----------------------------------------------------------------------------
# SPEC
# ----------------------------------------------------------------------------
SPEC = StrategySpec(
    id="midcap_value_momentum_composite",
    family="cross_sectional_equity_factor",
    title="Mid-Cap Value (B/M) + 12-1 Momentum 50/50 Composite",
    markets=["us_equities"],
    data_desc=(
        "Survivorship-clean Sharadar SEP daily closeadj for ~1500 US mid-cap "
        "common stocks (delisted included), plus point-in-time book value per "
        "share (sf1 bvps, as-of filing datekey). Value = bvps/px, Momentum = "
        "12-1 total return; winsorised 5/95, daily cross-sectional z-scores, "
        "blended 50/50; long top tercile, inverse-vol weighted, weekly "
        "rebalance with top-half hysteresis, signals lagged 1 day, ~8bps cost."
    ),
    pre_registration=(
        "HYPOTHESIS: An equal-weight composite of two independent, negatively "
        "correlated premia (value B/M + 12-1 momentum) in liquid US mid-caps "
        "earns a positive risk-adjusted long-only return net of ~8bps costs, "
        "smoother than either leg standalone. PRE-DECLARED: top-tercile entry, "
        "top-half hysteresis exit, max 50 names, inverse-vol sizing, weekly "
        "rebalance, 1-day lag. As a BROAD premium it must generalise to "
        "untouched large/small/sector slices and survive the 2022+ holdout."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "value_tilt": {"val_w": 0.7},
        "momentum_tilt": {"val_w": 0.3},
        "tight_decile": {"top_q": 0.2},
        "wide_top": {"top_q": 0.4},
    },
    scope="broad",
    generalization_universes=["large", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)