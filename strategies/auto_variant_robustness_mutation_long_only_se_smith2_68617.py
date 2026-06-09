# Small/Mid-Cap VALUE + MOMENTUM composite — long-only top-tercile book.
#
# THESIS (pre-registered, faithfully implemented below):
#   Two of the most-replicated cross-sectional equity premia are VALUE (cheap book-to-market
#   outperforms) and MOMENTUM (12-1 winners outperform). They are weakly/negatively correlated,
#   so an EQUAL-WEIGHT z-score COMBINATION of the two — sorted ONCE on the integrated composite —
#   is more robust than either leg standalone (the diversification across the two premia is the edge).
#   These anomalies live in SMALL/MID-cap, less-arbitraged names, so we test there (survivorship-
#   clean Sharadar SEP, delisted included) behind a $1M dollar-volume TRADABILITY floor so the
#   book is actually executable. Point-in-time fundamentals (ARQ 'datekey' filing-date lag) avoid
#   look-ahead. We hold the top tercile of the composite, using a HYSTERESIS buffer (enter top 30%,
#   exit only when a name drops out of the top 50%) to cut turnover, inverse-vol size within the
#   book, rebalance weekly, lag signals one day, and HARDEN costs (25/50/75 bps grid) because
#   small/mid names are expensive to trade — an honest premium must survive that.
#
# Standalone-first discipline: VALUE-only and MOMENTUM-only are pre-declared grid variants; the
# composite is only meaningful if it does not merely dilute a single strong leg. (No trend hedge
# is bolted on — a ~0-Sharpe overlay would halve a real premium.)

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1

START = "2004-01-01"
_SECTORS = [
    "Healthcare", "Financial Services", "Technology", "Industrials",
    "Consumer Cyclical", "Consumer Defensive", "Energy", "Basic Materials",
    "Real Estate", "Utilities", "Communication Services",
]


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    # Survivorship-clean SMALL/MID liquid universe (delisted included). Bounded ~1500 names.
    uni_small = us_universe(category="Domestic Common Stock", marketcap="Small",
                            include_delisted=True, top_n=1000)
    uni_mid = us_universe(category="Domestic Common Stock", marketcap="Mid",
                          include_delisted=True, top_n=500)
    tickers = sorted(set(uni_small) | set(uni_mid))

    # Split+div adjusted closes + raw volume from OWNED Sharadar SEP.
    px = sep_panel(tickers, START, field="closeadj").sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    try:
        vol = sep_panel(tickers, START, field="volume").reindex_like(px)
    except Exception:
        vol = px * np.nan
    dvol = (px * vol).rolling(20, min_periods=5).mean()  # ~$ dollar-volume (tradability)

    # Point-in-time book value per share -> book-to-market (B/M). Use 'datekey' (filing date).
    btm = _pit_book_to_market(tickers, px)

    cols = px.columns
    panel = pd.concat(
        {"px": px, "dvol": dvol.reindex(columns=cols), "btm": btm.reindex(columns=cols)},
        axis=1,
    )

    # Sector map for trade attribution / deployment-sanity (spread across sectors).
    sector_map = {}
    for s in _SECTORS:
        try:
            for t in us_universe(sector=s, category="Domestic Common Stock",
                                 include_delisted=True, top_n=10000):
                sector_map[t] = s
        except Exception:
            pass
    panel.attrs["sector_map"] = sector_map
    return panel


def _pit_book_to_market(tickers, px) -> pd.DataFrame:
    try:
        fund = sf1(tickers, ["bvps"], dimension="ARQ")
    except Exception:
        return px * np.nan
    f = fund.copy()
    if "datekey" not in f.columns:
        f = f.reset_index()
    if "datekey" not in f.columns or "ticker" not in f.columns or "bvps" not in f.columns:
        return px * np.nan
    f["datekey"] = pd.to_datetime(f["datekey"], errors="coerce")
    f = f.dropna(subset=["datekey", "bvps"])
    wide = (f.pivot_table(index="datekey", columns="ticker", values="bvps", aggfunc="last")
              .sort_index())
    bvps_daily = wide.reindex(px.index, method="ffill")  # forward-fill from filing date (PIT)
    bvps_daily = bvps_daily.reindex(columns=px.columns)
    btm = bvps_daily / px.replace(0.0, np.nan)
    return btm.where(btm > 0)  # negative book -> drop from value sort


# --------------------------------------------------------------------------- helpers
def _xs_z(df: pd.DataFrame, clip: float) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


# ---------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)

    px = panel["px"]
    dvol = panel["dvol"]
    btm = panel["btm"]
    sector_map = panel.attrs.get("sector_map", {})

    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)
    # 12-1 momentum (skip most recent month) and trailing vol for inverse-vol sizing.
    mom = px.shift(p["mom_skip"]) / px.shift(p["mom_lb"]) - 1.0
    vol = rets.rolling(p["vol_lb"], min_periods=20).std()

    # Weekly rebalance dates = last trading day of each calendar week.
    wk = px.index.to_series().dt.to_period("W")
    rb_dates = px.index[~wk.duplicated(keep="last").values]

    # Eligibility on rebalance dates: tradable ($ vol floor) + valid price.
    px_rb = px.reindex(rb_dates)
    elig = (dvol.reindex(rb_dates) >= p["dvol_floor"]) & (px_rb > 0)

    z_btm = _xs_z(btm.reindex(rb_dates).where(elig), p["z_clip"])
    z_mom = _xs_z(mom.reindex(rb_dates).where(elig), p["z_clip"])

    parts = []
    if p["w_value"] > 0:
        parts.append(p["w_value"] * z_btm)
    if p["w_mom"] > 0:
        parts.append(p["w_mom"] * z_mom)
    composite = parts[0] if len(parts) == 1 else (parts[0] + parts[1])

    # Single integrated cross-sectional rank.
    pct = composite.rank(axis=1, pct=True)
    enter_thr = 1.0 - p["enter_pct"]   # top enter_pct enters
    exit_thr = 1.0 - p["exit_pct"]     # held names exit only below top exit_pct

    cols = composite.columns
    held = pd.DataFrame(0.0, index=rb_dates, columns=cols)
    prev = set()
    for dt in rb_dates:
        r = pct.loc[dt].dropna()
        enter_names = set(r.index[r.values >= enter_thr])
        keep_names = set(r.index[r.values >= exit_thr])
        cur = (prev & keep_names) | enter_names
        if cur:
            held.loc[dt, list(cur)] = 1.0
        prev = cur

    # Inverse-vol weights within the long-only book (fully invested, gross = 1).
    iv = (1.0 / vol.reindex(rb_dates).replace(0.0, np.nan)).where(held > 0)
    weights = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)

    # Daily: forward-fill weekly weights, LAG 1 day (no look-ahead), apply turnover cost.
    w_daily = weights.reindex(px.index).ffill().fillna(0.0)
    w_eff = w_daily.shift(1).fillna(0.0)
    gross = (w_eff * rets).sum(axis=1)
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1)
    cost = turnover * (p["cost_bps"] / 1e4)
    daily = (gross - cost).fillna(0.0)
    daily.name = "smallmid_value_momentum"

    # ---- trades: one per held position run (for deployment-sanity) ----
    trades = _build_trades(weights, w_eff, rets, px, rb_dates, sector_map,
                           notional=1_000_000.0)
    return daily, trades


def _build_trades(weights, w_eff, rets, px, rb_dates, sector_map, notional):
    contrib = (w_eff * rets)  # daily per-name PnL fraction
    wmat = weights.values > 0
    rb = list(rb_dates)
    n = len(rb)
    last_day = px.index[-1]
    trades = []

    for j, tk in enumerate(weights.columns):
        col = wmat[:, j]
        if not col.any():
            continue
        sec = sector_map.get(tk, "Unknown")
        ctk = contrib[tk]
        wtk = weights[tk]
        in_run = False
        a = 0
        for i in range(n):
            if col[i] and not in_run:
                a, in_run = i, True
            if in_run and (not col[i] or i == n - 1):
                b = i - 1 if not col[i] else i        # last rb date still held
                entry_dt = rb[a]
                exit_dt = rb[b + 1] if (b + 1) < n else last_day
                window = ctk.loc[entry_dt:exit_dt]
                hold_days = int(len(px.loc[entry_dt:exit_dt].index))
                avg_w = float(wtk.loc[rb[a]:rb[b]].mean())
                pos_val = float(avg_w * notional)
                pnl = float(window.sum(skipna=True) * notional)
                if hold_days > 0 and pos_val > 0:
                    trades.append({
                        "ticker": tk,
                        "sector": sec,
                        "entry_date": entry_dt.strftime("%Y-%m-%d"),
                        "exit_date": exit_dt.strftime("%Y-%m-%d"),
                        "hold_days": hold_days,
                        "position_value": pos_val,
                        "pnl": pnl,
                    })
                in_run = False
    return trades


# --------------------------------------------------------------------------- params
DEFAULTS = dict(
    w_value=0.5, w_mom=0.5,           # equal-weight integrated composite
    enter_pct=0.30, exit_pct=0.50,    # top-tercile entry, hysteresis exit buffer
    dvol_floor=1_000_000.0,           # $1M dollar-volume tradability floor
    mom_lb=252, mom_skip=21,          # 12-1 momentum
    vol_lb=60, z_clip=3.0,
    cost_bps=25.0,                    # small/mid-cap realistic cost (hardened in grid)
)


SPEC = StrategySpec(
    id="smallmid_value_momentum_v1",
    family="equity_value_momentum",
    title="Small/Mid-Cap Value+Momentum Composite (long-only top tercile, hysteresis)",
    markets=["US equities (small/mid cap, survivorship-clean Sharadar SEP)"],
    data_desc=("Sharadar SEP adjusted closes + volume (delisted included) for ~1500 most-liquid "
               "Small/Mid domestic common stocks; Sharadar SF1 ARQ bvps lagged to filing 'datekey' "
               "for point-in-time book-to-market. Daily 2004-present."),
    pre_registration=(
        "Hypothesis: an equal-weight z-score COMBINATION of VALUE (book-to-market, PIT) and "
        "12-1 MOMENTUM, integrated per-name and sorted ONCE, forms a more robust long-only "
        "top-tercile book than either premium standalone, in less-arbitraged small/mid caps. "
        "Method: weekly rebalance, $1M dollar-volume floor, cross-sectional z (clipped ±3), "
        "composite = 0.5*z_value + 0.5*z_momentum, enter top 30% / exit below top 50% (hysteresis "
        "to cut turnover), inverse-vol sizing within the book, signals lagged 1 day, costs hardened "
        "25/50/75 bps. Standalone value-only and momentum-only are pre-declared variants so the "
        "composite must add diversification, not merely dilute one leg. PASS = positive net Sharpe "
        "surviving 50/75bps costs in-sample AND in the 2022+ holdout; broad factor -> must then "
        "GENERALISE to untouched large-cap / sector slices."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "cost_50bps": {"cost_bps": 50.0},
        "cost_75bps": {"cost_bps": 75.0},
        "value_only": {"w_value": 1.0, "w_mom": 0.0},
        "momentum_only": {"w_value": 0.0, "w_mom": 1.0},
        "tight_buffer": {"enter_pct": 0.25, "exit_pct": 0.40},
    },
    scope="broad",
    generalization_universes=["large", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=30,
)