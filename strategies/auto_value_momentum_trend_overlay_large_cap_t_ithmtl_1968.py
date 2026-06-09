"""
Value x Momentum + broad-market trend overlay — LARGE-cap TIER-NATIVE book.

Reproduces the validated small-cap trend-overlay V+M construction at the LARGE-cap tier,
with the grid searched ON THIS TIER (so PBO/DSR are measured on the tier's own OOS).
The V+M *mechanism* generalises across cap tiers (CPCV-hardened, 100% positive paths),
but a config tuned on Small/Mid does NOT transfer to Large (out-of-tier PBO 0.71-0.97).
This is the missing Large leg of a true multi-tier V+M portfolio.

No external side effects: pure data adapters + numpy/pandas. No writes, no capital, no config.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (
    sep_panel, us_universe, sf1, yf_panel, fred_series,
    trend_returns, carry_returns, inv_vol_position,
)
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
# Parameters (the grid is declared below; PBO/DSR effective-N = honest search burden)
# ----------------------------------------------------------------------------- #
DEFAULTS = dict(
    w_value=0.5,          # 50/50 composite
    w_mom=0.5,
    mom_long=252,         # 12-1 momentum: shift(21)/shift(252)-1
    mom_gap=21,
    vol_lb=63,            # inverse-vol sizing lookback (~3 months)
    entry_quantile=2.0 / 3.0,   # long-only TOP tercile
    band=0.0667,          # no-trade hysteresis (keep held names >= 2/3 - band, in quantile space)
    use_trend=True,       # broad-market trend overlay (risk-off to cash below MA)
    trend_ma=200,
    winsor=0.05,          # cross-sectional winsorise at 5/95 pct before z-score
    min_names=20,         # min cross-section to act on a rebalance
    cost_bps=0.0008,      # ~8bps on turnover
)

# Sharadar TICKERS GICS-style sectors (11). Per-sector Large-cap pull -> bounded universe.
_SECTORS = [
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive", "Energy",
    "Basic Materials", "Real Estate", "Utilities",
]

_START = "2003-01-01"
_BOOK = 1_000_000.0          # notional for trade-level position_value / pnl reporting
_SID = "valmom_trend_largecap"

# Module-level fallback cache (belt-and-suspenders if .attrs is dropped by a copy)
_BVPS_CACHE = None
_SECTOR_CACHE = None


# ----------------------------------------------------------------------------- #
# load_data
# ----------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Survivorship-clean LARGE-cap panel: SEP closeadj prices + SF1 ARQ bvps (datekey-lagged).

    Returns the price panel (DatetimeIndex x tickers). The point-in-time book-value-per-share
    panel and the ticker->sector map are attached via .attrs (and a module cache fallback).
    """
    global _BVPS_CACHE, _SECTOR_CACHE

    # --- bounded, survivorship-clean universe, built per-sector for a clean sector map ---
    tickers = []
    sector_map = {}
    for sec in _SECTORS:
        try:
            names = us_universe(
                sector=sec,
                category="Domestic Common Stock",
                marketcap="Large",
                include_delisted=True,
                top_n=130,
            )
        except Exception:
            names = []
        for t in names:
            if t not in sector_map:
                sector_map[t] = sec
                tickers.append(t)

    # fallback if sector strings under-deliver -> keep the book tradable (sector 'Unknown')
    if len(tickers) < 200:
        try:
            extra = us_universe(
                category="Domestic Common Stock",
                marketcap="Large",
                include_delisted=True,
                top_n=1200,
            )
        except Exception:
            extra = []
        for t in extra:
            if t not in sector_map:
                sector_map[t] = "Unknown"
                tickers.append(t)

    tickers = sorted(set(tickers))

    # --- prices: split+div adjusted, delisted included ---
    px = sep_panel(tickers, _START, field="closeadj")
    px = px.sort_index()
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep="last")]

    # --- value: point-in-time book-value-per-share, as-of datekey (NO look-ahead) ---
    bvps = _bvps_panel(list(px.columns), px.index)

    # align both maps to the actual traded names
    sector_map = {t: sector_map.get(t, "Unknown") for t in px.columns}

    _BVPS_CACHE = bvps
    _SECTOR_CACHE = sector_map
    try:
        px.attrs["bvps"] = bvps
        px.attrs["sector"] = sector_map
    except Exception:
        pass
    return px


def _bvps_panel(tickers, px_index) -> pd.DataFrame:
    """As-of (datekey-lagged) book-value-per-share panel reindexed to trading days."""
    raw = sf1(tickers, ["bvps"], dimension="ARQ")

    if isinstance(raw, pd.DataFrame) and "datekey" in raw.columns and "ticker" in raw.columns:
        df = raw[["ticker", "datekey", "bvps"]].copy()
        df["datekey"] = pd.to_datetime(df["datekey"])
        df = df.dropna(subset=["datekey"])
        df = df.sort_values("datekey")
        panel = df.pivot_table(index="datekey", columns="ticker", values="bvps", aggfunc="last")
    else:
        # assume already a date-indexed wide panel
        panel = pd.DataFrame(raw).copy()
        panel.index = pd.to_datetime(panel.index)

    panel = panel.sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    # forward-fill the latest filed value to each trading day, then reindex to price calendar
    panel = panel.reindex(panel.index.union(px_index)).ffill().reindex(px_index)
    return panel.reindex(columns=tickers)


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _zscore(df: pd.DataFrame, w: float) -> pd.DataFrame:
    """Per-day cross-sectional winsorise (w / 1-w) then z-score."""
    lo = df.quantile(w, axis=1)
    hi = df.quantile(1.0 - w, axis=1)
    clipped = df.clip(lower=lo, upper=hi, axis=0)
    mu = clipped.mean(axis=1)
    sd = clipped.std(axis=1).replace(0.0, np.nan)
    return clipped.sub(mu, axis=0).div(sd, axis=0)


def _rebalance_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week."""
    wk = pd.Index(idx).strftime("%G%V")
    tmp = pd.DataFrame({"dt": idx, "wk": wk})
    last = tmp.groupby("wk")["dt"].last()
    return pd.DatetimeIndex(sorted(last.values))


# ----------------------------------------------------------------------------- #
# signal
# ----------------------------------------------------------------------------- #
def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)

    px = panel
    bvps = panel.attrs.get("bvps", _BVPS_CACHE) if hasattr(panel, "attrs") else _BVPS_CACHE
    sector_map = panel.attrs.get("sector", _SECTOR_CACHE) if hasattr(panel, "attrs") else _SECTOR_CACHE
    if bvps is None:
        bvps = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    if sector_map is None:
        sector_map = {t: "Unknown" for t in px.columns}
    bvps = bvps.reindex(index=px.index, columns=px.columns)

    rets = px.pct_change()

    # ---- factor signals ----
    bm = bvps / px                                              # value: book-to-market (cheap = high)
    mom = px.shift(p["mom_gap"]) / px.shift(p["mom_long"]) - 1.0  # momentum: 12-1

    z_val = _zscore(bm, p["winsor"])
    z_mom = _zscore(mom, p["winsor"])

    wv, wm = float(p["w_value"]), float(p["w_mom"])
    if wv > 0 and wm > 0:
        score = wv * z_val + wm * z_mom        # composite requires both present
    elif wv > 0:
        score = z_val
    else:
        score = z_mom

    # ---- weekly membership with no-trade hysteresis (top tercile entry, looser exit) ----
    rebal = _rebalance_dates(px.index)
    entry_q = float(p["entry_quantile"])
    exit_q = max(0.0, entry_q - float(p["band"]))

    current = pd.Series(False, index=px.columns)
    mem = {}
    for dt in rebal:
        s = score.loc[dt]
        valid = s.dropna()
        if len(valid) >= p["min_names"]:
            e_th = valid.quantile(entry_q)
            x_th = valid.quantile(exit_q)
            entries = s >= e_th
            keep = current & (s >= x_th)
            current = ((entries | keep) & s.notna()).fillna(False)
        mem[dt] = current.copy()

    held = pd.DataFrame(mem).T
    held.index = pd.DatetimeIndex(held.index)
    held = held.reindex(index=px.index, method="ffill").reindex(columns=px.columns)
    held = held.fillna(False).astype(bool)

    # ---- inverse-vol sizing (long-only, fully invested across the held set) ----
    vol = rets.rolling(p["vol_lb"]).std()
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    raw_w = held.astype(float) * inv_vol
    raw_w = raw_w.fillna(0.0)
    denom = raw_w.sum(axis=1).replace(0.0, np.nan)
    weights = raw_w.div(denom, axis=0).fillna(0.0)

    # ---- broad-market trend overlay (equal-weight index < trailing MA -> cash) ----
    if p["use_trend"]:
        ew_ret = rets.mean(axis=1)
        ew_level = (1.0 + ew_ret.fillna(0.0)).cumprod()
        ma = ew_level.rolling(p["trend_ma"]).mean()
        trend_on = (ew_level > ma).astype(float)
    else:
        trend_on = pd.Series(1.0, index=px.index)

    # ---- portfolio returns: lag signals + overlay 1 day; ~8bps on turnover ----
    eff_w = weights.shift(1).fillna(0.0)
    eff_trend = trend_on.shift(1).fillna(0.0)
    applied = eff_w.mul(eff_trend, axis=0)

    port_gross = (applied * rets).sum(axis=1)
    turnover = applied.diff().abs().sum(axis=1).fillna(0.0)
    daily_returns = (port_gross - turnover * p["cost_bps"]).astype(float)
    daily_returns.name = _SID

    # trim to the first day with live exposure
    active = applied.abs().sum(axis=1) > 0
    if active.any():
        daily_returns = daily_returns.loc[active.idxmax():]
    daily_returns = daily_returns.dropna()

    # ---- trades: one per held position-run (factor-leg positions) ----
    trades = _build_trades(px, held, weights, sector_map)

    return daily_returns, trades


def _build_trades(px, held, weights, sector_map):
    idx = px.index
    cols = list(px.columns)
    held_vals = held.values
    w_vals = weights.values
    p_vals = px.values
    trades = []

    for j, tk in enumerate(cols):
        colj = held_vals[:, j].astype(np.int8)
        if colj.sum() == 0:
            continue
        prev = np.concatenate(([0], colj[:-1]))
        nxt = np.concatenate((colj[1:], [0]))
        starts = np.where((colj == 1) & (prev == 0))[0]
        ends = np.where((colj == 1) & (nxt == 0))[0]
        sec = sector_map.get(tk, "Unknown")
        pj = p_vals[:, j]
        wj = w_vals[:, j]
        for a, b in zip(starts, ends):
            pa, pb = pj[a], pj[b]
            if not (np.isfinite(pa) and np.isfinite(pb)) or pa <= 0:
                continue
            seg = wj[a:b + 1]
            avg_w = np.nanmean(seg)
            if not np.isfinite(avg_w) or avg_w <= 0:
                continue
            pv = float(avg_w * _BOOK)
            ret = float(pb / pa - 1.0)
            trades.append({
                "ticker": tk,
                "sector": sec,
                "entry_date": idx[a].strftime("%Y-%m-%d"),
                "exit_date": idx[b].strftime("%Y-%m-%d"),
                "hold_days": int(b - a + 1),
                "position_value": pv,
                "pnl": float(pv * ret),
            })
    return trades


# ----------------------------------------------------------------------------- #
# Pre-declared search grid (DSR effective-N) — searched ON THIS (Large) tier
# ----------------------------------------------------------------------------- #
GRID = {
    "default": {},                                   # primary: 50/50 V+M + 200d trend overlay
    "value_only": {"w_value": 1.0, "w_mom": 0.0},
    "mom_only": {"w_value": 0.0, "w_mom": 1.0},
    "no_trend": {"use_trend": False},
    "tight_band": {"band": 0.0167},                  # tighter no-trade hysteresis
    "trend_ma_150": {"trend_ma": 150},               # overlay MA-length robustness
}


# ----------------------------------------------------------------------------- #
# SPEC
# ----------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id=_SID,
    family="value_momentum",
    title="Value x Momentum + trend-overlay — LARGE-cap TIER-NATIVE re-tune (multi-tier scale-out leg)",
    markets=["US large-cap equities"],
    data_desc=(
        "Sharadar SEP closeadj (split+div adj, delisted incl -> survivorship-clean) + SF1 ARQ bvps "
        "lagged to 'datekey' filing date. Universe = us_universe(category='Domestic Common Stock', "
        "marketcap='Large', include_delisted=True) built per GICS-style sector (top_n~130/sector -> "
        "~1.0-1.3k bounded liquid names). Value = point-in-time book-to-market (bvps/closeadj); "
        "Momentum = 12-1 (shift21/shift252-1); both winsorised 5/95 + daily cross-sectional z; "
        "50/50 composite; long-only top tercile with hysteresis; inverse-vol (63d) sized; weekly "
        "rebalance; broad-market 200d trend overlay (risk-off to cash); all signals lagged 1d; ~8bps cost."
    ),
    pre_registration=(
        "FROZEN reproduction of the validated Small-cap trend-overlay V+M book at the LARGE-cap tier. "
        "Hypothesis: complementary Value (B/M) and Momentum (12-1) premia + a broad trend overlay clear "
        "CPCV/PBO/DSR on the LARGE tier's OWN OOS when the grid is searched on this tier (the V+M mechanism "
        "generalises across cap tiers, but a Small/Mid-tuned config does NOT transfer to Large: out-of-tier "
        "PBO 0.71-0.97). Construction (universe rule, B/M from datekey-lagged bvps, 12-1, 5/95 winsorise + "
        "daily z, 50/50, top tercile, inverse-vol 63d, weekly rebalance, 200d trend overlay, 1d lag, 8bps) "
        "is fixed in advance; only the declared grid is searched, on Large data. Crowding is high (V+M is "
        "published) — the test is whether a tier-native build still clears the rails on its own tier."
    ),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid=GRID,
    scope="broad",
    generalization_universes=["mid", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)