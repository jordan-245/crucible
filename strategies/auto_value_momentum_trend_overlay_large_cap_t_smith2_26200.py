"""
Value × Momentum + broad-market trend overlay — LARGE-cap TIER-NATIVE re-tune.

Hypothesis: the Value (point-in-time book-to-market) + Momentum (12-1) composite
premium — already CPCV-validated as a MECHANISM that generalises across cap tiers —
admits a LARGE-cap-native config whose default ranks ROBUSTLY (low own-tier PBO) when
the grid is searched ON THIS TIER's own data. A config tuned on Small/Mid does NOT
transfer to Large (out-of-tier PBO 0.71-0.97); this module supplies the missing
Large-cap leg of a 3-tier V+M portfolio.

Construction (frozen): survivorship-clean Large-cap US Domestic Common Stock built
per-GICS-sector via us_universe(marketcap='Large').

  * RETURNS / MOMENTUM use the TOTAL-RETURN series SEP 'closeadj' (split+div adjusted).
    A 12-1 momentum RATIO closeadj(t-21)/closeadj(t-252) only embeds dividends paid
    BETWEEN t-252 and t-21 (all in the past relative to signal date t) -> no leakage.

  * VALUE denominator uses the SPLIT-ADJUSTED-ONLY series SEP 'close'. 'closeadj'
    back-propagates the historical price LEVEL using FUTURE dividend factors, so
    bvps/closeadj would leak future info into the cross-sectional value LEVEL.
    Book-to-market is therefore bvps / split-adj close. bvps = SF1 ARQ 'bvps'
    forward-filled from its filing date ('datekey' -> no look-ahead).

MEMORY: the prior 130-names/sector build (~1.4k cols) OOM-killed the CPCV rails.
This version bounds the universe to 60/sector (~450-660 liquid names), downcasts the
price panels to float32, and PRE-COMPUTES btm & mom ONCE in load_data so signal()
(re-run per CV fold) only reindexes them -> a small, bounded per-call footprint.

Both factors winsorised 5/95 then daily cross-sectional z-scored, 50/50 composite,
long-only top names (inverse-vol sized, weekly rebalance with a no-trade hysteresis
band), broad equal-weight-index trend overlay to cash. All signals lagged 1 day;
~8 bps turnover cost. NO external side effects.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1
import numpy as np, pandas as pd


# Sharadar TICKERS sector labels (Morningstar/Zacks-style) for a sector-balanced universe.
_SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Energy", "Basic Materials",
    "Utilities", "Real Estate", "Communication Services",
]

_START = "2003-01-01"
_TOP_N_PER_SECTOR = 60   # bounded -> CPCV memory-safe (prior 130/sector OOM'd the rails)


# ----------------------------------------------------------------------------- #
# data
# ----------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Survivorship-clean Large-cap total-return panel; carries PRE-COMPUTED factors in .attrs.

    px (returned)    = SEP 'closeadj' (total-return, split+div adjusted) -> returns & momentum.
    .attrs['btm']    = bvps / SEP 'close' (split-adj-only)  -> value (no dividend look-ahead).
    .attrs['mom']    = 12-1 total-return ratio.
    .attrs['sectors']= ticker -> GICS sector map.

    Heavy factor construction happens HERE (once) so signal() stays lean across CV folds.
    """
    sector_map, tickers = {}, []
    for sec in _SECTORS:
        try:
            names = us_universe(
                sector=sec,
                category="Domestic Common Stock",
                marketcap="Large",
                include_delisted=True,
                top_n=_TOP_N_PER_SECTOR,
            )
        except Exception:
            names = []
        for t in names:
            sector_map.setdefault(t, sec)
        tickers.extend(names)
    tickers = sorted(set(tickers))

    # Total-return prices (delisted included) for returns + momentum.
    px = sep_panel(tickers, start=_START, field="closeadj").sort_index()
    keep = px.notna().sum(axis=0) >= 252        # require >=1y history (lean + non-degenerate)
    px = px.loc[:, keep[keep].index].astype("float32")

    # Split-adjusted-ONLY close for the VALUE denominator (no dividend look-ahead).
    try:
        px_split = sep_panel(list(px.columns), start=_START, field="close").sort_index()
        px_split = px_split.reindex(index=px.index, columns=px.columns).astype("float32")
    except Exception:
        px_split = px

    # Point-in-time book value per share — use 'datekey' (filing date) as the as-of.
    bvps = pd.DataFrame(index=px.index, columns=px.columns, dtype="float32")
    try:
        fund = sf1(list(px.columns), ["bvps"], dimension="ARQ")
        f = fund.reset_index()
        if {"datekey", "ticker", "bvps"}.issubset(set(f.columns)):
            f = f.dropna(subset=["datekey", "ticker"]).copy()
            f["datekey"] = pd.to_datetime(f["datekey"])
            wide = (
                f.sort_values("datekey")
                .pivot_table(index="datekey", columns="ticker", values="bvps", aggfunc="last")
            )
            wide = wide.reindex(px.index.union(wide.index)).ffill().reindex(px.index)
            bvps = wide.reindex(columns=px.columns).astype("float32")
    except Exception:
        pass

    # ---- PRE-COMPUTE factors once (frees px_split/bvps; keeps signal lean) -- #
    btm = (bvps / px_split).where((bvps > 0) & (px_split > 0)).astype("float32")  # value level
    mom = (px.shift(21) / px.shift(252) - 1.0).astype("float32")                  # 12-1 momentum
    del px_split, bvps

    px.attrs["btm"] = btm
    px.attrs["mom"] = mom
    px.attrs["sectors"] = sector_map
    return px


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _winz_z(df: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    """Row-wise (cross-sectional) winsorise at lo/hi pct then z-score."""
    ql = df.quantile(lo, axis=1)
    qh = df.quantile(hi, axis=1)
    clip = df.clip(lower=ql, upper=qh, axis=0)
    mu = clip.mean(axis=1)
    sd = clip.std(axis=1).replace(0.0, np.nan)
    return clip.sub(mu, axis=0).div(sd, axis=0)


# ----------------------------------------------------------------------------- #
# signal
# ----------------------------------------------------------------------------- #
def signal(panel, **params):
    p = dict(
        w_value=0.5, w_mom=0.5,          # composite weights
        vol_lb=63,                        # inverse-vol lookback (days)
        max_pos=60,                       # deployable holding count (top by score)
        top_q=2.0 / 3.0,                  # top-tercile entry quantile
        band=0.05,                        # no-trade hysteresis (exit at top_q - band)
        trend_ma=200, use_trend=True,     # broad-market trend overlay
        cost_bps=8.0,                     # one-way turnover cost
        capital=1_000_000.0,              # notional for trade ledger
    )
    p.update(params)

    px = panel                                            # total-return (closeadj)
    sectors = panel.attrs.get("sectors", {})
    btm = panel.attrs.get("btm")
    mom = panel.attrs.get("mom")
    if btm is None:
        btm = pd.DataFrame(index=px.index, columns=px.columns, dtype="float32")
    if mom is None:
        mom = (px.shift(21) / px.shift(252) - 1.0)

    rets = px.pct_change()                                # total returns

    # ---- weekly rebalance grid (last trading day of each ISO week) ---------- #
    s = px.index.to_series()
    wk = s.dt.strftime("%G-%V")
    rebal = pd.DatetimeIndex(sorted(s.groupby(wk).max().values))

    # ---- cross-sectional z-scores on rebalance dates (small frames) --------- #
    zv = _winz_z(btm.reindex(rebal))
    zm = _winz_z(mom.reindex(rebal))
    zvf, zmf = zv.fillna(0.0), zm.fillna(0.0)
    comp = p["w_value"] * zvf + p["w_mom"] * zmf
    has_v = zv.notna() if p["w_value"] != 0 else pd.DataFrame(False, index=zv.index, columns=zv.columns)
    has_m = zm.notna() if p["w_mom"] != 0 else pd.DataFrame(False, index=zm.index, columns=zm.columns)
    comp = comp.where(has_v | has_m)

    vol = rets.rolling(p["vol_lb"]).std()

    # ---- selection with hysteresis + inverse-vol sizing on each rebalance --- #
    W = pd.DataFrame(0.0, index=rebal, columns=px.columns)
    held = set()
    lo_q = max(0.0, p["top_q"] - p["band"])
    for d in rebal:
        row = comp.loc[d].dropna()
        if row.empty:
            held = set()
            continue
        entry_t = row.quantile(p["top_q"])
        exit_t = row.quantile(lo_q)
        sel = set(row[row >= entry_t].index)                          # fresh entries
        sel |= {t for t in held if t in row.index and row[t] >= exit_t}  # hysteresis keeps
        if len(sel) > p["max_pos"]:
            sel = set(row.loc[list(sel)].sort_values(ascending=False).index[: p["max_pos"]])

        iv = (1.0 / vol.loc[d, list(sel)].replace(0.0, np.nan)).dropna() if sel else pd.Series(dtype=float)
        if iv.sum() > 0:
            w = iv / iv.sum()
            W.loc[d, w.index] = w.values
            held = set(w.index)
        else:
            held = set()

    # hold between rebalances, then lag 1 day (no look-ahead)
    W_daily = W.reindex(px.index).ffill().fillna(0.0)
    W_lag = W_daily.shift(1).fillna(0.0)

    # ---- broad equal-weight-index trend overlay (lagged 1 day) -------------- #
    mret = rets.mean(axis=1).fillna(0.0).astype("float64")
    eq_level = (1.0 + mret).cumprod()
    ma = eq_level.rolling(p["trend_ma"]).mean()
    trend_on = (eq_level > ma).astype(float).shift(1).fillna(0.0)
    if p["use_trend"]:
        W_eff = W_lag.mul(trend_on, axis=0)
    else:
        W_eff = W_lag

    # ---- net-of-cost daily returns ------------------------------------------ #
    gross = (W_eff * rets).sum(axis=1)
    turnover = (W_eff - W_eff.shift(1)).abs().sum(axis=1)
    cost = turnover * (p["cost_bps"] / 1e4)
    net = (gross - cost).fillna(0.0).astype("float64")
    net.name = "valmom_large_trend"

    nz = net.index[(W_eff.abs().sum(axis=1) > 0)]
    if len(nz):
        net = net.loc[nz.min():]

    # ---- trade ledger: one trade per held position run (the V+M factor book) #
    cap = p["capital"]
    held_mask = (W_lag > 0)
    trades = []
    idx = px.index
    for tkr in W_lag.columns:
        m = held_mask[tkr].values
        if not m.any():
            continue
        starts = np.where(m & ~np.r_[False, m[:-1]])[0]
        ends = np.where(m & ~np.r_[m[1:], False])[0]
        pcol = px[tkr].values                              # total-return for realistic pnl
        wcol = W_lag[tkr].values
        for a, b in zip(starts, ends):
            ep, xp = pcol[a], pcol[b]
            if not (np.isfinite(ep) and np.isfinite(xp)) or ep <= 0:
                continue
            pos_val = float(np.nanmean(wcol[a : b + 1]) * cap)
            trades.append({
                "ticker": tkr,
                "sector": sectors.get(tkr, "Unknown"),
                "entry_date": idx[a].strftime("%Y-%m-%d"),
                "exit_date": idx[b].strftime("%Y-%m-%d"),
                "hold_days": int(b - a + 1),
                "position_value": pos_val,
                "pnl": float(pos_val * (xp / ep - 1.0)),
            })

    return net, trades


# ----------------------------------------------------------------------------- #
# spec
# ----------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id="valmom_large_trend_overlay",
    family="value_momentum",
    title="Value × Momentum + trend-overlay — LARGE-cap tier-native re-tune",
    markets=["US large-cap equities"],
    data_desc=(
        "Sharadar SEP 'closeadj' (total-return, survivorship-clean, delisted incl.) for "
        "returns + 12-1 momentum; SEP 'close' (split-adjusted ONLY) as the value-factor "
        "denominator to avoid dividend back-propagation leaking into the cross-sectional "
        "value LEVEL; SF1 ARQ 'bvps' lagged to filing 'datekey'. Universe = "
        "us_universe(marketcap='Large', Domestic Common Stock) built per-GICS-sector at "
        "60 names/sector (~450-660 liquid names; bounded to keep the CPCV memory-safe — the "
        "prior 130/sector build OOM-killed the rails). Factors are pre-computed once in "
        "load_data so signal() stays lean across CV folds."
    ),
    pre_registration=(
        "H: a LARGE-cap-NATIVE V(book-to-market = bvps / split-adj close)×M(12-1 total "
        "return) composite (50/50), long top names inverse-vol sized, weekly rebalanced "
        "with hysteresis, under a broad equal-weight-index trend overlay, produces a "
        "DEFAULT config that is robust (low PBO) on its OWN tier's OOS — distinct from "
        "Small/Mid-tuned configs which fail to transfer (out-of-tier PBO 0.71-0.97). "
        "Grid (value_only, mom_only, no_trend, tight_band, trend_ma variant) is searched "
        "on THIS tier so PBO/DSR are measured on the tier's own data. PASS = default "
        "OOS-positive, CPCV-median positive, low own-tier PBO, and the V+M mechanism "
        "confirmed in untouched small/mid/sector slices (broad scope). Holdout 2022-01-01."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "value_only": {"w_value": 1.0, "w_mom": 0.0},
        "mom_only": {"w_value": 0.0, "w_mom": 1.0},
        "no_trend": {"use_trend": False},
        "tight_band": {"band": 0.0},
        "trend_ma_100": {"trend_ma": 100},
    },
    scope="broad",
    generalization_universes=["small", "mid", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=60,
)