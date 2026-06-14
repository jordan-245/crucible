"""
Commodity hedging-pressure premium (Basu-Miffre, 2013) — REAL COT commercial positioning.

Mechanism (Keynes normal-backwardation / insurance provision): speculators are paid to
absorb the net hedging demand of commercial producers/consumers. Measured DIRECTLY from
CFTC COT commercial net positioning (not a 12m-return proxy — that proxy is what the dead
`hedging_pressure_footprint_ls_v2` falsified; this is the canonical axis-change to new data).

FROZEN signal: weekly, on each COT release date, HP = commercial_net / open_interest per root;
convert to a per-root 52-week ROLLING PERCENTILE (so the signal is the DEVIATION in hedging
demand, not the structural level — commercials are persistently net-short in some markets).
Cross-sectional sort across the roots -> LONG lowest-HP tercile (commercials most net-short ->
hedgers paying longs to carry inventory risk), SHORT highest-HP tercile. Inverse-60d-vol risk
weight, 10% vol target, gross <= 2x, weekly rebalance on release dates, 1-day execution lag.

Data: cot_positioning() (release-date indexed -> look-ahead pre-closed) for the signal;
fut_curve() Databento individual-contract futures (roll-aware, within-contract front-month
returns; cross-roll gaps excluded internally). BOTH ADAPTERS ARE SINGLE-ROOT (one owned parquet
per root) -> they are called PER-ROOT and assembled here (the prior version passed the whole list
as one "root", which stringified into the filename and 404'd — that was the bug).
Both OWNED ($0 incremental). NO trend hedge here — the 2026-06-08 lesson says test the premium
STANDALONE first; a sized tail-overlay is a later step.

scope='broad': the insurance-provision premium is a UNIVERSAL positioning mechanism, so it must
generalise to DISJOINT, untouched futures markets (softs / currencies / financials — share NO
tickers with the 16 commodity search roots). Within-commodity sector sub-slices would NOT be
disjoint from the all-16 search book, so generalisation is tested cross-asset (a stronger test).
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, cot_positioning, fut_curve   # COT + curve adapters (owned, 2026-06-12)
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- constants
SID        = "comm_hedging_pressure_cot_ls_v1"
START      = "2010-01-01"
HOLDOUT    = "2022-01-01"
ROLL_BPS   = 4.0          # 1 tick + fees per side, applied per-root on any detected roll days
DEFAULTS   = dict(long_frac=1/3., short_frac=1/3., vol_lb=60, target_vol=0.10, gross_cap=2.0)

# 16 CME roots (PA excluded — thin rank-2 coverage)
SEARCH_ROOTS = ["CL","NG","HO","RB","GC","SI","HG","PL",
                "ZC","ZS","ZW","ZL","ZM","LE","HE","GF"]

# DISJOINT cross-asset generalisation universes (share NO tickers with SEARCH_ROOTS).
# Returns via free yf_panel front-month, COT via the same cot_positioning() adapter.
GEN_UNIVERSES = {
    "softs":       ["SB","KC","CC","CT","OJ"],          # ags softs
    "currencies":  ["6E","6J","6B","6A","6C","6S","6N"],# FX futures
    "financials":  ["ZN","ZB","ZF","ZT","ES","NQ","YM"],# rates + equity index
}

YF_SYMBOL = {
    "SB":"SB=F","KC":"KC=F","CC":"CC=F","CT":"CT=F","OJ":"OJ=F",
    "6E":"6E=F","6J":"6J=F","6B":"6B=F","6A":"6A=F","6C":"6C=F","6S":"6S=F","6N":"6N=F",
    "ZN":"ZN=F","ZB":"ZB=F","ZF":"ZF=F","ZT":"ZT=F","ES":"ES=F","NQ":"NQ=F","YM":"YM=F",
}
YF_TO_ROOT = {v: k for k, v in YF_SYMBOL.items()}

SECTOR_MAP = {
    "CL":"Energy","NG":"Energy","HO":"Energy","RB":"Energy",
    "GC":"Metals","SI":"Metals","HG":"Metals","PL":"Metals",
    "ZC":"Grains","ZS":"Grains","ZW":"Grains","ZL":"Grains","ZM":"Grains",
    "LE":"Livestock","HE":"Livestock","GF":"Livestock",
    "SB":"Softs","KC":"Softs","CC":"Softs","CT":"Softs","OJ":"Softs",
    "6E":"FX","6J":"FX","6B":"FX","6A":"FX","6C":"FX","6S":"FX","6N":"FX",
    "ZN":"Rates","ZB":"Rates","ZF":"Rates","ZT":"Rates","ES":"Equity","NQ":"Equity","YM":"Equity",
}

# ----------------------------------------------------------------------------- helpers
def _col(df, a, b):
    """Fetch a MultiIndex column tolerant of (a,b) vs (b,a) ordering + case; else None."""
    if df is None or not isinstance(df.columns, pd.MultiIndex):
        return None
    if (a, b) in df.columns: return df[(a, b)]
    if (b, a) in df.columns: return df[(b, a)]
    low = {tuple(str(x).lower() for x in c): c for c in df.columns}
    k1, k2 = (str(a).lower(), str(b).lower()), (str(b).lower(), str(a).lower())
    if k1 in low: return df[low[k1]]
    if k2 in low: return df[low[k2]]
    return None


def _field(cot, root, candidates):
    """First non-null COT field for `root` among candidate names (schema-tolerant)."""
    for f in candidates:
        s = _col(cot, root, f)
        if s is not None:
            return s
    return None


def _as_returns(s):
    """Interpret a bare numeric Series: if it looks like a PRICE level (all-positive, median
    >> 1) convert via pct_change; otherwise it is already a return series -> use as-is."""
    s = s.astype(float)
    v = s.dropna()
    if v.empty:
        return s
    if (v > 0).all() and v.median() > 1.0:
        return s.pct_change()
    return s


def _root_returns(fc):
    """Daily return Series from ONE root's fut_curve output (single-root adapter).
    Prefers an explicit roll-aware return column; else derives from a price column
    (close/settle) of the roll-adjusted continuous front-month; else interprets a bare Series.
    fut_curve excludes cross-roll gaps internally, so no extra roll masking is required."""
    if fc is None:
        return None
    if isinstance(fc, pd.Series):
        s = fc.copy(); s.index = pd.to_datetime(s.index)
        return _as_returns(s)
    if isinstance(fc, pd.DataFrame):
        df = fc.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(x) for x in c if x is not None).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        cols = list(df.columns)
        for k in ("ret", "return", "returns", "front_ret", "ret_front", "logret", "log_ret"):
            if k in cols:
                return df[k].astype(float)
        for k in ("closeadj", "adj_close", "close", "settle", "settlement", "px", "price", "last"):
            if k in cols:
                r = df[k].astype(float).pct_change()
                for rk in ("roll", "is_roll", "roll_flag"):
                    if rk in cols:
                        r = r.mask(df[rk].astype(bool).values, 0.0)
                        break
                return r
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 1:
            return _as_returns(num.iloc[:, 0])
    return None


def _cot_one(root, start):
    """COT for ONE root (field columns: comm_net, oi, ...). Robust to start= kwarg presence."""
    for call in (lambda: cot_positioning(root, start=start), lambda: cot_positioning(root)):
        try:
            c = call()
            if c is not None:
                return c
        except Exception:
            continue
    return None


def _cot_panel(roots, start):
    """Assemble per-root COT into a MultiIndex (root, field) frame. Works whether the adapter
    is single-root (the proven fut_curve-style signature) or, as a fallback, list-aware."""
    frames = {}
    for r in roots:
        cr = _cot_one(r, start)
        if cr is None:
            continue
        if isinstance(cr, pd.Series):
            cr = cr.to_frame()
        if isinstance(cr.columns, pd.MultiIndex):
            top = cr.columns.get_level_values(0)
            if r in set(top):
                cr = cr.xs(r, axis=1, level=0)
            else:
                cr.columns = ["_".join(str(x) for x in c if x is not None) for c in cr.columns]
        cr = cr.copy(); cr.index = pd.to_datetime(cr.index)
        frames[r] = cr
    if frames:
        return pd.concat(frames, axis=1)
    # fallback: adapter may be list-aware and return a (root, field) MultiIndex directly
    for call in (lambda: cot_positioning(roots, start=start), lambda: cot_positioning(roots)):
        try:
            c = call()
            if c is not None and isinstance(getattr(c, "columns", None), pd.MultiIndex):
                return c
        except Exception:
            continue
    return pd.DataFrame()


def _hp_percentile(cot, roots, lookback=52):
    """Per-root HP = comm_net / OI -> trailing 52-week rolling percentile (no look-ahead:
    the window is strictly trailing; percentile = fraction of PRIOR window values below current)."""
    out = {}
    for r in roots:
        comm = _field(cot, r, ["comm_net", "commercial_net", "comm_net_pos", "net_comm", "comm"])
        oi   = _field(cot, r, ["oi", "open_interest", "openinterest", "oi_total", "open_int"])
        if comm is None or oi is None:
            continue
        hp = (comm.astype(float) / oi.astype(float).replace(0, np.nan)).dropna()
        if len(hp) < lookback // 2:
            continue
        out[r] = hp.rolling(lookback, min_periods=max(13, lookback // 2)).apply(
            lambda w: float((w[:-1] < w[-1]).mean()), raw=True)
    return pd.DataFrame(out) if out else pd.DataFrame()


def _build_panel(roots, start, source):
    """Panel schema (shared by load_data + load_gen_data): MultiIndex columns
    {'ret','hp_pct','roll'} x root, daily DatetimeIndex. hp_pct is the release-date HP
    percentile ffilled forward (PIT-safe: only info public on/after release is used)."""
    start_ts = pd.Timestamp(start)
    if source == "databento":
        rets = {}
        for r in roots:                                   # SINGLE-ROOT adapter -> loop per root
            try:
                fc = fut_curve(r)
            except Exception:
                continue
            s = _root_returns(fc)
            if s is not None and s.notna().any():
                rets[r] = s
        rets = pd.DataFrame(rets)
    else:  # free yf front-month continuous (gen universes only)
        px = yf_panel([YF_SYMBOL[r] for r in roots], start=start)
        px = px.rename(columns={c: YF_TO_ROOT.get(c, c) for c in px.columns})
        px = px[[r for r in roots if r in px.columns]].astype(float)
        rets = px.pct_change()
    if rets is None or rets.empty:
        return pd.DataFrame()
    rets.index = pd.to_datetime(rets.index)
    rets = rets.loc[rets.index >= start_ts].dropna(how="all")
    hp = _hp_percentile(_cot_panel(roots, start), roots).reindex(rets.index, method="ffill")
    common = [r for r in roots if r in rets.columns and r in hp.columns]
    if not common:
        return pd.DataFrame()
    rets, hp = rets[common], hp[common]
    rolls = pd.DataFrame(0.0, index=rets.index, columns=common)
    return pd.concat({"ret": rets, "hp_pct": hp, "roll": rolls}, axis=1).dropna(how="all")


def _signal_weights(panel, long_frac, short_frac, vol_lb, target_vol, gross_cap):
    """Cross-sectional tercile L/S, inverse-vol risk weight, vol-target (de-lever only),
    weekly rebalance. Weights are built from same-day data; the 1-day execution lag is
    applied by the CALLER (signal) via W.shift(1)."""
    rets, hp = panel["ret"], panel["hp_pct"]
    sig = -hp                                   # low HP (commercials net-short) -> long
    rank = sig.rank(axis=1, pct=True)
    long_mask, short_mask = rank > (1 - long_frac), rank <= short_frac

    vol = rets.rolling(vol_lb, min_periods=max(10, vol_lb // 2)).std()
    iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    lw = iv.where(long_mask).fillna(0.0)
    sw = iv.where(short_mask).fillna(0.0)
    lw = lw.div(lw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = sw.div(sw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    raw = lw - sw                               # dollar-neutral, gross ~2

    # vol target: trailing realised book vol -> single de-lever factor (never adds leverage)
    book = (raw.shift(1) * rets).sum(axis=1)
    rv = book.rolling(63, min_periods=21).std() * np.sqrt(252)
    scale = (target_vol / rv).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(lower=0.0, upper=gross_cap / 2.0).ffill().fillna(0.0)  # raw gross=2 -> cap scale at 1
    W = raw.mul(scale, axis=0)

    # weekly rebalance on release-week Fridays; hold constant within the week
    idx = W.index
    W = W.resample("W-FRI").last().reindex(idx, method="ffill").fillna(0.0)
    return W


# ----------------------------------------------------------------------------- API
def load_data() -> pd.DataFrame:
    return _build_panel(SEARCH_ROOTS, START, source="databento")


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    W = _signal_weights(panel, p["long_frac"], p["short_frac"], p["vol_lb"], p["target_vol"], p["gross_cap"])
    Wlag = W.shift(1).fillna(0.0)               # built same-day -> trade next day (no look-ahead)
    rets = panel["ret"]

    daily = net_of_cost(Wlag, rets, cost_bps=8.0, name=SID)   # 8bps on rebalance turnover
    try:                                                       # + explicit per-roll cost (if roll days detected)
        roll = panel["roll"].reindex(columns=Wlag.columns).fillna(0.0)
        roll_cost = (Wlag.abs() * roll * (ROLL_BPS / 1e4)).sum(axis=1)
        daily = daily.sub(roll_cost.reindex(daily.index).fillna(0.0))
    except Exception:
        pass
    daily.name = SID

    trades = trades_from_weights(Wlag, rets, SECTOR_MAP)       # kit stamps entry_regime
    return daily, trades


def load_gen_data(label) -> pd.DataFrame:
    return _build_panel(GEN_UNIVERSES[label], START, source="yfinance")


# ----------------------------------------------------------------------------- soft expectations
def _exp_dispersion(ctx):
    try:
        hp = ctx["panel"]["hp_pct"]
        hp = hp.loc[hp.index < pd.Timestamp(ctx["holdout_start"])]
        disp = float(hp.std(axis=1).mean())
        return {"pass": bool(disp > 0.15), "observed": round(disp, 4)}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


def _exp_long_leg_premium(ctx):
    """Sign of the mechanism: low-HP long leg should out-earn high-HP short leg (in-sample)."""
    try:
        panel = ctx["panel"]; cut = pd.Timestamp(ctx["holdout_start"])
        m = panel.index < cut
        rets, hp = panel["ret"].loc[m], panel["hp_pct"].loc[m]
        rank = hp.rank(axis=1, pct=True)
        lw = (rank <= 1/3.).astype(float); lw = lw.div(lw.sum(axis=1).replace(0, np.nan), axis=0)
        sw = (rank > 2/3.).astype(float); sw = sw.div(sw.sum(axis=1).replace(0, np.nan), axis=0)
        spread = float(((lw.shift(1) * rets).sum(axis=1) - (sw.shift(1) * rets).sum(axis=1)).mean() * 252)
        return {"pass": bool(spread > 0), "observed": round(spread, 4)}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


def _exp_time_series(ctx):
    """Independent construction: long own-HP<0.3, short own-HP>0.7 should also be positive."""
    try:
        panel = ctx["panel"]; cut = pd.Timestamp(ctx["holdout_start"])
        m = panel.index < cut
        rets, hp = panel["ret"].loc[m], panel["hp_pct"].loc[m]
        pos = pd.DataFrame(0.0, index=hp.index, columns=hp.columns).mask(hp < 0.3, 1.0).mask(hp > 0.7, -1.0)
        w = (pos / rets.rolling(60, min_periods=30).std()).replace([np.inf, -np.inf], np.nan)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        r = (w.shift(1) * rets).sum(axis=1)
        sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
        return {"pass": bool(r.mean() > 0), "observed": round(sharpe, 3)}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id=SID,
    family="hedging_pressure",
    title="Commodity hedging-pressure premium (Basu-Miffre) — real COT commercial positioning",
    markets=["commodity_futures"],
    data_desc="CFTC COT commercial net positioning (release-date indexed, look-ahead pre-closed) "
              "+ Databento individual-contract futures via fut_curve (single-root adapter; roll-aware "
              "within-contract front-month returns; cross-roll gaps excluded internally). 16 CME roots, "
              "2010-2026, weekly signal / daily returns. Fully OWNED.",
    pre_registration=(
        "FROZEN, no variant selection. Per COT release: HP = commercial_net / open_interest per root, "
        "converted to a per-root trailing 52-week rolling percentile (signal = deviation in hedging "
        "demand vs own history, not the structural level). Cross-sectional: LONG lowest-HP tercile "
        "(commercials most net-short -> paid to carry inventory risk), SHORT highest-HP tercile. "
        "Inverse-60d-vol risk weight, 10% annualised vol target (de-lever only, gross<=2x), weekly "
        "rebalance on release Fridays, 1-day execution lag (W.shift(1)). Costs: 8bps on rebalance "
        "turnover; fut_curve is roll-aware (within-contract returns, cross-roll gaps excluded), so an "
        "explicit ~4bps/side roll cost is applied only on any detected roll days. fut_curve and "
        "cot_positioning are SINGLE-ROOT adapters and are called per-root then assembled. Standalone "
        "only — NO trend blend (2026-06-08 lesson: validate the premium alone before adding a sized "
        "tail-overlay). scope=broad: an insurance-provision premium is a universal positioning "
        "mechanism, so it is generalisation-tested on DISJOINT cross-asset futures (softs / currencies "
        "/ financials, sharing no tickers with the 16 commodity search roots) — within-commodity "
        "sub-slices are not disjoint from the all-16 search book, so cross-asset is the stronger "
        "universality test (2026-06-09 BAB lesson: an edge that lives in one corner is a non-"
        "generalising outlier). MCPT applies (market-neutral book -> absolute null). Checkable "
        "mechanism claims declared in expectations: (1) HP percentile has real cross-sectional "
        "dispersion; (2) the low-HP long leg out-earns the high-HP short leg (sign of the premium); "
        "(3) an independent time-series construction is also positive. Grid variants are declared ONLY "
        "for honest DSR effective-N (search burden), not for selection; default={} is primary."),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid={
        "default":   {},
        "quartile":  {"long_frac": 0.25, "short_frac": 0.25},
        "vol_lb_40": {"vol_lb": 40},
        "tv_15":     {"target_vol": 0.15},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=10,
    expectations=[
        {"name": "hp_xs_dispersion",
         "claim": "mean per-date cross-sectional std of HP percentile > 0.15 (non-degenerate signal; "
                  "roots are not all in the same tercile)",
         "check": _exp_dispersion},
        {"name": "long_leg_beats_short_leg",
         "claim": "low-HP (commercials net-short) long leg out-earns high-HP short leg in-sample "
                  "(correct sign of the insurance-provision mechanism)",
         "check": _exp_long_leg_premium},
        {"name": "time_series_construction_positive",
         "claim": "independent time-series construction (long own-HP<0.3, short own-HP>0.7) is also "
                  "positive in-sample (premium is not a pure cross-sectional artifact)",
         "check": _exp_time_series},
    ],
)