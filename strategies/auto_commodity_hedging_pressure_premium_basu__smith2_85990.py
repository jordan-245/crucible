"""
hedging_pressure_cot_xs_v1 — Commodity hedging-pressure premium (Basu–Miffre, JBF 2013).

Mechanism: speculators are paid to absorb the net hedging demand of commercial
producers/consumers (Keynes normal backwardation, measured DIRECTLY from CFTC COT
commercial positioning rather than the 12m-return proxy that hedging_pressure_footprint_ls_v2
falsified — that null killed the proxy, not the premium).

Data: cot_positioning() (RELEASE-date indexed — the adapter pre-closes look-ahead; we
additionally lag weights 1 day) for the signal; fut_curve() Databento individual contract
months for returns. ALL returns are computed WITHIN a contract (same contract symbol on both
sides of every ratio) — never across rolls. Front contract held until 5 days before roll,
then rank-2.

FIX vs previous submission: fut_curve() exposes contract identity as symbol_1/symbol_2
(no instrument_id_* columns) — within-contract return matching now keys on the contract
SYMBOL. Symbol recycling (CLZ0 etc.) happens on a DECADE scale, so day-over-day symbol
equality uniquely identifies the same live instrument; the recycling hazard applies to
long-horizon joins, not adjacent-day matching.

No side effects: this module only defines load_data / signal / load_gen_data / SPEC.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import fut_curve, cot_positioning
from sdk.signal_kit import net_of_cost, trades_from_weights

NAME = "hedging_pressure_cot_xs_v1"
START = "2010-01-01"          # grains/livestock curve coverage starts 2012-13; min_roots gate
ROLL_BUFFER = 5               # exit front contract 5 days before roll (per frozen spec)

# 16 CME roots — PA excluded (91% rank-2-thin coverage, per 2026-06-12 data note)
ROOTS = ["CL", "NG", "HO", "RB",            # energy
         "GC", "SI", "HG", "PL",            # metals
         "ZC", "ZS", "ZW", "ZL", "ZM",      # grains
         "LE", "HE", "GF"]                  # livestock

SECTOR_MAP = {"CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
              "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
              "ZC": "grains", "ZS": "grains", "ZW": "grains", "ZL": "grains", "ZM": "grains",
              "LE": "livestock", "HE": "livestock", "GF": "livestock"}

# Stage-2 generalization sub-universes (intra-sector cross-sections; see pre_registration
# for the disjointness caveat — commodities have no ticker-disjoint alternative universe).
GEN_UNIVERSES = {"energy": ["CL", "NG", "HO", "RB"],
                 "metals": ["GC", "SI", "HG", "PL"],
                 "grains_livestock": ["ZC", "ZS", "ZW", "ZL", "ZM", "LE", "HE", "GF"]}

DEFAULTS = dict(
    lookback_weeks=52,   # per-root rolling percentile window of HP (deviation, not level)
    frac=1.0 / 3.0,      # tercile sort: long bottom-k, short top-k, k = round(n/3)
    vol_lb=60,           # inverse-vol sizing lookback (daily)
    target_vol=0.10,     # 10% annualized portfolio vol target
    gross_cap=2.0,       # leverage never exceeds 2x gross
    cost_bps=8.0,        # bps on turnover (covers ~1 tick + fees on liquid CME)
    roll_cost_bps=8.0,   # extra bps * |w| charged on each held-contract switch day
    min_roots=4,         # below this cross-section width, stay flat (lets sector slices run)
    inv_vol=True,
)


# ---------------------------------------------------------------- helpers

def _fetch(fn, root):
    """Call an adapter tolerantly (some accept start=, fut_curve does NOT), then slice
    to START by date — semantics identical either way, no look-ahead implications."""
    try:
        df = fn(root, start=START)
    except TypeError:
        df = fn(root)
    df = df.sort_index()
    try:
        return df.loc[df.index >= pd.Timestamp(START)]
    except TypeError:
        return df


def _col(df, *names):
    """Resolve an adapter column across minor naming variants."""
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def _held_returns(fc):
    """Daily returns of the held contract. Hold rank-1 until days_to_roll_1 <= ROLL_BUFFER,
    then rank-2. Every return is within-contract: today's close of the held instrument over
    YESTERDAY's close of the SAME contract symbol (looked up among yesterday's rank-1/rank-2;
    adjacent-day symbol equality uniquely identifies the live instrument — recycling is a
    decade-scale hazard, not a day-over-day one).
    Returns (ret Series, roll-event flag Series — flags every held-instrument switch)."""
    fc = fc.sort_index()
    d2r = _col(fc, "days_to_roll_1", "days_to_exp_1", "dte_1").astype(float)
    c1 = _col(fc, "close_1").astype(float)
    c2 = _col(fc, "close_2").astype(float)
    s1 = _col(fc, "symbol_1", "instrument_id_1", "iid_1")
    s2 = _col(fc, "symbol_2", "instrument_id_2", "iid_2")

    on_rank2 = d2r <= ROLL_BUFFER
    held_c = c2.where(on_rank2, c1)
    held_s = s2.where(on_rank2, s1)

    p1, p2 = c1.shift(1), c2.shift(1)
    j1, j2 = s1.shift(1), s2.shift(1)
    prev = pd.Series(
        np.where(held_s.values == j1.values, p1.values,
                 np.where(held_s.values == j2.values, p2.values, np.nan)),
        index=fc.index,
    )
    ret = held_c / prev - 1.0  # NaN whenever same-contract prior close unavailable
    roll = ((held_s != held_s.shift(1)) & held_s.notna() & held_s.shift(1).notna()).astype(float)
    return ret, roll


def _build_panel(roots):
    """Flat panel: '{root}|ret' (daily within-contract held return), '{root}|roll'
    (held-instrument switch flag), '{root}|hp' (commercial net / OI, NaN off release dates —
    the RELEASE-date index is the adapter's look-ahead guarantee, kept raw here)."""
    frames = {}
    for r in roots:
        fc = _fetch(fut_curve, r)
        ret, roll = _held_returns(fc)
        frames[f"{r}|ret"] = ret
        frames[f"{r}|roll"] = roll
        c = _fetch(cot_positioning, r)  # RELEASE-date indexed
        oi = _col(c, "oi", "open_interest").astype(float).replace(0.0, np.nan)
        net = _col(c, "comm_net", "commercial_net", "net_commercial")
        frames[f"{r}|hp"] = net.astype(float) / oi
    return pd.concat(frames, axis=1).sort_index()


def load_data():
    return _build_panel(ROOTS)


def load_gen_data(label):
    return _build_panel(GEN_UNIVERSES[label])


def _hp_percentiles(hp, lookback_weeks):
    """Per-root rolling percentile of HP over its own trailing release-date observations.
    Handles the structural fact that some markets' commercials are persistently net short:
    the signal is the DEVIATION in hedging demand, not the level."""
    minp = max(20, int(lookback_weeks * 0.75))
    out = {}
    for col in hp.columns:
        v = hp[col].dropna()
        if len(v) == 0:
            out[col] = v
            continue
        out[col] = v.rolling(lookback_weeks, min_periods=minp).apply(
            lambda x: float((x <= x[-1]).mean()), raw=True)
    return pd.concat(out, axis=1).sort_index()


# ---------------------------------------------------------------- signal

def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)

    roots = sorted({c.split("|")[0] for c in panel.columns})
    rets = pd.DataFrame({r: panel[f"{r}|ret"] for r in roots})
    rolls = pd.DataFrame({r: panel[f"{r}|roll"] for r in roots}).fillna(0.0)
    hp = pd.DataFrame({r: panel[f"{r}|hp"] for r in roots})

    pct = _hp_percentiles(hp, p["lookback_weeks"])          # release-date rows only
    vol = rets.rolling(p["vol_lb"], min_periods=int(p["vol_lb"] * 0.66)).std()
    vol_at = vol.reindex(pct.index, method="ffill")

    # Rebalance ONLY on COT release dates (weekly): tercile sort on HP percentile.
    W_reb = pd.DataFrame(0.0, index=pct.index, columns=roots)
    for dt in pct.index:
        s, sv = pct.loc[dt], vol_at.loc[dt]
        ok = s.notna() & sv.notna() & (sv > 0)
        n = int(ok.sum())
        if n < p["min_roots"]:
            continue
        sr = s[ok].sort_values()
        k = max(1, int(round(n * p["frac"])))
        longs = sr.index[:k]    # commercials most net-short -> hedgers pay longs
        shorts = sr.index[-k:]  # commercials most net-long -> premium to shorts
        if p["inv_vol"]:
            wl = 1.0 / sv[longs]
            ws = 1.0 / sv[shorts]
        else:
            wl = pd.Series(1.0, index=longs)
            ws = pd.Series(1.0, index=shorts)
        w = pd.Series(0.0, index=roots)
        w[longs] = 0.5 * wl / wl.sum()
        w[shorts] = -0.5 * ws / ws.sum()
        # Ex-ante vol targeting (independence approx), gross hard-capped at 2x.
        est = float(np.sqrt(((w.abs() * sv.fillna(0.0)) ** 2).sum() * 252.0))
        lever = min(p["gross_cap"], p["target_vol"] / est) if est > 0 else 0.0
        W_reb.loc[dt] = w * lever

    # Daily weights: hold between releases; LAG 1 DAY (our responsibility) on top of the
    # adapter's release-date indexing — signal is strictly known before any return it earns.
    idx = rets.index
    W = W_reb.reindex(idx.union(W_reb.index)).ffill().reindex(idx).fillna(0.0)
    W_lag = W.shift(1).fillna(0.0)

    rets_f = rets.fillna(0.0)
    net = net_of_cost(W_lag, rets_f, cost_bps=p["cost_bps"], name=NAME)
    # Explicit roll friction: net_of_cost charges weight turnover, not contract switches.
    roll_cost = (rolls.reindex(idx).fillna(0.0) * W_lag.abs()).sum(axis=1) * (p["roll_cost_bps"] / 1e4)
    net = (net - roll_cost).rename(NAME)

    trades = trades_from_weights(W_lag, rets_f, SECTOR_MAP)  # kit stamps entry_regime
    return net, trades


# ---------------------------------------------------------------- soft expectations

def _chk_dispersion(ctx):
    """Claim: HP percentile has real cross-sectional dispersion (no degenerate sort)."""
    panel = ctx["panel"]
    cut = pd.Timestamp(ctx["holdout_start"])
    roots = sorted({c.split("|")[0] for c in panel.columns})
    hp = pd.DataFrame({r: panel[f"{r}|hp"] for r in roots})
    pct = _hp_percentiles(hp.loc[hp.index < cut], DEFAULTS["lookback_weeks"])
    rows = pct.dropna(thresh=4)
    if len(rows) == 0:
        return {"pass": False, "observed": "no valid release rows"}
    disp = float(rows.std(axis=1).median())
    return {"pass": disp >= 0.20, "observed": disp}


def _chk_slow_signal(ctx):
    """Claim: weekly structural hedger demand decays slowly -> median hold >= 10 trading days."""
    hd = [t["hold_days"] for t in ctx["trades"]]
    med = float(np.median(hd)) if hd else 0.0
    return {"pass": med >= 10.0, "observed": med}


def _chk_sector_spread(ctx):
    """Claim: the premium is not one sector's bet — >=3 sectors held, max sector <=60% of position-days."""
    if not ctx["trades"]:
        return {"pass": False, "observed": "no trades"}
    df = pd.DataFrame(ctx["trades"])
    pdays = df.groupby("sector")["hold_days"].sum()
    share = float(pdays.max() / pdays.sum())
    nsec = int((pdays > 0).sum())
    return {"pass": (share <= 0.60) and (nsec >= 3),
            "observed": f"max_sector_share={share:.2f}, sectors={nsec}"}


def _chk_lookback_insensitive(ctx):
    """Claim: mechanism not knife-edge on the 52w window — if default is positive in-search,
    the pre-declared 26w grid variant is too (uses ctx['grid'], zero extra signal() calls)."""
    g = ctx["grid"]
    d, a = g.get("default"), g.get("hp_26w")
    if d is None or a is None or len(d) == 0 or len(a) == 0:
        return {"pass": False, "observed": "missing grid series"}
    sd = float(d.mean() / (d.std() + 1e-12) * np.sqrt(252))
    sa = float(a.mean() / (a.std() + 1e-12) * np.sqrt(252))
    return {"pass": not (sd > 0 and sa <= 0), "observed": f"sharpe default={sd:.2f}, 26w={sa:.2f}"}


# ---------------------------------------------------------------- spec

PREREG = """FROZEN PRIMARY SPEC (no variant selection; grid is for DSR search-burden only):
weekly, on each COT RELEASE date, HP = commercial_net / open_interest per root; convert to a
per-root 52-week rolling percentile (signal = DEVIATION in hedging demand, not level — some
markets' commercials are structurally net short). Cross-sectional tercile sort across 16 CME
roots (PA excluded, thin rank-2): LONG lowest-HP-percentile tercile (commercials most net
short pay longs to carry inventory risk), SHORT highest. Inverse-60d-vol weights per leg,
10% ann. vol target, gross <=2x, rebalance on release dates only, weights lagged 1 further day.
Returns from individual Databento contract months, computed strictly WITHIN contracts (matched
by adjacent-day contract symbol — recycling is decade-scale, not day-over-day); front held to
5 days before roll then rank-2; 8bps on turnover + 8bps*|w| per contract-switch event.
Below 4 valid roots the book is flat (full panel effectively starts when energy+metals curves
are live; grains/livestock join 2012-13). HISTORY: hedging_pressure_footprint_ls_v2 failed on
a 12m-return PROXY — this is the canonical axis change to real positioning data (acquired
2026-06-12, $0 incremental). GENERALIZATION (scope=broad): the insurance mechanism must
reappear in intra-sector cross-sections — energy-only, metals-only, grains+livestock-only —
each sort recomputed WITHIN the slice, producing books economically distinct from the full
16-root book. Caveat declared honestly: only ~16 liquid COT-covered roots exist, so a
ticker-disjoint commodity universe is impossible; sector slices are the strongest available
out-of-construction test (a pass living in one sector only = BAB-style outlier, reject). The
proposal's time-series own-percentile variant is a DIFFERENT construction and cannot run under
the frozen-signal stage-2 battery — it stays prose, for a follow-up hypothesis if stage 1
passes. MCPT mandatory (market-neutral book -> absolute null). Soft expectations machine-check
the mechanism story: real XS dispersion, slow signal (median hold >=10d), >=3 sectors with no
sector >60% of position-days, 26w-lookback sign agreement. Deployment at ~$5K via CME
micro/minis on a reduced top/bottom book is the deployment spec, checked in deployment_sanity."""

SPEC = StrategySpec(
    id=NAME,
    family="futures_hedging_pressure",
    title="Commodity hedging-pressure premium (Basu-Miffre) — real COT commercial positioning, XS L/S on owned curve data",
    markets=ROOTS,
    data_desc="fut_curve() Databento individual contract months (within-contract returns matched on symbol_1/symbol_2, 5d roll buffer); cot_positioning() CFTC commercial net / OI, RELEASE-date indexed (look-ahead pre-closed in adapter)",
    pre_registration=PREREG,
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "hp_26w": {"lookback_weeks": 26},
        "quartile": {"frac": 0.25},
        "equal_weight": {"inv_vol": False},
    },
    scope="broad",
    generalization_universes=["energy", "metals", "grains_livestock"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=10,
    expectations=[
        {"name": "hp_dispersion",
         "claim": "median per-release-date cross-sectional std of HP percentile >= 0.20 (non-degenerate sort)",
         "check": _chk_dispersion},
        {"name": "slow_signal",
         "claim": "structural hedger demand decays slowly: median trade hold >= 10 trading days",
         "check": _chk_slow_signal},
        {"name": "sector_spread",
         "claim": ">=3 commodity sectors held; no sector exceeds 60% of position-days",
         "check": _chk_sector_spread},
        {"name": "lookback_insensitive",
         "claim": "if default (52w) is positive in-search, the pre-declared 26w variant is also positive",
         "check": _chk_lookback_insensitive},
    ],
)