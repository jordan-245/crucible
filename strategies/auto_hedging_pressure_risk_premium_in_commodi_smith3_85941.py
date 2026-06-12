"""
cot_hedging_pressure_xs_ls — Hedging-Pressure Risk Premium in Commodity Futures.

Mechanism (Keynes normal backwardation; Basu-Miffre 2013): speculators are PAID to
absorb commercial hedgers' net positioning. We go LONG roots where commercials are
most net-short (hedgers paying for insurance), SHORT where most net-long. This is a
paid-to-bear-risk premium, not a prediction. First hypothesis on the real COT
substrate (owned 2026-06-12); the failed hedging_pressure_footprint_ls_v2 used a
price-return PROXY for positioning and therefore never tested this premium.

FIX (vs failed run): the adapter returns a FLAT WIDE frame with PREFIXED columns —
'{ROOT}_comm_net', '{ROOT}_noncomm_net', '{ROOT}_oi' (e.g. 'CL_comm_net', 'CL_oi'),
DatetimeIndex on release date. The previous module's schema detection only handled
MultiIndex-wide and long (root-column) layouts, so it fell through to the long-format
branch and raised "no root/symbol column". _hp_panel now handles the prefixed-flat
layout FIRST (per-root suffix matching: comm+net excluding 'non', oi/open_interest),
keeping the MultiIndex and long-format paths as fallbacks for the gen universes.

NO-LOOK-AHEAD DESIGN:
- cot_positioning is RELEASE-date indexed (zero-Tuesday test passed at adapter build);
  HP is ffilled onto trading dates only as of release.
- Weights are formed at the Friday close of release; net_of_cost / trades_from_weights
  receive W.shift(2): weight earns its first return on the SECOND trading row after
  formation, which under the weight_t x ret_t convention means the fill is at the
  FIRST CLOSE STRICTLY AFTER the release date (e.g. Monday close for a Friday
  release; first PnL accrues Tuesday). A single shift would fill at the Friday
  (release-day) close — look-ahead for grains/livestock whose daily closes print
  BEFORE the ~3:30pm ET COT release.
- Returns are computed WITHIN contracts (the contract held since yesterday, marked at
  its own close today) — never by differencing close_1 across a roll.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fut_curve, cot_positioning
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2010-01-01"

# Search universe: 16 CME roots with full Databento curve coverage (PA excluded: 91% rank-2 thin)
ROOTS = ["CL", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
         "ZC", "ZS", "ZW", "ZL", "ZM", "LE", "HE", "GF"]

SECTOR = {
    "CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
    "ZC": "grains", "ZS": "grains", "ZW": "grains", "ZL": "grains", "ZM": "grains",
    "LE": "livestock", "HE": "livestock", "GF": "livestock",
}

# Generalization universes: DISJOINT from the 16 search roots (zero shared tickers).
# Same frozen signal + default params; mechanism is universal hedger-insurance so it
# must show up out-of-substrate or the stage-1 pass is an overfit outlier.
# Prices via yf continuous futures (FREE; roll gaps add NOISE, not directional bias —
# stated caveat: these are direction-of-effect checks, not tradable books).
GEN_UNIVERSES = {
    "ice_softs": {"KC": "KC=F", "SB": "SB=F", "CC": "CC=F", "CT": "CT=F", "OJ": "OJ=F"},
    "fx":        {"6E": "6E=F", "6J": "6J=F", "6B": "6B=F", "6A": "6A=F",
                  "6C": "6C=F", "6S": "6S=F", "6N": "6N=F", "6M": "6M=F"},
    "rates":     {"ZT": "ZT=F", "ZF": "ZF=F", "ZN": "ZN=F", "ZB": "ZB=F", "UB": "UB=F"},
}
for _label, _m in GEN_UNIVERSES.items():
    for _r in _m:
        SECTOR.setdefault(_r, _label)


# ----------------------------------------------------------------------------- data

def _front_returns(root, roll_buffer=5):
    """Within-contract daily returns for one root from the Databento curve.

    Hold rank-1 while days_to_roll_1 > roll_buffer, else rank-2. Each day's return is
    the contract HELD SINCE YESTERDAY marked at ITS OWN close today (looked up across
    rank columns, instrument-consistent) — the roll itself is a costless switch at the
    close, never a price difference across two contracts.
    """
    c = fut_curve(root).sort_index()
    use2 = c["days_to_roll_1"] <= roll_buffer
    held_sym = c["symbol_2"].where(use2, c["symbol_1"])
    held_px = c["close_2"].where(use2, c["close_1"])
    prev_sym, prev_px = held_sym.shift(1), held_px.shift(1)

    px_today = pd.Series(np.nan, index=c.index)
    for k in (1, 2, 3):
        sc, cc = f"symbol_{k}", f"close_{k}"
        if sc in c.columns and cc in c.columns:
            px_today = c[cc].where(c[sc].eq(prev_sym), px_today)
    return px_today / prev_px - 1.0


def _pick(cols, *patterns, exclude=()):
    """First column whose lowercase name contains ALL patterns and none of exclude."""
    for c in cols:
        lc = str(c).lower()
        if all(p in lc for p in patterns) and not any(e in lc for e in exclude):
            return c
    return None


def _cot_fields(cols):
    """Locate (net, long, short, oi) commercial-positioning fields across known
    CFTC schema variants. 'non' is excluded so noncommercial fields never match."""
    net = (_pick(cols, "comm", "net", exclude=("non",))
           or _pick(cols, "prod", "net") or _pick(cols, "hedg", "net"))
    lg = (_pick(cols, "comm", "long", exclude=("non",))
          or _pick(cols, "prod", "long") or _pick(cols, "hedg", "long"))
    sh = (_pick(cols, "comm", "short", exclude=("non",))
          or _pick(cols, "prod", "short") or _pick(cols, "hedg", "short"))
    oi = None
    for c in cols:
        if str(c).lower() in ("oi", "open_interest", "openinterest",
                              "oi_all", "open_interest_all"):
            oi = c
            break
    if oi is None:
        oi = _pick(cols, "open", "interest")
    return net, lg, sh, oi


def _hp_prefixed_flat(cot, roots):
    """ACTUAL adapter layout: flat wide frame, columns '{ROOT}_comm_net',
    '{ROOT}_noncomm_net', '{ROOT}_oi', DatetimeIndex on release date.
    Per-root suffix matching via _cot_fields on the root's own columns."""
    out = {}
    for r in roots:
        pref = f"{r}_"
        rcols = [c for c in cot.columns if str(c).startswith(pref)]
        if not rcols:
            continue
        suffixes = [str(c)[len(pref):] for c in rcols]
        net, lg, sh, oi = _cot_fields(suffixes)
        if net is not None:
            netf = cot[pref + str(net)].astype(float)
        elif lg is not None and sh is not None:
            netf = cot[pref + str(lg)].astype(float) - cot[pref + str(sh)].astype(float)
        else:
            continue
        if oi is not None:
            out[r] = netf / cot[pref + str(oi)].astype(float)
        elif lg is not None and sh is not None:
            out[r] = netf / (cot[pref + str(lg)].astype(float)
                             + cot[pref + str(sh)].astype(float))
        else:
            out[r] = netf  # per-root 52w z-score downstream normalizes the level
    if not out:
        return None
    return pd.DataFrame(out)


def _hp_panel(roots):
    """Hedging pressure HP = commercial net / open interest per root,
    RELEASE-date indexed, wide (columns = roots). Schema-robust field detection.
    Layouts handled, in order: prefixed-flat ('CL_comm_net'/'CL_oi' — the actual
    adapter output), MultiIndex-wide, long (root-column)."""
    cot = cot_positioning(roots, start_year=2010)

    hp = None
    if not isinstance(cot.columns, pd.MultiIndex):
        hp = _hp_prefixed_flat(cot, roots)

    if hp is None and isinstance(cot.columns, pd.MultiIndex):
        for lev in range(cot.columns.nlevels):
            vals = list(pd.unique(cot.columns.get_level_values(lev)))
            net, lg, sh, oi = _cot_fields(vals)
            if net is None and (lg is None or sh is None):
                continue
            df = cot if lev == 0 else cot.swaplevel(0, lev, axis=1)
            netf = df[net] if net is not None else df[lg] - df[sh]
            if oi is not None:
                hp = netf / df[oi]
            elif lg is not None and sh is not None:
                hp = netf / (df[lg] + df[sh])
            else:
                hp = netf
            break
        if hp is None:
            raise KeyError(
                f"COT: no commercial fields found in MultiIndex columns "
                f"{list(map(str, cot.columns[:12]))}")

    if hp is None:  # long format fallback: one root/symbol column + field columns
        df = cot.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            dcol = next(c for c in df.columns if "date" in str(c).lower())
            df = df.set_index(dcol)
        net, lg, sh, oi = _cot_fields(df.columns)
        if net is not None:
            netf = df[net].astype(float)
        elif lg is not None and sh is not None:
            netf = df[lg].astype(float) - df[sh].astype(float)
        else:
            raise KeyError(
                f"COT: no commercial fields found in columns {list(df.columns)}")
        if oi is not None:
            denom = df[oi].astype(float)
        elif lg is not None and sh is not None:
            denom = df[lg].astype(float) + df[sh].astype(float)
        else:
            denom = 1.0
        df["_hp"] = netf / denom
        rcol = next((c for c in ("root", "symbol", "ticker", "market", "code", "name")
                     if c in df.columns), None)
        if rcol is None:
            for c in df.columns:
                if df[c].dtype == object and df[c].astype(str).isin(roots).any():
                    rcol = c
                    break
        if rcol is None:
            raise KeyError(f"COT: no root/symbol column among {list(df.columns)}")
        hp = df.pivot_table(index=df.index, columns=rcol, values="_hp")

    hp.index = pd.to_datetime(hp.index)
    return hp.reindex(columns=roots).sort_index()


def _assemble(rets, roots):
    rets = rets.dropna(how="all").sort_index()
    # fill only INTERNAL gaps (after a root's first valid print) with 0.0
    rets = rets.apply(lambda s: s.loc[s.first_valid_index():].fillna(0.0)
                      if s.first_valid_index() is not None else s)
    # release-date HP -> as-of trading dates; limit=15 trading days so a stalled
    # release (e.g. 2020 COVID delay weeks) goes NaN instead of silently going stale
    hp = _hp_panel(roots).reindex(rets.index, method="ffill", limit=15)
    panel = pd.concat({"ret": rets, "hp": hp}, axis=1)
    return panel.loc[panel.index >= pd.Timestamp(START)]


def load_data():
    rets = pd.DataFrame({r: _front_returns(r) for r in ROOTS})
    return _assemble(rets, ROOTS)


def load_gen_data(label):
    m = GEN_UNIVERSES[label]
    px = yf_panel(list(m.values()), start=START)
    px = px.rename(columns={v: k for k, v in m.items()})
    return _assemble(px.pct_change(), list(m.keys()))


# --------------------------------------------------------------------------- signal

def signal(panel, n_legs=4, z_lb=52, z_min_periods=26, hysteresis=True, buffer=1,
           vol_lb=63, target_vol=0.10, gross_cap=2.0, cost_bps=8.0):
    """FROZEN: weekly HP 52w z-score; long 4 most-negative-HP roots, short 4 most-
    positive; rank hysteresis (hold until rank exits top/bottom n+1); inverse-vol
    within legs; 10% vol target; gross capped 2x; 8bps on turnover.

    Execution: weights form at the release-Friday close and are shift(2)-lagged so
    the fill is the FIRST close STRICTLY AFTER the release date (typically Monday;
    first PnL accrues Tuesday). One shift would fill AT the release-day close —
    look-ahead for grains/livestock whose daily session closes (~2:05-2:20pm ET)
    precede the ~3:30pm ET COT release.
    """
    rets, hp = panel["ret"], panel["hp"]

    hp_w = hp.resample("W-FRI").last()
    mu = hp_w.rolling(z_lb, min_periods=z_min_periods).mean()
    sd = hp_w.rolling(z_lb, min_periods=z_min_periods).std()
    z = (hp_w - mu) / sd

    volw = (rets.rolling(vol_lb, min_periods=21).std() * np.sqrt(252)).resample("W-FRI").last()

    long_set, short_set, rows = [], [], {}
    for dt in z.index:
        r = z.loc[dt].dropna()
        n = min(n_legs, len(r) // 3)  # adapts to small generalization universes
        w = pd.Series(0.0, index=rets.columns)
        if n < 1:
            rows[dt] = w
            continue
        asc = list(r.sort_values().index)          # most negative HP-z first -> LONG
        desc = asc[::-1]                           # most positive HP-z first -> SHORT
        if hysteresis:
            lkeep, skeep = set(asc[:n + buffer]), set(desc[:n + buffer])
            keep_l = [t for t in long_set if t in lkeep]
            long_set = keep_l + [t for t in asc if t not in keep_l][:max(0, n - len(keep_l))]
            keep_s = [t for t in short_set if t in skeep and t not in long_set]
            short_set = keep_s + [t for t in desc
                                  if t not in keep_s and t not in long_set][:max(0, n - len(keep_s))]
        else:
            long_set, short_set = asc[:n], desc[:n]

        v = volw.loc[dt] if dt in volw.index else pd.Series(dtype=float)
        iv = 1.0 / v.replace(0.0, np.nan)
        ivl, ivs = iv.reindex(long_set).dropna(), iv.reindex(short_set).dropna()
        if len(ivl):
            w[ivl.index] = 0.5 * ivl / ivl.sum()
        if len(ivs):
            w[ivs.index] = -0.5 * ivs / ivs.sum()
        rows[dt] = w

    Ww = pd.DataFrame(rows).T.sort_index()
    W = Ww.reindex(rets.index, method="ffill").fillna(0.0)

    # vol-target the book on its own TRAILING realized vol (unit-gross proxy returns,
    # lagged) -> leverage known at formation time t, applied via the shift below
    ru = (W.shift(1) * rets).sum(axis=1)
    lev = (target_vol / (ru.rolling(vol_lb, min_periods=21).std() * np.sqrt(252)))
    lev = lev.clip(upper=gross_cap).fillna(1.0)
    Ws = W.mul(lev, axis=0)

    # THE LAG: under the weight_t x ret_t convention used throughout this module,
    # a weight on row t implies entry at close t-1. shift(2) therefore places the
    # release-Friday weight on Tuesday's row -> entry at MONDAY's close, the first
    # close STRICTLY AFTER the release date, per the pre-registration. shift(1)
    # would imply entry at the release-day close itself (look-ahead for the 8
    # grain/livestock roots whose closes print before the 3:30pm ET release).
    Wlag = Ws.shift(2).fillna(0.0)
    smap = {c: SECTOR.get(c, "other") for c in rets.columns}
    daily = net_of_cost(Wlag, rets, cost_bps=cost_bps, name="cot_hp_xs_ls")
    trades = trades_from_weights(Wlag, rets, smap)
    return daily, trades


# --------------------------------------------------------------- soft expectations

def _check_hysteresis(ctx):
    """Mechanism claim: rank hysteresis materially extends holds (cuts churn)."""
    p = ctx["panel"]
    p = p.loc[p.index < pd.Timestamp(ctx["holdout_start"])]
    _, tr_nh = ctx["spec"].signal(p, hysteresis=False)  # the ONE allowed extra call
    med = np.median([t["hold_days"] for t in ctx["trades"]]) if ctx["trades"] else 0.0
    med_nh = np.median([t["hold_days"] for t in tr_nh]) if tr_nh else 1.0
    obs = float(med) / max(float(med_nh), 1.0)
    return {"pass": bool(obs >= 1.25), "observed": round(obs, 2)}


def _check_sector_breadth(ctx):
    """Insurance demand is economy-wide: >=3 sectors traded, none >60% of position-days."""
    pdays = {}
    for t in ctx["trades"]:
        pdays[t["sector"]] = pdays.get(t["sector"], 0) + t["hold_days"]
    tot = sum(pdays.values()) or 1
    mx = max(pdays.values()) / tot if pdays else 1.0
    return {"pass": bool(len(pdays) >= 3 and mx < 0.60),
            "observed": f"{len(pdays)} sectors, max share {mx:.2f}"}


def _check_lookback_robust(ctx):
    """A real premium can't hinge on z_lb=52: the pre-declared 104w variant is also
    positive in the search window (free — read from ctx['grid'])."""
    s = ctx["grid"].get("z104")
    obs = float(s.mean() * 252) if s is not None and len(s) else float("nan")
    return {"pass": bool(obs > 0), "observed": round(obs, 4)}


# ------------------------------------------------------------------------------ SPEC

SPEC = StrategySpec(
    id="cot_hedging_pressure_xs_ls",
    family="carry_insurance",
    title="Hedging-Pressure Premium in Commodity Futures (real CFTC COT, XS L/S, release-lagged)",
    markets=["futures"],
    data_desc=("OWNED Databento fut_curve (16 CME roots, contract-month closes, within-contract "
               "returns, roll buffer 5d) + OWNED cot_positioning (weekly commercial net/OI, "
               "RELEASE-date indexed; prefixed-flat '{ROOT}_comm_net'/'{ROOT}_oi' schema, "
               "robust field detection). Gen universes: CFTC COT for ICE softs / FX / rates "
               "with FREE yf continuous-futures prices (direction-of-effect check; roll noise "
               "stated)."),
    pre_registration=(
        "FROZEN SPEC (pre-registered before first run): each COT RELEASE date, HP = commercial "
        "net positioning / open interest per root (fields located schema-robustly; if no OI "
        "field, gross commercial long+short is the normalizer — a scale choice neutralized by "
        "the per-root z-score), 52-week rolling z-score (min 26 obs) to normalize cross-root "
        "levels. LONG the 4 most NEGATIVE HP-z roots (commercials most net-short -> hedgers "
        "paying speculators), SHORT the 4 most POSITIVE. Rank hysteresis: hold until a name "
        "exits the top/bottom 5. Inverse-vol within each leg (0.5 gross per side), 10% "
        "annualized vol target on trailing realized book vol, gross capped 2x. Weights form at "
        "the release-Friday close and are shift(2)-lagged under the weight_t x ret_t convention "
        "-> fill is at the first close STRICTLY AFTER the release date (Monday close; first PnL "
        "Tuesday). Returns are within-contract only (held contract marked at its own close; "
        "rolls switch contracts, never difference across them). Costs: 8bps on turnover via "
        "net_of_cost — a conservative proxy for the 2-ticks-per-round-turn tick-ruler estimate "
        "on these liquid roots. PRIMARY = default params, a SINGLE spec; 'no_hysteresis' and "
        "'z104' are pre-declared robustness probes only (counted in DSR effective-N), not a "
        "search. Scope is BROAD: the mechanism is universal hedger insurance, so the frozen "
        "signal must be OOS-positive on disjoint COT universes (ICE softs, FX futures, Treasury "
        "futures) or it is rejected. Sub-basket sign consistency is enforced as a "
        "machine-checkable sector-breadth expectation rather than prose. MCPT applies "
        "(market-neutral book -> absolute null). The per-root TIME-SERIES variant from the "
        "proposal is NOT machine-checkable here without re-implementing a second signal (a "
        "distinct strategy, not a check) — it is deferred to a separate pre-registered "
        "follow-up if stage 1+2 pass."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "no_hysteresis": {"hysteresis": False},
        "z104": {"z_lb": 104},
    },
    scope="broad",
    generalization_universes=["ice_softs", "fx", "rates"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "hysteresis_extends_holds",
         "claim": "median hold_days with hysteresis >= 1.25x the no-hysteresis book (search window)",
         "check": _check_hysteresis},
        {"name": "sector_breadth",
         "claim": ">=3 commodity sectors traded and no sector >60% of position-days",
         "check": _check_sector_breadth},
        {"name": "lookback_robust",
         "claim": "pre-declared 104-week z-score variant also has positive search-window ann. return",
         "check": _check_lookback_robust},
    ],
)