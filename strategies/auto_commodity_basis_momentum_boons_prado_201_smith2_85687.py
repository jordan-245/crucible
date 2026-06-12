"""
Basis-momentum on the owned CME commodity contract-month substrate.

Boons & Prado (Journal of Finance, 2019): the DIFFERENCE between the momentum of the
first-nearby and the second-nearby futures contract ("basis-momentum") predicts commodity
returns cross-sectionally, beating both static basis (carry) and outright momentum. The
premium is compensation for volatility risk borne in curve dynamics — paid risk-bearing,
not a price forecast.

FROZEN SPEC (queue id 83a7fd317b20, pre-registered 2026-06-12, single experiment, NO
lookback search): per root, within-contract cumulative log return of close_1 (R1) and
close_2 (R2) over 252 trading days excluding the most recent 5; BM = R1 - R2. Month-end
cross-sectional sort: long top-4 / bottom-4 short of the 16 roots, inverse-vol sized
within leg, portfolio levered to ~10% ann. vol (cap 2x), weekly-checked holdings,
1-day execution lag, 8bps on turnover + extra 2-sided 8bps on each front-contract roll.

SCOPE = 'local' (justification): the search universe IS the entire owned commodity
cross-section (16 roots; PA excluded — 91% rank-2 coverage). No DISJOINT commodity
universe exists in owned data, so a stage-2 disjoint battery cannot be honestly built;
sector-slice sign-consistency is instead declared as a MACHINE-CHECKABLE soft
expectation, and forward validation on the 2022+ holdout is the confirmation path.
The sector panels are still exposed via load_gen_data for reporting.

All novel code below is the signal itself; lagging, costs, trade ledger and regime
stamping use the mandatory kit.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import fut_curve
from sdk.signal_kit import net_of_cost, trades_from_weights

# ---------------------------------------------------------------- universe
ROOTS = ["CL", "NG", "HO", "RB",            # energy
         "GC", "SI", "HG", "PL",            # metals
         "ZC", "ZS", "ZW", "ZL", "ZM",      # grains
         "LE", "HE", "GF"]                  # livestock   (PA EXCLUDED per DATA_CATALOG)

SECTOR = {"CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
          "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
          "ZC": "ags_livestock", "ZS": "ags_livestock", "ZW": "ags_livestock",
          "ZL": "ags_livestock", "ZM": "ags_livestock",
          "LE": "ags_livestock", "HE": "ags_livestock", "GF": "ags_livestock"}

GROUPS = {"energy":        ["CL", "NG", "HO", "RB"],
          "metals":        ["GC", "SI", "HG", "PL"],
          "ags_livestock": ["ZC", "ZS", "ZW", "ZL", "ZM", "LE", "HE", "GF"]}

MIN_OBS = 8 * 252        # gate0 pre-reg: >= 8y of front-contract history per root


# ---------------------------------------------------------------- data
def _root_frame(root):
    """Per-root frame: WITHIN-CONTRACT daily returns for rank-1 and rank-2 (groupby
    contract symbol -> no roll-jump contamination) plus a roll-day flag."""
    df = fut_curve(root, n_contracts=2).sort_index()
    r1 = df["close_1"].groupby(df["symbol_1"]).pct_change()
    r2 = df["close_2"].groupby(df["symbol_2"]).pct_change()
    roll = (df["symbol_1"] != df["symbol_1"].shift(1)).astype(float)
    if len(roll):
        roll.iloc[0] = 0.0
    return pd.DataFrame({f"{root}|r1": r1, f"{root}|r2": r2, f"{root}|roll": roll})


def _load(roots):
    frames = []
    for root in roots:
        f = _root_frame(root)
        if f[f"{root}|r1"].notna().sum() >= MIN_OBS:   # history check before inclusion
            frames.append(f)
    return pd.concat(frames, axis=1).sort_index()


def load_data() -> pd.DataFrame:
    return _load(ROOTS)


def load_gen_data(label) -> pd.DataFrame:
    """Sector sub-panels (informational under scope='local'; see module docstring)."""
    return _load(GROUPS[label])


# ---------------------------------------------------------------- signal helpers
def _roots_in(panel):
    return sorted({c.split("|")[0] for c in panel.columns})


def _field(panel, roots, f):
    return pd.DataFrame({r: panel[f"{r}|{f}"] for r in roots})


def _bm_scores(panel, lookback=252, skip=5):
    """Returns (R1 daily returns, BM score, outright-momentum score)."""
    roots = _roots_in(panel)
    R1 = _field(panel, roots, "r1")
    R2 = _field(panel, roots, "r2")
    win = lookback - skip
    mp = int(win * 0.9)
    s1 = np.log1p(R1).rolling(win, min_periods=mp).sum().shift(skip)
    s2 = np.log1p(R2).rolling(win, min_periods=mp).sum().shift(skip)
    return R1, s1 - s2, s1


def _month_end_hold(score, idx):
    """Sample score at the last trading day of each month, hold until next sample."""
    me = score.groupby(score.index.to_period("M")).tail(1)
    return me.reindex(idx).ffill()


def _signs(score, k):
    """+1 top-k / -1 bottom-k by score per date. k shrinks (n//3, floor 1) only when
    the cross-section is small (sector sub-panels); full 16-root book uses k=4."""
    ranks = score.rank(axis=1)
    n = ranks.count(axis=1)
    k_eff = (n // 3).clip(lower=1, upper=k)
    top = ranks.ge(n - k_eff + 1, axis=0)
    bot = ranks.le(k_eff, axis=0)
    return top.astype(float) - bot.astype(float)


# ---------------------------------------------------------------- signal
def signal(panel, k=4, lookback=252, skip=5, vol_lb=60, target_vol=0.10, cost_bps=8.0):
    roots = _roots_in(panel)
    R1, BM, _ = _bm_scores(panel, lookback=lookback, skip=skip)
    ROLL = _field(panel, roots, "roll").fillna(0.0)

    # monthly re-sort, held between month-ends
    bm_held = _month_end_hold(BM, panel.index)
    sign = _signs(bm_held, k)

    # inverse-vol sizing within each leg; legs +0.5 / -0.5 (gross 1.0 pre-lever)
    vol = R1.rolling(vol_lb, min_periods=vol_lb // 2).std()
    w = sign / vol
    long_w = w.clip(lower=0.0)
    short_w = (-w).clip(lower=0.0)
    long_w = long_w.div(long_w.sum(axis=1).replace(0.0, np.nan), axis=0) * 0.5
    short_w = short_w.div(short_w.sum(axis=1).replace(0.0, np.nan), axis=0) * 0.5
    W = long_w.fillna(0.0) - short_w.fillna(0.0)

    # weekly-checked holdings: snap to Friday, hold through the week
    W = W.resample("W-FRI").last().reindex(panel.index, method="ffill").fillna(0.0)

    # portfolio-level vol targeting (trailing only — lever at t uses data through t)
    raw = (W.shift(1) * R1).sum(axis=1)
    ann_vol = raw.rolling(60, min_periods=40).std() * np.sqrt(252)
    lever = (target_vol / ann_vol).clip(upper=2.0).fillna(0.0)
    W = W.mul(lever, axis=0)

    # 1-day execution lag is OURS: weights formed on day t trade at t+1
    W_lag = W.shift(1).fillna(0.0)

    net = net_of_cost(W_lag, R1, cost_bps=cost_bps, name="basis_momentum_252_5")
    # extra cost on front-contract roll days: sell old + buy new = 2 sides
    roll_cost = (W_lag.abs() * ROLL.reindex_like(W_lag).fillna(0.0)).sum(axis=1) \
        * (2.0 * cost_bps / 1e4)
    net = (net - roll_cost).rename("basis_momentum_252_5").dropna()

    sector_map = {r: SECTOR[r] for r in roots}
    trades = trades_from_weights(W_lag, R1, sector_map)   # kit stamps entry_regime
    return net, trades


# ---------------------------------------------------------------- soft expectations
def _ls_returns(score, R1, k):
    """Cheap equal-weight monthly long-short from a score panel (checks only)."""
    held = _month_end_hold(score, R1.index)
    sign = _signs(held, k)
    nl = sign.gt(0).sum(axis=1).replace(0, np.nan)
    ns = sign.lt(0).sum(axis=1).replace(0, np.nan)
    W = sign.gt(0).astype(float).div(nl, axis=0) * 0.5 \
        - sign.lt(0).astype(float).div(ns, axis=0) * 0.5
    return (W.fillna(0.0).shift(1) * R1).sum(axis=1)


def _check_orthogonal_to_momentum(ctx):
    """Claim: BM is mechanically distinct from outright front momentum (B-P Table II);
    corr(search returns, outright 252-5 momentum sort, same construction) < 0.7."""
    ho = pd.Timestamp(ctx["holdout_start"])
    R1, _, MOM = _bm_scores(ctx["panel"])
    mom = _ls_returns(MOM, R1, 4)
    mom = mom[mom.index < ho]
    j = pd.concat([ctx["search"], mom], axis=1).dropna()
    corr = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 50 else 1.0
    return {"pass": bool(corr < 0.7), "observed": round(corr, 3)}


def _check_sector_sign_consistency(ctx):
    """Claim: BM spread sign-consistent across sectors — search-window L/S cumulative
    return positive in >= 2 of 3 sector groups (sign, not magnitude)."""
    panel = ctx["panel"]
    ho = pd.Timestamp(ctx["holdout_start"])
    obs, pos = {}, 0
    for g, members in GROUPS.items():
        cols = [c for c in panel.columns if c.split("|")[0] in members]
        if len({c.split("|")[0] for c in cols}) < 3:
            continue
        R1, BM, _ = _bm_scores(panel[cols])
        r = _ls_returns(BM, R1, 4)
        r = r[r.index < ho]
        obs[g] = float(round(r.sum(), 4))
        pos += int(r.sum() > 0)
    return {"pass": bool(pos >= 2), "observed": str(obs)}


# ---------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="basis_momentum_bp2019",
    family="futures_term_structure",
    title="Commodity basis-momentum (Boons-Prado JF 2019): R1-R2 cross-sectional sort, "
          "16 owned CME roots (queue 83a7fd317b20)",
    markets=["CME commodity futures: CL NG HO RB GC SI HG PL ZC ZS ZW ZL ZM LE HE GF"],
    data_desc="OWNED Databento GLBX individual contract months via fut_curve(root, "
              "n_contracts=2): daily close_1/close_2/symbol_1/symbol_2, outright-only, "
              "decade-disambiguated, 2010+ ($0 marginal). PA excluded (91% rank-2 cover).",
    pre_registration=(
        "FROZEN single spec, no lookback search: BM = within-contract cumulative log "
        "return of nearby minus second-nearby over 252d excluding the most recent 5d. "
        "Month-end sort, long top-4 / short bottom-4 of 16 roots, inverse-vol within "
        "leg, ~10% ann. vol target (2x lever cap, trailing-only), weekly-checked "
        "holdings, 1d execution lag, 8bps on turnover + 2-sided 8bps per front roll. "
        "Mechanism: compensation for volatility risk in curve dynamics — orthogonal to "
        "static carry (level) and outright momentum (single series); registry carry "
        "FAIL and momentum tests could not compute this (no rank-2 data until "
        "2026-06-12, so zero in-house mining on this substrate). Prior medium-high "
        "(JF 2019, Qian 2025 OOS follow-up); honest caveat: post-publication decay "
        "expected, 2010+ window largely post-sample. scope='local' because the search "
        "universe is the ENTIRE owned commodity cross-section — no disjoint commodity "
        "universe exists to run an honest stage-2 battery; sector sign-consistency is "
        "machine-checked as a soft expectation instead, holdout 2022+ confirms. "
        "Expected near-zero equity beta and <0.3 corr to Boreas trend (reported at "
        "review; not machine-checked here — trend leg returns are not in ctx and a "
        "full trend_returns() rebuild exceeds the one-extra-call budget). Retail $5k "
        "deployment NOT direct (margin for 8 contracts); paper-validated leg / micro "
        "contract expression — flag at Stage-D. If PASS, ONE pre-registered variant "
        "(time-series BM>0 long-only) may be queued separately."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                # the frozen primary spec
        "topbot3": {"k": 3},          # pre-declared breadth robustness (not searched)
        "noskip":  {"skip": 0},       # pre-declared skip robustness (not searched)
    },
    scope="local",
    generalization_universes=["energy", "metals", "ags_livestock"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "orthogonal_to_outright_momentum",
         "claim": "corr(strategy, outright 252-5 front-momentum sort) < 0.7 in search window",
         "check": _check_orthogonal_to_momentum},
        {"name": "sector_sign_consistency",
         "claim": "search-window BM long-short cumulative return positive in >=2 of 3 "
                  "sector groups (energy/metals/ags+livestock)",
         "check": _check_sector_sign_consistency},
    ],
)