"""
Crypto delta-neutral funding-carry — HYSTERESIS variant.

Evolution of the elite single-threshold parent (DSR 0.9846). Identical economics:
long spot 1x / short USDT-perp 1x per coin (BTC, ETH), delta-neutral, equal-risk,
daily 00:00 UTC regime check, flat in stablecoin otherwise. ONE frozen mutation:
the parent's knife-edge 5%-ann. on/off hurdle is replaced by a hysteresis band
(enter > 10% ann. trailing-7d funding, exit < 2% ann.) plus a 7-day minimum hold,
with a 3-consecutive-negative-day fast-exit (min-hold exempt). All thresholds are
frozen at pre-registration; the parent construction is carried as a grid variant
purely for the head-to-head turnover/Sharpe diagnostic.

Book daily return = funding accrued to the short-perp leg (positive funding is
received). Basis mark-to-market is excluded (no owned daily perp-close panel in
the catalog yet); this is stated in pre_registration — basis MTM is mean-zero
over a held round-trip but adds variance, so reported vol is a lower bound.
Costs: 10 bps per leg, 2 legs traded per unit of delta-neutral notional change
=> 20 bps on |dW| turnover (40 bps gross per full round-trip, as pre-registered).

NO LOOK-AHEAD: the entry/exit state machine at date t uses funding data through
t; the resulting same-day weight matrix is shift(1)-lagged before costs/trades,
so the position only earns from t+1 onward.

FIX vs failed run: funding_rates() does not accept a `start` kwarg (and may not
accept `symbols`). We now introspect the adapter's signature, pass only kwargs
it supports, and apply the start-date cut OURSELVES by slicing the returned
frame — same data, no adapter API drift.
"""

import inspect

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, funding_rates  # funding_rates landed crucible 0be1ef7
from sdk.signal_kit import net_of_cost, trades_from_weights

COINS = ["BTC", "ETH"]
SECTOR_MAP = {"BTC": "crypto-btc", "ETH": "crypto-eth"}
COST_BPS = 20.0          # 10 bps/leg x 2 legs per unit of |dW| (spot + perp)
ANNUALIZER = 365.0       # crypto trades every day
START = "2019-01-01"

DEFAULTS = {
    "mode": "hysteresis",   # 'hysteresis' (this candidate) | 'parent' (frozen parent, diagnostic)
    "enter_ann": 0.10,      # ENTER when trailing-7d mean annualized funding > 10%
    "exit_ann": 0.02,       # EXIT when trailing-7d mean annualized funding < 2%
    "neg_days": 3,          # fast-exit on 3 consecutive negative daily funding prints
    "min_hold": 7,          # minimum holding days after entry (fast-exit exempt)
    "single_ann": 0.05,     # parent's single threshold (used only when mode='parent')
    "vol_lb": 30,           # trailing spot-vol lookback for inverse-vol split
    "fund_lb": 7,           # trailing funding mean window
}

# Module-level diagnostics side-channel (in-memory only; no file/config writes).
# signal() records turnover / per-coin nets / fast-exit dates per resolved-param key
# so the soft-expectation checks can compare variants without extra signal() calls.
_DIAG = {}


def _resolve(params):
    p = dict(DEFAULTS)
    p.update(params or {})
    return p


def _pkey(p):
    return tuple(sorted((k, p[k]) for k in DEFAULTS))


def _fetch_funding():
    """Call funding_rates() passing ONLY kwargs its signature actually supports.

    The previous run died on TypeError: unexpected keyword 'start' — the adapter
    returns its own (full) history window. We introspect rather than guess, then
    slice by date ourselves in load_data().
    """
    want = {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "start": START,
        "start_date": START,
    }
    try:
        sig_params = inspect.signature(funding_rates).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
        kwargs = {}
        if "symbols" in sig_params:
            kwargs["symbols"] = want["symbols"]
        if "start" in sig_params or has_var_kw:
            kwargs["start"] = START
        elif "start_date" in sig_params:
            kwargs["start_date"] = START
        return funding_rates(**kwargs)
    except (TypeError, ValueError):
        # Signature not introspectable or kwargs still rejected: degrade gracefully.
        for attempt in ({"symbols": want["symbols"]}, {}):
            try:
                return funding_rates(**attempt)
            except TypeError:
                continue
        return funding_rates()


def _daily_funding(fund_raw):
    """Aggregate 8h funding prints to daily UTC sums; pass through if already daily."""
    idx = pd.DatetimeIndex(fund_raw.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    f = fund_raw.copy()
    f.index = idx
    days = f.index.normalize()
    if days.duplicated().any():
        f = f.groupby(days).sum(min_count=1)
    else:
        f.index = days
    return f.sort_index()


def load_data() -> pd.DataFrame:
    """Panel: fund_BTC/fund_ETH (daily funding rate, decimal) + px_BTC/px_ETH (spot close)."""
    fund_raw = _fetch_funding()
    cols = {}
    for c in fund_raw.columns:
        cu = str(c).upper()
        if "BTC" in cu:
            cols[c] = "BTC"
        elif "ETH" in cu:
            cols[c] = "ETH"
    fund_raw = fund_raw.rename(columns=cols)
    missing = [c for c in COINS if c not in fund_raw.columns]
    if missing:
        raise ValueError(f"funding_rates() panel missing coins {missing}; columns={list(fund_raw.columns)}")
    fund = _daily_funding(fund_raw[COINS])
    fund = fund.loc[fund.index >= pd.Timestamp(START)]  # start cut applied HERE, not via adapter kwarg

    # Gate-0 trap check: full history, not a recent window (caught before on this substrate).
    first_btc = fund["BTC"].first_valid_index()
    if first_btc is None or first_btc > pd.Timestamp("2020-06-01"):
        raise ValueError(f"funding_rates BTC history starts {first_btc} — expected 2019/2020+, refusing truncated sample")

    spot = yf_panel(["BTC-USD", "ETH-USD"], start=START)
    spot = spot.rename(columns={"BTC-USD": "BTC", "ETH-USD": "ETH"})[COINS]
    spot.index = pd.DatetimeIndex(spot.index).normalize()

    panel = pd.concat(
        {"fund": fund, "px": spot.reindex(fund.index).ffill(limit=3)}, axis=1
    )
    panel.columns = [f"{a}_{b}" for a, b in panel.columns]
    return panel.dropna(how="all")


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': perpetual funding exists only in crypto perp markets — no breadth analogue.
    raise ValueError(f"local-scope strategy has no generalization universe: {label!r}")


def _state_machine(ann7, fund, p):
    """Per-coin entry/exit states. Uses data through t (caller lags by 1 day). Returns (mask, fast_exit_dates)."""
    neg_streak = (fund < 0).rolling(int(p["neg_days"])).sum() >= int(p["neg_days"])
    mask = pd.Series(0.0, index=ann7.index)
    fast_exits = []
    active, hold = False, 0
    for t in ann7.index:
        a = ann7.loc[t]
        if np.isnan(a):
            mask.loc[t] = 1.0 if active else 0.0
            continue
        if p["mode"] == "parent":
            active = bool(a > p["single_ann"])
        else:
            if active:
                hold += 1
                if bool(neg_streak.loc[t]):           # fast-exit, min-hold exempt
                    active = False
                    fast_exits.append(t.strftime("%Y-%m-%d"))
                elif hold >= int(p["min_hold"]) and a < p["exit_ann"]:
                    active = False
            else:
                if a > p["enter_ann"]:
                    active, hold = True, 0
        mask.loc[t] = 1.0 if active else 0.0
    return mask, fast_exits


def signal(panel, **params):
    p = _resolve(params)
    fund = panel[[f"fund_{c}" for c in COINS]].copy()
    fund.columns = COINS
    px = panel[[f"px_{c}" for c in COINS]].copy()
    px.columns = COINS

    ann = fund.rolling(int(p["fund_lb"])).mean() * ANNUALIZER  # trailing 7d mean, annualized
    spot_ret = px.pct_change()

    # Inverse-vol equal-risk split, vol updated weekly only (turnover hygiene).
    vol = spot_ret.rolling(int(p["vol_lb"])).std()
    iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    iv_weekly = iv.resample("W-FRI").last().reindex(fund.index, method="ffill")

    fast_exit_dates = []
    mask = pd.DataFrame(0.0, index=fund.index, columns=COINS)
    for c in COINS:
        m, fx = _state_machine(ann[c], fund[c], p)
        mask[c] = m
        fast_exit_dates += fx

    w_raw = (mask * iv_weekly).fillna(0.0)
    rs = w_raw.sum(axis=1)
    W_same_day = w_raw.div(rs.where(rs > 0, np.nan), axis=0).fillna(0.0)  # total target = 1.0 unit

    # THE LAG: same-day state -> position earns from next day.
    W = W_same_day.shift(1).fillna(0.0)

    # Position return = funding received by the short-perp leg (delta hedged by spot).
    pos_rets = fund.fillna(0.0)
    daily = net_of_cost(W, pos_rets, cost_bps=COST_BPS, name="crypto_carry_hysteresis")
    trades = trades_from_weights(W, pos_rets, SECTOR_MAP)

    # --- diagnostics for soft-expectation checks (in-memory only) ---
    per_coin = {c: net_of_cost(W[[c]], pos_rets[[c]], cost_bps=COST_BPS, name=c) for c in COINS}
    _DIAG[_pkey(p)] = {
        "turnover_total": float(W.diff().abs().sum().sum()),
        "fast_exit_dates": fast_exit_dates,
        "per_coin_net": per_coin,
        "n_entries": int((mask.diff() > 0).sum().sum()),
    }
    return daily, trades


# ---------------- soft expectation checks (machine-checkable mechanism claims) ----------------

def _sharpe(r):
    r = r.dropna()
    if len(r) < 60 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ANNUALIZER))


def _check_turnover_reduced(ctx):
    v = _DIAG.get(_pkey(_resolve({})))
    par = _DIAG.get(_pkey(_resolve({"mode": "parent"})))
    if not v or not par or par["turnover_total"] <= 0:
        return {"pass": False, "observed": "diagnostics missing"}
    ratio = v["turnover_total"] / par["turnover_total"]
    return {"pass": bool(ratio <= 0.70), "observed": round(ratio, 3)}


def _check_sharpe_ge_parent(ctx):
    g = ctx["grid"]
    sv, sp = _sharpe(g["default"]), _sharpe(g["parent_single_threshold"])
    if np.isnan(sv) or np.isnan(sp):
        return {"pass": False, "observed": "insufficient data"}
    return {"pass": bool(sv >= sp), "observed": f"variant={sv:.2f} parent={sp:.2f}"}


def _check_fast_exit_may2021(ctx):
    v = _DIAG.get(_pkey(_resolve({})))
    if not v:
        return {"pass": False, "observed": "diagnostics missing"}
    hits = [d for d in v["fast_exit_dates"] if "2021-04-15" <= d <= "2021-07-31"]
    return {"pass": bool(hits), "observed": hits[:3] if hits else "no fast-exit in May-2021 collapse"}


def _check_both_coins_positive(ctx):
    v = _DIAG.get(_pkey(_resolve({})))
    if not v:
        return {"pass": False, "observed": "diagnostics missing"}
    cut = pd.Timestamp(ctx["holdout_start"])
    obs = {}
    for c in COINS:
        r = v["per_coin_net"][c]
        obs[c] = round(float(r[r.index < cut].sum()), 4)
    return {"pass": bool(all(x > 0 for x in obs.values())), "observed": obs}


SPEC = StrategySpec(
    id="crypto_funding_carry_hysteresis_v1",
    family="carry",
    title="Crypto delta-neutral funding-carry — hysteresis band + min-hold (BTC+ETH, turnover-hardened)",
    markets=["crypto"],
    data_desc="Owned funding_rates() Binance perp funding history (BTCUSDT/ETHUSDT, 2019+), 8h prints "
              "aggregated to daily UTC; yf_panel BTC-USD/ETH-USD spot closes for inverse-vol sizing.",
    pre_registration=(
        "FROZEN evolution of the elite single-threshold funding-carry parent: identical book "
        "(long spot 1x / short USDT-perp 1x per coin, delta-neutral, inverse-vol split BTC/ETH, "
        "daily 00:00 UTC regime check, flat otherwise), ONE mutation — hysteresis entry/exit: "
        "ENTER coin when trailing-7d mean annualized funding > 10% (2x cost hurdle); EXIT when "
        "< 2% OR funding negative 3 consecutive days (fast-exit, min-hold exempt); 7-day minimum "
        "hold. All thresholds frozen, no optimization. Mechanism claim: the parent's dominant "
        "leak is threshold whipsaw (~40bps gross/round-trip at 10bps x 4 legs); the band + "
        "min-hold cuts round-trips materially without sacrificing risk-off in true funding "
        "collapses. Machine-checked: (a) variant turnover <= 70% of the parent grid variant; "
        "(b) variant net Sharpe >= parent (head-to-head, identical data) — otherwise the "
        "mutation is rejected and the parent stands; (c) fast-exit demonstrably fires in the "
        "May-2021 funding collapse (in-sample stress); (d) BTC-only and ETH-only sub-books both "
        "net-positive in-sample — a one-coin result is a fail. Costs: 20bps on |dW| turnover "
        "(= 40bps gross round-trip). Returns = funding accrued to the short-perp leg; basis "
        "mark-to-market EXCLUDED (no owned daily perp-close panel — mean-zero over a held "
        "round-trip but adds variance, so reported vol is a lower bound; not machine-checkable "
        "without perp closes). State machine uses data through t, weights shift(1)-lagged. "
        "LOCAL scope: perpetual funding exists only in crypto; validation via stage-1 gates + "
        "MCPT absolute-Sharpe null + forward paper with its own pre-registered verdict date "
        "before re-entering any carry+trend book (2026-06-10 Midas closure requirement)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "parent_single_threshold": {"mode": "parent", "single_ann": 0.05},
        "band_8_3": {"enter_ann": 0.08, "exit_ann": 0.03},
        "band_12_1": {"enter_ann": 0.12, "exit_ann": 0.01},
        "minhold_5": {"min_hold": 5},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
    expectations=[
        {"name": "turnover_reduced_vs_parent",
         "claim": "hysteresis total turnover <= 70% of the frozen parent single-threshold construction",
         "check": _check_turnover_reduced},
        {"name": "net_sharpe_ge_parent",
         "claim": "variant in-sample net Sharpe >= parent on identical data (else mutation rejected)",
         "check": _check_sharpe_ge_parent},
        {"name": "fast_exit_fires_may2021",
         "claim": "3-consecutive-negative-day fast-exit triggers during the May-2021 funding collapse",
         "check": _check_fast_exit_may2021},
        {"name": "both_coins_positive",
         "claim": "BTC-only and ETH-only sub-books are independently net-positive in-sample",
         "check": _check_both_coins_positive},
    ],
)