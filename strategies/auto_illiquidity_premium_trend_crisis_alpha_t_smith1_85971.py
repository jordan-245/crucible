"""
Illiquidity-Premium x Trend Crisis-Alpha Two-Premium Book — DEPLOYABLE-SLEEVE VARIANT
=====================================================================================
Parent design, with ONE mutation: the trend crisis-alpha hedge is rebuilt on the FIXED,
actually-tradable 5-ETF sleeve (SPY/EFA/TLT/GLD/DBC) instead of the research-grade
21-futures trend_returns() stream, so the gated verdict applies to the book that will
trade at $5K. The Amihud leg design is FROZEN (no parameter touches); the trend RULE is
the canonical 12m time-series momentum + inverse-vol, mapped onto a pre-named ETF list.

Frozen design (written before any backtest of this variant):
  - Amihud leg: small-cap sector-spread universe, 21d Amihud illiquidity, long top-40
    most-illiquid, inverse-vol sized, weekly rebalance with 4-week tranching, residual
    IWM short at 0.5x long notional (declared hedge sleeve).
  - Trend sleeve: sign(252d total return) long/short on the 5 ETFs, inverse-vol weighted,
    weekly held.
  - Combination (PRIMARY, inherited): tail overlay — trend at 25% of book risk, Amihud
    at 100%, trend leg scaled to the Amihud leg's trailing-60d vol (shift(1), no lookahead).
  - Costs 8bps on turnover. All weights lagged one day before P&L (W.shift(1) passed to kit).
Pre-registered success criteria vs the standalone (IWM-hedged) Amihud leg over the
identical search window: MaxDD reduced >=20%, Sharpe degradation <=10%, leg correlation
<= +0.10 — all declared below as machine-checkable soft expectations, plus the two
non-selectable diagnostics (canonical-trend tracking corr >= 0.5; stress-window sleeve
contribution sign).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel, trend_returns
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

# ----------------------------- frozen constants ------------------------------------
START = "2012-01-01"
HOLDOUT_START = "2022-01-01"

TREND_ETFS = ["SPY", "EFA", "TLT", "GLD", "DBC"]   # fixed BEFORE any backtest
HEDGE_ETF = "IWM"                                   # residual beta trim on the Amihud leg
ALL_ETFS = TREND_ETFS + [HEDGE_ETF]

ILLIQ_LB = 21          # Amihud rolling window (frozen, parent leg)
N_LONG = 40            # names held per weekly selection (frozen)
TRANCHE_WEEKS = 4      # tranched_v3 turnover smoothing (frozen)
VOL_LB = 60            # inverse-vol / vol-parity lookback (frozen)
TREND_LB = 252         # canonical 12m TSMOM lookback (frozen, no search)
COST_BPS = 8.0

# stress windows pre-named in the proposal; only the two pre-holdout ones are checkable
STRESS_WINDOWS = [("2015-08-01", "2016-02-29"), ("2020-02-20", "2020-04-30")]

_CACHE = {}


# ----------------------------- universe + data -------------------------------------
def _universe():
    if "univ" not in _CACHE:
        tickers, sector_map = sector_universe(marketcap="Small", top_n_per_sector=40)
        tickers = [t for t in tickers if t not in ALL_ETFS]
        sector_map = dict(sector_map)
        for t in TREND_ETFS:
            sector_map[t] = "ETF-Trend"
        sector_map[HEDGE_ETF] = "ETF-Hedge"
        _CACHE["univ"] = (tickers, sector_map)
    return _CACHE["univ"]


def load_data() -> pd.DataFrame:
    """Panel: MultiIndex columns — ('px', tkr) adj close, ('vol', tkr) share volume
    (Sharadar SEP, survivorship-clean), ('etf', tkr) ETF closes (yfinance)."""
    tickers, _ = _universe()
    px = sep_panel(tickers, START, field="closeadj")
    vol = sep_panel(tickers, START, field="volume")
    etf = yf_panel(ALL_ETFS, START)
    panel = pd.concat({"px": px, "vol": vol, "etf": etf}, axis=1)
    return panel.sort_index()


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' by design (both legs' standalone generalization is settled upstream;
    # the only new claim is book-level complementarity on tradable instruments).
    raise ValueError(f"scope='local': no generalization universe '{label}'")


# ----------------------------- leg construction ------------------------------------
def _weekly_rebal_dates(idx):
    return pd.Series(idx, index=idx).resample("W-FRI").last().dropna().values


def _amihud_weights(px, vol):
    """FROZEN parent leg: long top-N_LONG most-illiquid, inv-vol sized, 4-week tranched."""
    rets = px.pct_change()
    dollar = (px * vol).replace(0.0, np.nan)
    illiq = (rets.abs() / dollar).rolling(ILLIQ_LB, min_periods=15).mean()
    z = xs_zscore(np.log(illiq.where(illiq > 0)))          # long HIGH illiquidity
    inv_vol = 1.0 / rets.rolling(VOL_LB, min_periods=40).std()
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    rows = []
    for d in _weekly_rebal_dates(px.index):
        s = z.loc[d].dropna()
        if len(s) < 2 * N_LONG:
            rows.append(pd.Series(dtype=float, name=d))
            continue
        sel = s.nlargest(N_LONG).index
        iv = inv_vol.loc[d, sel].dropna()
        rows.append((iv / iv.sum()).rename(d) if iv.sum() > 0
                    else pd.Series(dtype=float, name=d))
    Wk = pd.DataFrame(rows).reindex(columns=px.columns).fillna(0.0)
    Wk = Wk.rolling(TRANCHE_WEEKS, min_periods=1).mean()   # tranching: avg of last 4 weekly books
    return Wk.reindex(px.index).ffill().fillna(0.0)


def _trend_weights(etf_px):
    """Canonical TSMOM rule on the fixed ETF list: sign(252d ret), inv-vol, weekly held."""
    p = etf_px[TREND_ETFS]
    r = p.pct_change()
    sig = np.sign(p / p.shift(TREND_LB) - 1.0)
    iv = (1.0 / r.rolling(VOL_LB, min_periods=40).std()).replace([np.inf, -np.inf], np.nan)
    raw = (sig * iv)
    rebal = _weekly_rebal_dates(p.index)
    raw = raw.loc[raw.index.isin(rebal)].reindex(p.index).ffill()
    gross = raw.abs().sum(axis=1)
    return raw.div(gross.where(gross > 0), axis=0).fillna(0.0)


def _build_book(panel, trend_risk_frac, hedge_ratio):
    """Returns (W_all daily target weights, rets_all). Weights are SAME-DAY built —
    caller must shift(1) before P&L (stated per contract)."""
    px, vol, etf = panel["px"], panel["vol"], panel["etf"]
    rets_stk = px.pct_change()
    rets_etf = etf.pct_change()

    W_am = _amihud_weights(px, vol)
    W_tr = _trend_weights(etf)

    # vol-parity scalar for the trend sleeve (trailing 60d, shift(1) — trailing data only)
    r_am_pre = (W_am.shift(1) * rets_stk).sum(axis=1)
    r_tr_pre = (W_tr.shift(1) * rets_etf[TREND_ETFS]).sum(axis=1)
    va = r_am_pre.rolling(VOL_LB, min_periods=40).std().shift(1)
    vt = r_tr_pre.rolling(VOL_LB, min_periods=40).std().shift(1)
    s_tr = (va / vt).replace([np.inf, -np.inf], np.nan).clip(0.2, 5.0).fillna(0.0)

    W_trend = trend_risk_frac * W_tr.mul(s_tr, axis=0)
    W_hedge = pd.DataFrame({HEDGE_ETF: -hedge_ratio * W_am.sum(axis=1)}, index=W_am.index)

    W_all = pd.concat([W_am, W_trend, W_hedge], axis=1).fillna(0.0)
    rets_all = pd.concat([rets_stk, rets_etf], axis=1)
    return W_all, rets_all


def _leg_returns(panel):
    """Pre-cost leg streams for the expectation checks (pure pandas, no extra signal())."""
    key = ("legs", id(panel))
    if key not in _CACHE:
        px, vol, etf = panel["px"], panel["vol"], panel["etf"]
        rets_stk, rets_etf = px.pct_change(), etf.pct_change()
        W_am = _amihud_weights(px, vol)
        W_tr = _trend_weights(etf)
        r_am = ((W_am.shift(1) * rets_stk).sum(axis=1)
                - 0.5 * W_am.sum(axis=1).shift(1) * rets_etf[HEDGE_ETF])  # as-deployed (IWM-hedged) leg
        r_tr = (W_tr.shift(1) * rets_etf[TREND_ETFS]).sum(axis=1)
        _CACHE[key] = (r_am.rename("amihud_leg"), r_tr.rename("etf_trend_sleeve"))
    return _CACHE[key]


# ----------------------------- signal ----------------------------------------------
def signal(panel, trend_risk_frac=0.25, hedge_ratio=0.5, **params):
    _, sector_map = _universe()
    W_all, rets_all = _build_book(panel, trend_risk_frac, hedge_ratio)

    W_lag = W_all.shift(1)  # explicit one-day lag: weights built at close t earn t+1
    daily = net_of_cost(W_lag, rets_all, cost_bps=COST_BPS, name="amihud_x_etf_trend_book")
    daily = daily.loc[daily.first_valid_index():]
    trades = trades_from_weights(W_lag, rets_all, sector_map)  # kit stamps entry_regime
    return daily, trades


# ----------------------------- soft expectations -----------------------------------
def _dd(r):
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 60 and r.std() > 0 else 0.0


def check_maxdd_reduced(ctx):
    dd_c, dd_a = _dd(ctx["search"]), _dd(ctx["grid"]["standalone_amihud"])
    ratio = dd_c / dd_a if dd_a < 0 else np.nan
    return {"pass": bool(ratio <= 0.80), "observed": round(ratio, 3)}


def check_sharpe_dilution(ctx):
    s_c, s_a = _sharpe(ctx["search"]), _sharpe(ctx["grid"]["standalone_amihud"])
    ok = (s_c >= 0.90 * s_a) if s_a > 0 else (s_c >= s_a)
    return {"pass": bool(ok), "observed": f"combined={s_c:.2f} vs standalone={s_a:.2f}"}


def check_leg_corr(ctx):
    r_am, r_tr = _leg_returns(ctx["panel"])
    cut = ctx["holdout_start"]
    c = r_am.loc[:cut].corr(r_tr.loc[:cut])
    return {"pass": bool(c <= 0.10), "observed": round(float(c), 3)}


def check_sleeve_earns_in_stress(ctx):
    r_am, r_tr = _leg_returns(ctx["panel"])
    wins, obs = 0, []
    for a, b in STRESS_WINDOWS:  # both windows end before holdout_start
        cum_tr = float((1 + r_tr.loc[a:b].fillna(0)).prod() - 1)
        cum_am = float((1 + r_am.loc[a:b].fillna(0)).prod() - 1)
        wins += int(cum_tr > 0 and cum_tr > cum_am)
        obs.append(f"{a[:7]}: sleeve={cum_tr:+.1%} vs amihud={cum_am:+.1%}")
    return {"pass": bool(wins >= 1), "observed": "; ".join(obs)}


def check_sleeve_tracks_canonical_trend(ctx):
    try:
        _, r_tr = _leg_returns(ctx["panel"])
        canon, _ = trend_returns()
        cut = ctx["holdout_start"]
        df = pd.concat([r_tr.loc[:cut], canon.loc[:cut]], axis=1).dropna()
        c = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
        return {"pass": bool(c >= 0.50), "observed": round(c, 3)}
    except Exception as e:
        return {"pass": False, "observed": f"diagnostic-error: {e}"}


# ----------------------------- spec -------------------------------------------------
SPEC = StrategySpec(
    id="amihud_x_etf_trend_deployable_v1",
    family="two_premium_combination",
    title=("Illiquidity Premium x Trend Crisis-Alpha — deployable 5-ETF trend sleeve "
           "(frozen Amihud tranched_v3 leg + canonical TSMOM on SPY/EFA/TLT/GLD/DBC, "
           "25% tail-overlay sizing)"),
    markets=["US small-cap equities", "SPY", "EFA", "TLT", "GLD", "DBC", "IWM(hedge)"],
    data_desc=("Sharadar SEP/TICKERS (owned, survivorship-clean, delisted incl.) for the "
               "Amihud leg; yfinance daily closes (free) for the 6 ETFs; $0 incremental."),
    pre_registration=(
        "FROZEN before backtest: Amihud leg untouched (21d illiq, top-40, inv-vol, weekly "
        "rebalance, 4-week tranching, 0.5x IWM residual short); trend sleeve = canonical "
        "sign(252d) TSMOM + 60d inv-vol on the pre-named 5-ETF list, NO parameter search; "
        "PRIMARY sizing = trend at 25% of book risk, vol-parity via trailing-60d (lagged). "
        "Success vs standalone (IWM-hedged) Amihud on the identical window: MaxDD -20% or "
        "better, Sharpe dilution <=10%, leg corr <= +0.10 (all machine-checked below). "
        "Non-selectable diagnostics (reported, not gated): 5-ETF sleeve must track "
        "trend_returns() at corr>=0.5 or the crisis-alpha-transfer premise is flagged weak; "
        "stress-window (2015-16, 2020) sleeve contribution sign reported. The 2022 stress "
        "window falls in holdout and is deliberately NOT checked pre-holdout. Grid is the "
        "honest 3-variant burden: primary 25% overlay, 0% (standalone control), 50% "
        "(reflexive-blend control) — declared for DSR effective-N, NOT for selection; "
        "the 25% overlay is the registered primary regardless of in-sample ranking."),
    load_data=load_data,
    signal=signal,
    default_params={"trend_risk_frac": 0.25, "hedge_ratio": 0.5},
    grid={
        "default": {},
        "standalone_amihud": {"trend_risk_frac": 0.0},
        "half_blend_control": {"trend_risk_frac": 0.5},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT_START,
    deploy_max_positions=75,  # ~40-65 tranched small-caps + 6 ETFs
    hedge_tickers=["SPY", "EFA", "TLT", "GLD", "DBC", "IWM"],
    hedge_cap=0.35,
    expectations=[
        {"name": "maxdd_reduced_20pct",
         "claim": "combined-book MaxDD <= 80% of standalone Amihud MaxDD (search window)",
         "check": check_maxdd_reduced},
        {"name": "sharpe_dilution_le_10pct",
         "claim": "combined Sharpe >= 90% of standalone Amihud Sharpe (search window)",
         "check": check_sharpe_dilution},
        {"name": "leg_corr_le_0p10",
         "claim": "daily corr(Amihud leg, ETF trend sleeve) <= +0.10 pre-holdout",
         "check": check_leg_corr},
        {"name": "sleeve_earns_in_stress",
         "claim": "ETF trend sleeve positive AND beats Amihud leg in >=1 of 2 pre-holdout "
                  "stress windows (2015-16, 2020) — crisis-alpha transfer diagnostic",
         "check": check_sleeve_earns_in_stress},
        {"name": "sleeve_tracks_canonical_trend",
         "claim": "5-ETF sleeve daily corr >= 0.5 vs validated 21-market trend_returns() "
                  "pre-holdout — if it fails, ETF-ization of crisis alpha is flagged weak",
         "check": check_sleeve_tracks_canonical_trend},
    ],
)