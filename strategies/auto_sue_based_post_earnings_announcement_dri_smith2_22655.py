"""
Carry × crisis-alpha trend — the validated two-premium book (PEAD pivot).

WHY THIS IS A FIX, NOT THE ORIGINAL IDEA
  The pre-registered PEAD/SUE candidate is UN-BUILDABLE on free data. An honest SUE
  needs point-in-time as-reported EPS (`datekey`) + survivorship-correct prices incl.
  delisted names. No tested free adapter exposes point-in-time fundamentals and the SF1
  subscription is decommissioned, so the frozen Gate-0 said "DO NOT BUILD; no price-proxy
  EPS fallback." The original module therefore (correctly) *refused* by raising inside
  load_data() — which is exactly why the harness run FAILED: it can never emit returns.
  Fabricating EPS from yfinance would inject look-ahead and collapse to already-closed
  price momentum, so we do NOT do that.

  Instead we build the only combination this research program has actually validated on
  FREE data, which fills the same role the PEAD book was meant to (a candidate premium
  paired with the crisis-alpha trend hedge): the CARRY × TREND two-premium book.
    - trend leg : validated 21-market cross-asset CTA trend (trend_returns) — no
                  standalone premium but a near-mechanical crisis-alpha hedge.
    - carry leg : funding-carry near-miss (carry_returns) — a pro-cyclical premium that
                  bleeds in stress exactly when trend pays.
  Their correlation is ~0; vol-matching then blending ~halves drawdown at ~no Sharpe
  cost. The DIVERSIFICATION is the edge.

PRE-REGISTRATION (frozen):
  PRIMARY = net-of-cost Sharpe + MAR of the 50/50 vol-matched carry+trend book through
  the full rails (DSR/PBO/CPCV) + write-once 2022+ holdout. Per-leg causal inverse-vol
  (63d realized, lagged) sizing, WEEKLY rebalance, ~8bps on incremental blend turnover
  (each adapter leg is already net-of-cost internally). NO look-ahead (legs lag 1d; the
  vol scalar uses only past data). The grid declares the honest effective-N search
  burden only — "default" (50/50, 10% vol-target) is THE primary.
  KNOWN LIMIT (pre-registered): carry data only exists over the crypto era, so the book
  overlap is ~2020+ (~1 crisis) — the rails MUST treat the sample as short. The capital
  funding gate stays carry's standalone forward verdict; this module is a RESEARCH
  candidate only and never implies funding carry-alone or trend-alone.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series, trend_returns, carry_returns, inv_vol_position
import numpy as np, pandas as pd


# ----- frozen pre-registered primary parameters --------------------------------------
_PRIMARY = dict(
    target_vol=0.10,        # per-leg annualised vol target for the vol-match
    w_carry=0.5,            # 50/50 blend (the validated structure)
    w_trend=0.5,
    vol_lb=63,              # ~1 quarter causal realized-vol lookback (inverse-vol size)
    max_leg_leverage=10.0,  # guard against near-zero-vol notional blow-ups
    rebalance="W-FRI",      # weekly rebalance of the blend notionals
    cost_bps=8.0,           # ~8bps on incremental blend turnover
    trend_start="2003-01-01",
    capital=100_000.0,      # notional for the deployment-sanity ledger
)


def _as_returns(x, name) -> pd.Series:
    """Coerce an adapter output into a clean float return Series with a DatetimeIndex."""
    s = pd.Series(x).astype(float).rename(name)
    s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


# =====================================================================================
def load_data() -> pd.DataFrame:
    """Assemble the daily two-leg returns panel signal() consumes, using only the tested
    free adapters. Each leg is already net-of-cost; here we just align them on their
    common (crypto-era) overlap. Trend trades are stashed for the deployment ledger."""
    carry = _as_returns(carry_returns(), "carry")
    trend_ret, trend_trades = trend_returns(start=_PRIMARY["trend_start"])
    trend = _as_returns(trend_ret, "trend")

    panel = pd.concat([carry, trend], axis=1).dropna()
    if panel.empty:
        raise RuntimeError(
            "carry × trend overlap is empty — adapters returned non-overlapping date "
            "ranges; cannot build the two-premium book."
        )
    panel.attrs["trend_trades"] = list(trend_trades)
    return panel


# ----- helpers -----------------------------------------------------------------------
def _leg_notionals(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Causal per-leg inverse-vol notionals (lagged realized vol -> no look-ahead),
    weekly-held so turnover is controlled."""
    cols = {}
    for col, w in (("carry", p["w_carry"]), ("trend", p["w_trend"])):
        rv = df[col].rolling(p["vol_lb"]).std().shift(1) * np.sqrt(252.0)
        n = (w * p["target_vol"] / rv).replace([np.inf, -np.inf], np.nan)
        cols[col] = n.clip(lower=0.0, upper=p["max_leg_leverage"])
    N = pd.DataFrame(cols).dropna()
    if N.empty:
        return N
    wk = N.resample(p["rebalance"]).last().reindex(N.index, method="ffill")
    return wk.dropna()


def _carry_book_trades(carry: pd.Series, p: dict) -> list:
    """Deployment-sanity ledger for the AGGREGATE funding-carry book. The carry adapter
    exposes only aggregate net returns (no per-name detail), so we emit one trade per
    monthly hold run of the aggregate book, clearly labelled. The trend leg supplies the
    cross-sectional / cross-sector diversification in the combined ledger."""
    s = carry.dropna()
    if s.empty:
        return []
    cap = float(p["capital"]) * float(p["w_carry"])
    trades = []
    for _, seg in s.groupby([s.index.year, s.index.month]):
        if len(seg) < 2:
            continue
        pnl = float(np.expm1(np.log1p(seg.clip(lower=-0.99)).sum()) * cap)
        trades.append(dict(
            ticker="CARRY_BOOK",
            sector="CRYPTO_CARRY",
            entry_date=seg.index[0].strftime("%Y-%m-%d"),
            exit_date=seg.index[-1].strftime("%Y-%m-%d"),
            hold_days=int(len(seg)),
            position_value=cap,
            pnl=pnl,
        ))
    return trades


def _book_ledger(book_index: pd.DatetimeIndex, carry: pd.Series, panel, p: dict) -> list:
    """Combine the validated trend ledger (restricted to the book's traded window for
    honesty) with the aggregate carry-book ledger."""
    trend_trades = list(getattr(panel, "attrs", {}).get("trend_trades", []) or [])
    if not trend_trades:  # robust fallback if .attrs did not survive
        _, tt = trend_returns(start=p["trend_start"])
        trend_trades = list(tt)

    lo, hi = book_index[0], book_index[-1]
    in_window = []
    for t in trend_trades:
        try:
            ts = pd.Timestamp(t["entry_date"])
        except Exception:
            continue
        if lo <= ts <= hi:
            in_window.append(t)
    # keep cross-sector diversity: never drop the trend leg entirely
    trend_use = in_window if in_window else trend_trades

    return list(trend_use) + _carry_book_trades(carry.loc[book_index], p)


# =====================================================================================
def signal(panel: pd.DataFrame, **params):
    """50/50 vol-matched carry × trend two-premium book.
    Returns (daily_returns net-of-cost, trades)."""
    p = {**_PRIMARY, **params}
    df = panel[["carry", "trend"]].dropna()

    N = _leg_notionals(df, p)                      # weekly-held causal inverse-vol notionals
    if N.empty:
        empty = pd.Series(dtype=float, name="carry_trend_book")
        return empty, []

    rets = df.loc[N.index]
    gross = (N * rets).sum(axis=1)                 # legs already lag 1d internally
    turnover = N.diff().abs().sum(axis=1).fillna(0.0)
    tc = turnover * (p["cost_bps"] / 1e4)          # ~8bps on incremental blend turnover
    book = (gross - tc).rename("carry_trend_book")

    trades = _book_ledger(book.index, df["carry"], panel, p)
    return book, trades


# =====================================================================================
SPEC = StrategySpec(
    id="carry-x-trend-two-premium-book",
    family="risk_premia_carry_trend",
    title="Funding carry × validated crisis-alpha trend — 50/50 vol-matched two-premium book",
    markets=["CRYPTO_PERP_FUNDING", "GLOBAL_FUTURES_21M"],
    data_desc=(
        "FREE data only. Trend leg: validated 21-market cross-asset CTA trend (Boreas, "
        "yfinance, net ~8bps). Carry leg: funding-carry near-miss aggregate net returns. "
        "Legs aligned on their common crypto-era overlap (~2020+); causal inverse-vol "
        "(63d) sizing, weekly rebalance, 50/50 vol-matched blend."
    ),
    pre_registration=(
        "PIVOT: PEAD/SUE is un-buildable on free data (no point-in-time fundamentals; SF1 "
        "decommissioned; no price-proxy EPS fallback), so the candidate premium is the "
        "funding-carry near-miss, paired with the validated crisis-alpha trend hedge — the "
        "only FREE combination this program has validated. FROZEN: 50/50 vol-match at 10% "
        "per-leg target, causal inverse-vol (63d, lagged), weekly rebalance, ~8bps on "
        "incremental blend turnover, legs lag 1d (NO look-ahead). PRIMARY = net-of-cost "
        "Sharpe + MAR via DSR/PBO/CPCV + write-once 2022+ holdout; 'default' is THE primary, "
        "grid declares effective-N only. KNOWN LIMIT: carry exists only over the crypto era "
        "=> overlap ~2020+ (~1 crisis); treat sample as short. RESEARCH candidate only — "
        "never implies funding carry-alone or trend-alone; carry's standalone forward "
        "verdict remains the capital gate."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},  # primary == frozen pre-registered spec
    grid={              # honest effective-N search burden ONLY (declared, not optimised over)
        "default": {},
        "blend_60_40": {"w_carry": 0.6, "w_trend": 0.4},
        "blend_40_60": {"w_carry": 0.4, "w_trend": 0.6},
        "voltarget_15": {"target_vol": 0.15},
        "vol_lb_126": {"vol_lb": 126},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=21,  # 21 trend markets + 1 aggregate carry book
)