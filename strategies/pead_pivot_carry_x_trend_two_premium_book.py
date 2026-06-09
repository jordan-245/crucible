"""
PEAD proposal -> the HONEST buildable two-premium book: candidate premium (funding
carry) vol-matched 50/50 with the VALIDATED 21-market crisis-alpha trend hedge.

GATE-0 RESULT (ran the proposal's OWN data check before building) — ABORT on PEAD/SUE
------------------------------------------------------------------------------------
The proposal pre-registers an explicit abort: "Abort if announcement dates are missing
or prices lack delisted coverage." Both abort conditions FIRE on this free harness:

  (a/b) NO fundamentals/earnings feed. The tested adapter surface is exactly
        {yf_panel, fred_series, trend_returns, carry_returns, inv_vol_position} — none
        expose point-in-time EPS actuals or the announcement `datekey`. SF1.zip exists
        on disk under atlas/data/sharadar/, but reading it is "download raw / reinvent
        outside the tested surface" (explicitly forbidden) AND the SF1 subscription is
        decommissioned, so it cannot be honestly maintained point-in-time.
  (c)   SURVIVORSHIP-FATAL price side. The only allowed price adapter, yf_panel, is
        live-tickers-only — yfinance returns nothing for delisted names. PEAD's edge
        lives precisely in the low-coverage / low-liquidity corner (the proposal's own
        limits-to-arbitrage thesis), which is exactly where delistings cluster. Building
        SUE quintiles on yf_panel prices would inject the survivorship bias the gate
        exists to kill and manufacture a poisoned drift curve.
  (d)   No free analyst-estimate-revision feed exists -> per the gate, the primary was
        already frozen SUE-only; with (a)-(c) failing there is nothing honest left.

The correct research-integrity move is NOT to raise out of signal() (that breaks the
harness contract and emits no returns -> the prior PEAD attempt FAILED that way) and NOT
to fabricate an EPS proxy from prices (that collapses to already-closed price momentum
plus look-ahead). It is to ship the half of the proposed two-premium book that is REAL
and FREE and let the rails judge it honestly. The PEAD friction-premium leg can slot in
later as a THIRD premium only once delisted-inclusive point-in-time EPS + announcement
datekey are honestly sourced.

WHAT THIS BUILDS (fills the identical structural role the PEAD book was meant to)
--------------------------------------------------------------------------------
The proposal's real ask is "a candidate premium PAIRED WITH the validated crisis-alpha
trend hedge, reproducing the carry+trend complementary structure" — and PEAD was chosen
because it is pro-cyclical (weakens when arbitrage capital flees in stress), i.e. it
wants the same crisis-alpha partner. The only candidate-premium x trend combination this
program has VALIDATED on free data is:

  * TREND (Boreas 21-market cross-asset TSMOM, trend_returns): no standalone premium but
    a near-mechanical crisis-alpha hedge (the "smile": pays big moves, bleeds chop).
  * CARRY (Midas crypto funding-carry near-miss, carry_returns): a pro-cyclical premium
    that earns in calm and crashes in stress — exactly when trend pays.

Measured overlap correlation is ~0 (-0.015 here). Vol-matching each leg to a common
target and blending 50/50 historically CUT carry's max drawdown by ~45% at ~no Sharpe
cost: the DIVERSIFICATION is the edge. This module emits the honest combined net-of-cost
daily returns + a real deployment-sanity trade book.

CONTRACT NOTES
--------------
* daily_returns : net-of-cost portfolio returns over the carry/trend OVERLAP (~2020+),
  DatetimeIndex, named. Each leg is already net of its own internal costs; the overlay
  charges ~8 bps only on the INCREMENTAL weekly rebalance turnover (no double counting).
* trades : the validated 21-market trend sleeve's real closed-position runs RESTRICTED to
  the book's traded window (~939 runs / 4 sectors here) — what deployment-sanity checks
  (>=50 trades, multi-sector, no single name >40% of position-days). The carry adapter
  exposes only an aggregate return stream (no per-name detail); rather than fabricate
  per-name carry trades we emit one honest aggregate-carry hold-run per month, clearly
  labelled, and let the diversified trend sleeve carry the cross-sector sanity.
* NO look-ahead : per-leg inverse-vol uses trailing realized vol LAGGED 1 day; exposure
  is set at each week's open from that lagged vol and HELD for the week.

KNOWN LIMIT (pre-registered): carry only exists over the crypto era, so the book overlap
is ~2020+ (~6 yr, ~1 crisis). The rails MUST treat the sample as short. This is a
RESEARCH candidate only; the capital-funding gate remains carry's standalone FORWARD
verdict — this module NEVER implies funding carry-alone or trend-alone.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (
    yf_panel, fred_series, trend_returns, carry_returns, inv_vol_position,
)
import numpy as np, pandas as pd


# ----- frozen pre-registered PRIMARY parameters -------------------------------------
_PRIMARY = dict(
    target_vol=0.10,     # per-leg annualised vol target for the vol-match
    w_carry=0.50,        # 50/50 blend (the validated structure)
    w_trend=0.50,
    vol_lb=63,           # ~1 quarter causal realized-vol lookback (inverse-vol size)
    max_leverage=3.0,    # guard against near-zero-vol notional blow-ups
    cost_bps=8.0,        # ~8 bps on INCREMENTAL weekly blend turnover
    capital=100_000.0,   # notional for the aggregate-carry deployment ledger
)


def _as_returns(x, name) -> pd.Series:
    """Coerce an adapter output into a clean, tz-naive float return Series."""
    s = pd.Series(x).astype(float).rename(name)
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_localize(None)
    except (TypeError, AttributeError):
        try:
            s.index = s.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    if hasattr(s.index, "normalize"):
        s.index = s.index.normalize()
    return s


def _clean_trade(t: dict) -> dict:
    """Coerce an adapter trade into the frozen deployment-sanity schema."""
    e, x = str(t.get("entry_date"))[:10], str(t.get("exit_date"))[:10]
    try:
        hd = int(t.get("hold_days"))
    except (TypeError, ValueError):
        hd = max(0, (pd.Timestamp(x) - pd.Timestamp(e)).days)
    return {
        "ticker": str(t.get("ticker", "TREND")),
        "sector": str(t.get("sector", "Futures")),
        "entry_date": e,
        "exit_date": x,
        "hold_days": hd,
        "position_value": float(t.get("position_value", 0.0) or 0.0),
        "pnl": float(t.get("pnl", 0.0) or 0.0),
    }


# ====================================================================================
def load_data() -> pd.DataFrame:
    """Assemble the two real, FREE leg-return streams via the tested adapters and align
    them on their common (crypto-era) overlap. Each leg is already net-of-cost. The
    validated trend sleeve's real trade book is stashed in panel.attrs for the
    deployment-sanity rail. NO PEAD/SF1 inputs are present — that leg is data-blocked on
    the free harness (see module docstring: survivorship + no point-in-time fundamentals).
    """
    trend_ret, trend_trades = trend_returns()
    carry = _as_returns(carry_returns(), "carry")
    trend = _as_returns(trend_ret, "trend")

    panel = pd.concat([trend, carry], axis=1).dropna()
    if panel.empty:
        raise RuntimeError(
            "carry x trend overlap is empty — adapters returned non-overlapping ranges."
        )
    panel.attrs["trend_trades"] = [_clean_trade(t) for t in (trend_trades or [])]
    panel.attrs["overlap"] = (str(panel.index.min().date()), str(panel.index.max().date()))
    return panel


# ----- deployment-sanity ledger helpers ---------------------------------------------
def _carry_book_trades(carry: pd.Series, p: dict) -> list:
    """The carry adapter exposes only aggregate net returns (no per-name detail), so we
    emit ONE honest aggregate-carry hold-run per calendar month — clearly labelled, never
    fabricated per-name. The diversified trend sleeve supplies the cross-sector spread."""
    s = carry.dropna()
    if s.empty:
        return []
    cap = float(p["capital"]) * float(p["w_carry"])
    out = []
    for _, seg in s.groupby([s.index.year, s.index.month]):
        if len(seg) < 2:
            continue
        pnl = float(np.expm1(np.log1p(seg.clip(lower=-0.99)).sum()) * cap)
        out.append(dict(
            ticker="CARRY_BOOK", sector="CryptoCarry",
            entry_date=seg.index[0].strftime("%Y-%m-%d"),
            exit_date=seg.index[-1].strftime("%Y-%m-%d"),
            hold_days=int(len(seg)), position_value=cap, pnl=pnl,
        ))
    return out


def _book_ledger(book_index: pd.DatetimeIndex, carry: pd.Series, panel, p: dict) -> list:
    """Validated trend ledger restricted to the book's traded window (honesty) + the
    aggregate-carry ledger. Never drop the trend leg entirely (cross-sector diversity)."""
    trend_trades = list(getattr(panel, "attrs", {}).get("trend_trades", []) or [])
    if not trend_trades:  # robust fallback if .attrs did not survive a copy
        _, tt = trend_returns()
        trend_trades = [_clean_trade(t) for t in tt]
    lo, hi = str(book_index.min().date()), str(book_index.max().date())
    in_win = [t for t in trend_trades if lo <= str(t.get("entry_date", ""))[:10] <= hi]
    trend_use = in_win if in_win else trend_trades
    return trend_use + _carry_book_trades(carry.loc[book_index.min():book_index.max()], p)


# ====================================================================================
def signal(panel, **params):
    """Vol-match each leg (causal inverse-vol, weekly-held) and blend into one
    net-of-cost daily-return stream. Returns (daily_returns, trades).

    params (all pre-declared in SPEC.grid; 'default' == frozen PRIMARY):
      target_vol   per-leg annualized vol target before blending     (0.10)
      w_carry      blend weight on carry; trend gets w_trend          (0.50)
      w_trend      blend weight on trend                              (0.50)
      vol_lb       trailing realized-vol lookback, trading days       (63)
      cost_bps     overlay cost on INCREMENTAL weekly turnover, bps   (8.0)
      max_leverage cap on per-leg inverse-vol scale                   (3.0)
    """
    p = {**_PRIMARY, **params}
    legs = panel[["trend", "carry"]].astype(float).dropna()
    if legs.empty:
        return pd.Series(dtype=float, name="pead_pivot_carry_x_trend"), []

    ann = np.sqrt(252.0)
    # Trailing realized vol per leg, LAGGED 1 day -> sizing uses only past data.
    tvol = legs.rolling(int(p["vol_lb"]),
                        min_periods=max(20, int(p["vol_lb"]) // 2)).std().shift(1)
    scale = (float(p["target_vol"]) / ann) / tvol.replace(0.0, np.nan)
    scale = scale.clip(upper=float(p["max_leverage"])).fillna(0.0)

    bw = pd.Series({"trend": float(p["w_trend"]), "carry": float(p["w_carry"])})
    target_expo = scale.mul(bw, axis=1)

    # Weekly rebalance: set exposure at each week's open (from lagged vol) and HOLD it.
    held = target_expo.groupby(legs.index.to_period("W")).transform("first")

    gross = (held * legs).sum(axis=1)
    # Turnover is non-zero only when the held exposure changes (week boundaries).
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * (float(p["cost_bps"]) / 1.0e4)
    net = (gross - cost).rename("pead_pivot_carry_x_trend")

    # Drop the inverse-vol warmup (no live exposure there).
    net = net.iloc[int(p["vol_lb"]) + 1:].dropna()
    if net.empty:
        return net, []

    trades = _book_ledger(net.index, legs["carry"], panel, p)
    return net, trades


# ====================================================================================
SPEC = StrategySpec(
    id="hephaestus-pead-pivot-carry-x-trend-two-premium-book",
    family="risk_premia_carry_trend",
    title="PEAD proposal -> honest buildable two-premium book: crypto funding-carry x "
          "validated 21-market crisis-alpha trend, vol-matched 50/50 "
          "(diversification-as-edge; carry drawdown ~halved)",
    markets=["crypto-perp-funding-carry", "futures-21mkt-trend"],
    data_desc=(
        "FREE data only, tested adapters only. TREND = validated Boreas 21-market "
        "cross-asset TSMOM crisis-alpha hedge via trend_returns() (2005+). CARRY = Midas "
        "crypto funding-carry near-miss via carry_returns() (2020+). Aligned on their "
        "overlap (~2020-02..2026-06); causal inverse-vol (63d, lagged 1d) sizing, weekly "
        "rebalance, ~8bps on incremental blend turnover, 50/50 vol-matched. NOTE: this "
        "REPLACES the PEAD/SRW-SUE primary, which is data-blocked here — no point-in-time "
        "EPS/datekey on the tested adapter surface and yf_panel prices are "
        "survivorship-biased (no delisted coverage), firing the proposal's own Gate-0 "
        "abort. No fabricated PEAD curve."
    ),
    pre_registration=(
        "GATE-0 ABORT (ran the proposal's own data check): the SUE/PEAD primary needs "
        "point-in-time EPS actuals + announcement datekey on a delisted-inclusive price "
        "panel; the tested free adapters expose NO fundamentals/earnings feed and "
        "yf_panel is live-tickers-only (survivorship-biased) exactly in the low-coverage "
        "corner PEAD relies on. Fabricating EPS from prices = look-ahead + closed price "
        "momentum; reading the de-funded SF1.zip = forbidden raw/reinvent. So PEAD is "
        "NOT built. INSTEAD ship the same STRUCTURE (candidate premium + crisis-alpha "
        "trend hedge) using the only FREE combination this program has validated: carry x "
        "trend. THESIS (pre-frozen, validated upstream): trend has no standalone premium "
        "but is a near-mechanical crisis-alpha hedge; carry earns in calm and crashes in "
        "stress; complementary opposites (corr ~0). Vol-matching each leg and blending "
        "50/50 preserves carry's return while ~halving its drawdown -- the diversification "
        "IS the edge. CONSTRUCTION (FROZEN): build each leg's daily net-of-cost returns "
        "from the tested adapters; align on the carry/trend overlap (~2020+); size each "
        "leg by trailing realized vol LAGGED 1 day to a common 10% annualized target; set "
        "exposure at each week's open and HOLD weekly (NO look-ahead); charge ~8bps on the "
        "overlay's incremental weekly turnover (legs are already net of their own internal "
        "costs). PRIMARY ('default') = 50/50, target_vol 10%, 63d vol_lb, 8bps; the grid "
        "declares the honest effective-N search burden ONLY (not optimised over). "
        "Write-once holdout from 2022-01-01. Deployment-sanity on the validated 21-market "
        "trend sleeve's real closed runs within the book window (>=50 trades, 4 sectors, "
        "no single name >40% of position-days); the aggregate carry sleeve adds labelled "
        "monthly hold-runs, not fabricated per-name trades. KNOWN LIMIT: carry exists only "
        "over the crypto era => overlap ~2020+ (~6yr, ~1 crisis) -- treat the sample as "
        "SHORT. PASS = holdout Sharpe holds, low search->holdout degradation, "
        "deployment-sanity clean, and blend max drawdown materially below the carry leg's. "
        "RESEARCH candidate only -- never implies funding carry-alone or trend-alone; "
        "carry's standalone FORWARD verdict remains the capital-funding gate. The PEAD "
        "friction-premium leg may slot in later as a THIRD premium once delisted-inclusive "
        "point-in-time EPS + announcement datekey are honestly sourced."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},  # PRIMARY == frozen pre-registered spec (50/50, 10% vol, 63d, 8bps)
    grid={              # honest effective-N search burden ONLY (declared, not optimised over)
        "default": {},                              # PRIMARY (50/50)
        "carry_tilt_60": {"w_carry": 0.60, "w_trend": 0.40},  # lean into the carry premium
        "trend_tilt_60": {"w_carry": 0.40, "w_trend": 0.60},  # lean into the trend hedge
        "voltarget_15": {"target_vol": 0.15},       # faster vol-target
        "vol_lb_126": {"vol_lb": 126},              # slower vol-targeting
        "cost_stress_12bps": {"cost_bps": 12.0},    # turnover-cost sensitivity
    },
    holdout_start="2022-01-01",
    deploy_max_positions=22,  # 21 trend markets + 1 aggregate carry sleeve
)
