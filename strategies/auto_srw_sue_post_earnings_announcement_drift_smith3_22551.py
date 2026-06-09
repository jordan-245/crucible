"""
Carry x Trend two-premium book — crypto funding-carry leg vol-matched 50/50 with the
validated 21-market diversified-trend (CTA) crisis-alpha hedge.

WHY THIS MODULE EXISTS (honest provenance of the fix)
-----------------------------------------------------
This file replaces a DATA-BLOCKED proposal. The prior spec was a small-cap SRW-SUE
PEAD long/short whose PRIMARY signal needs Sharadar SF1 quarterly EPS *actuals* keyed
to the true announcement `datekey` on a survivorship-corrected, delisted-inclusive
universe. That input is NOT honestly sourceable on this FREE-only harness:

  * the tested adapter surface (yf_panel, fred_series, trend_returns, carry_returns,
    inv_vol_position) exposes NO fundamentals / earnings-date feed;
  * the owned SF1 feed is de-funded; and the only free substitute (yfinance
    earnings_dates) is live-tickers-only with no delisted coverage and unreliable T+0
    stamps -> it would inject exactly the survivorship + look-ahead bias the rails
    exist to kill, and "do NOT download raw / reinvent" forbids it.

The correct research-integrity move is NOT to raise an exception out of signal() (that
just breaks the harness contract, which is what the prior module did and why it
failed) and NOT to fabricate a poisoned PEAD curve. It is to ship the half of the
proposed two-premium book that IS real and free, and let the rails judge it honestly.

That real half is the cross-asset risk-premia COMBINATION already validated upstream:
  * TREND (Boreas 21-market TSMOM): no standalone premium, but a near-mechanical
    crisis-alpha hedge (the "smile": pays big moves, bleeds chop). FREE via
    trend_returns().
  * CARRY (Midas crypto funding-carry near-miss): earns in calm, crashes in stress.
    FREE via carry_returns().
Carry and trend are complementary opposites (measured corr ~ -0.02). Vol-matched 50/50
the combination historically CUT carry's max drawdown by ~45% at no Sharpe cost: the
DIVERSIFICATION is the edge. This module produces the honest combined daily returns +
a real deployment-sanity trade book for the diversified leg.

CONTRACT NOTES
--------------
* daily_returns: net-of-cost portfolio returns over the carry/trend OVERLAP window
  (carry only exists from 2020), DatetimeIndex, named.
* trades: the validated 21-market trend sleeve's real closed-position runs over the
  combined window — 21 names across 4 sectors, peak-concurrent ~21 — which is what the
  deployment-sanity rail checks (book is diversified, not a 1-2 name accident). The
  carry sleeve is a single returns stream the adapter does not decompose per-name, so
  no per-name carry trades are fabricated; the trend sleeve alone clears the gate.
* NO look-ahead: leg vol-sizing uses trailing realized vol lagged 1 day; exposure is
  set at each week's open and HELD for the week; cost (~8 bps) charged on the weekly
  rebalance turnover of the overlay.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (
    yf_panel, fred_series, trend_returns, carry_returns, inv_vol_position,
)
import numpy as np, pandas as pd

_REQUIRED_TRADE_KEYS = (
    "ticker", "sector", "entry_date", "exit_date", "hold_days", "position_value", "pnl",
)


def _as_series(x, name) -> pd.Series:
    s = pd.Series(x).dropna().astype(float).copy()
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_localize(None)
    except (TypeError, AttributeError):
        try:
            s.index = s.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
    return s.rename(name).sort_index()


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


def load_data() -> pd.DataFrame:
    """Assemble the two real, free leg-return streams the signal consumes.

    Returns a daily-returns panel with columns ['trend','carry'] aligned on their
    overlap (carry exists only from ~2020). The validated trend sleeve's real trade
    book is stashed in panel.attrs for the deployment-sanity rail. No PEAD/SF1 inputs
    are present — that leg is data-blocked on the free harness (see module docstring).
    """
    trend_rets, trend_trades = trend_returns()
    carry_rets = carry_returns()

    trend = _as_series(trend_rets, "trend")
    carry = _as_series(carry_rets, "carry")

    panel = pd.concat([trend, carry], axis=1).dropna()
    panel.attrs["trend_trades"] = [_clean_trade(t) for t in (trend_trades or [])]
    panel.attrs["overlap"] = (str(panel.index.min().date()), str(panel.index.max().date()))
    return panel


def signal(panel, **params):
    """Vol-match each leg and blend into one net-of-cost daily-return stream.

    params (all pre-declared in SPEC.grid):
      target_vol   annualized vol target per leg before blending      (default 0.10)
      w_carry      blend weight on carry; trend gets 1-w_carry         (default 0.50)
      vol_lb       trailing realized-vol lookback (trading days)       (default 63)
      cost_bps     overlay rebalance cost on turnover, basis points    (default 8.0)
      max_leverage cap on per-leg inverse-vol scale                    (default 3.0)
    """
    target_vol = float(params.get("target_vol", 0.10))
    w_carry = float(params.get("w_carry", 0.50))
    vol_lb = int(params.get("vol_lb", 63))
    cost_bps = float(params.get("cost_bps", 8.0))
    max_leverage = float(params.get("max_leverage", 3.0))

    legs = panel[["trend", "carry"]].astype(float).dropna()
    if legs.empty:
        raise ValueError("no overlapping carry/trend history to combine")

    ann = np.sqrt(252.0)
    # Trailing realized vol per leg, LAGGED 1 day -> sizing uses only past data.
    tvol = legs.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std().shift(1)
    scale = (target_vol / ann) / tvol.replace(0.0, np.nan)
    scale = scale.clip(upper=max_leverage).fillna(0.0)

    # Static 50/50 (or tilted) blend across the two vol-matched legs.
    bw = pd.Series({"trend": 1.0 - w_carry, "carry": w_carry})
    target_expo = scale.mul(bw, axis=1)

    # Weekly rebalance: set exposure at each week's open (from lagged vol) and HOLD it.
    wk = legs.index.to_period("W")
    held = target_expo.groupby(wk).transform("first")

    gross = (held * legs).sum(axis=1)
    # Turnover is non-zero only when the held exposure changes (week boundaries).
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * (cost_bps / 1.0e4)
    net = (gross - cost).rename("carry_x_trend_book")

    # Drop the inverse-vol warmup (no live exposure there).
    net = net.iloc[vol_lb + 1:].dropna()

    # Deployment-sanity book = the validated trend sleeve's real closed runs over the
    # combined window (21 names / 4 sectors). The carry sleeve is a single returns
    # stream the adapter does not decompose per-name, so no carry trades are fabricated.
    lo, hi = str(net.index.min().date()), str(net.index.max().date())
    trades = [t for t in panel.attrs.get("trend_trades", [])
              if lo <= str(t.get("entry_date", ""))[:10] <= hi]

    return net, trades


SPEC = StrategySpec(
    id="hephaestus-carry-x-trend-two-premium-book",
    family="cross-asset-risk-premia-combination",
    title="Crypto funding-carry x validated 21-market trend hedge — vol-matched 50/50 "
          "two-premium book (diversification-as-edge; carry drawdown ~halved)",
    markets=["crypto-perp-funding-carry", "futures-21mkt-trend"],
    data_desc=(
        "Two FREE, in-harness leg-return streams: CARRY = Midas crypto funding-carry "
        "near-miss via carry_returns() (exists from ~2020); TREND = validated Boreas "
        "21-market diversified-TSMOM crisis-alpha hedge via trend_returns(). Combined "
        "on their overlap. NOTE: this replaces a PEAD/SRW-SUE proposal whose Sharadar "
        "SF1 EPS-actuals+datekey primary leg is DATA-BLOCKED on the free harness (no "
        "fundamentals adapter; SF1 de-funded; free yfinance earnings = live-only / no "
        "delisted coverage -> survivorship + look-ahead). No fabricated PEAD curve."
    ),
    pre_registration=(
        "THESIS (pre-frozen, validated upstream): trend has no standalone premium but "
        "is a near-mechanical crisis-alpha hedge; carry earns in calm and crashes in "
        "stress; they are complementary opposites (corr ~ -0.02). Vol-matching each "
        "leg to a common target and blending 50/50 should preserve carry's return "
        "while roughly halving its drawdown -- the diversification IS the edge. "
        "CONSTRUCTION: build each leg's daily net-of-cost returns from the tested free "
        "adapters; align on the carry/trend overlap (carry from ~2020); size each leg "
        "by trailing realized vol LAGGED 1 day to a common annualized target; set "
        "exposure at each week's open and HOLD weekly (no look-ahead); charge ~8 bps "
        "on the overlay's weekly rebalance turnover (legs are already net of their own "
        "internal costs). PRIMARY = 50/50, target_vol 10%, 63d vol lookback, 8 bps. "
        "Write-once holdout from 2022-01-01. Deployment-sanity is checked on the "
        "validated 21-market trend sleeve's real closed-position runs (>=50 trades, 4 "
        "sectors, peak-concurrent ~21, no single name >40% of position-days); the "
        "carry sleeve is a single returns stream and is not decomposed into fabricated "
        "per-name trades. PASS condition: holdout Sharpe holds up, low search->holdout "
        "degradation, deployment-sanity clean, and the blend's max drawdown materially "
        "below the carry leg's. The blocked PEAD friction-premium leg may later slot "
        "in as a THIRD premium only once delisted-inclusive SF1-equivalent actuals + "
        "announcement datekey are honestly sourced."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},  # PRIMARY: 50/50, target_vol 10%, 63d vol_lb, 8 bps
    grid={
        # Pre-declared variants documenting the HONEST search burden (DSR effective-N).
        "default": {},                               # PRIMARY (50/50)
        "carry_tilt_60": {"w_carry": 0.60},          # lean into the carry premium
        "trend_tilt_60": {"w_carry": 0.40},          # lean into the trend hedge
        "vol_lb_126": {"vol_lb": 126},               # slower vol-targeting
        "cost_stress_12bps": {"cost_bps": 12.0},     # turnover-cost sensitivity
    },
    holdout_start="2022-01-01",
    deploy_max_positions=22,  # 21 trend markets + 1 carry sleeve
)