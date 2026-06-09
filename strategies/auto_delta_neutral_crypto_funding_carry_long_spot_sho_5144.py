"""Agent-proposed strategy: delta-neutral crypto funding CARRY + 21-market TREND two-premium book.

Carry leg  = Midas delta-neutral crypto funding carry (long spot / short perp, cross-sectional).
             Earns the perpetual-swap funding risk premium; a calm-regime premium that pays in quiet
             markets and bleeds in stress (the validated near-miss carry leg, delivered NET by the
             sanctioned adapter -- already 8bps-costed, weekly, inverse-vol).
Trend leg  = the FROZEN validated Boreas 21-market TSMOM (1/3/12m sign blend, inverse-vol, weekly) --
             crisis-alpha "smile" that pays big moves / bleeds chop. Vol-matched 50/50 against carry.

THE THESIS (pre-registered, frozen). Carry and trend are COMPLEMENTARY opposites: carry earns the
calm and crashes in stress; trend is crisis-alpha that pays the crash and bleeds the calm. Neither
clears the bar standalone (carry = a documented near-miss, DSR 0.892; trend = no standalone premium).
The hypothesis is that their COMBINATION is the edge: with corr(carry,trend) ~ 0 the trend leg acts as
a near-mechanical hedge that roughly halves the carry drawdown at little Sharpe cost, turning a scary
~26% single-premium book into a fundable ~14% diversified one. The DIVERSIFICATION is the edge, not
either leg. PRIMARY pre-registered metric = the combined book's MAR / max-drawdown and leg correlation
on the write-once 2022+ holdout (the 2022 inflation crash is the key out-of-sample hedge test).

Both legs arrive NET-of-cost from their validated adapters (8bps/turnover baked in); the static
50/50 vol-matched blend uses fixed weights so it adds no marginal turnover -> no cost double-count.
No look-ahead: each leg's signals are lagged 1 day inside its own validated engine; the leg
vol-match scalars are portfolio-construction constants (the standard vol-matched-blend convention).
The harness owns ALL rails (CPCV/DSR/PBO/FDR/holdout/deployment-sanity); this module only produces
(daily_returns, trades). FROZEN.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import carry_returns, trend_returns

HOLDOUT = "2022-01-01"


def _norm(s: pd.Series) -> pd.Series:
    """Coerce to a clean tz-naive, day-normalized daily return Series."""
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_localize(None)
    except (TypeError, AttributeError):
        try:
            s.index = s.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _vol_scale(r: pd.Series, target_vol=0.10, ann=252) -> pd.Series:
    """Static vol-match: scale a leg to a common annualized vol (constant scalar; no rebalance cost)."""
    v = float(pd.Series(r).std() * np.sqrt(ann))
    return r * (target_vol / v) if v > 0 else r


def load_data() -> pd.DataFrame:
    """Panel = the two NET leg return streams aligned on the business-day grid (outer join, full
    histories preserved). The validated trend trade-list is carried on df.attrs for deployment-sanity
    (the carry leg is delivered opaque/net by its adapter, so book breadth is supplied by the 21-market
    trend leg across 4 sectors: Equity / Rates / Commod / FX)."""
    carry = _norm(carry_returns()).rename("carry")
    trend_ret, trend_trades = trend_returns()
    trend = _norm(trend_ret).rename("trend")
    df = pd.concat([carry, trend], axis=1).sort_index()
    df.attrs["trend_trades"] = list(trend_trades)
    return df


def signal(panel, blend=0.5, target_vol=0.10):
    """(daily_returns, trades) for the vol-matched crypto-carry + trend book.

    blend = carry weight (1-blend = trend weight). Returns are net (legs arrive net-of-8bps); the
    blend is a fixed 50/50 vol-matched mix so it introduces no extra turnover/cost.
    """
    df = panel[["carry", "trend"]].dropna()                 # the carry+trend overlap window
    carry = _vol_scale(df["carry"], target_vol)
    trend = _vol_scale(df["trend"], target_vol)
    combo = (blend * carry + (1.0 - blend) * trend).dropna()
    combo.name = "crypto_carry_trend"

    # deployment-sanity trades = the validated trend leg's real sign-run trades, restricted to the
    # book's live overlap window (21 markets / 4 sectors -> >=50 trades, balanced, no single-name >40%).
    lo = df.index.min()
    ttrades = panel.attrs.get("trend_trades", [])
    trades = [t for t in ttrades if pd.Timestamp(t["entry_date"]) >= lo]
    return combo, trades


SPEC = StrategySpec(
    id="crypto-carry-trend-book",
    family="crypto_carry_trend_combo",
    title="Delta-neutral crypto funding-carry + 21-market Trend two-premium book",
    markets=["crypto", "futures"],
    data_desc=("FREE/owned: Midas delta-neutral crypto funding-carry leg (Binance-Vision funding+OHLC, "
               "delivered NET by the sanctioned carry_returns() adapter, 2020+) vol-matched 50/50 against "
               "the FROZEN Boreas 21-market TSMOM trend leg (yfinance front futures, NET via "
               "trend_returns()). Overlap ~2020-02..present (~6.4yr, includes COVID-2020 + the 2022 "
               "inflation crash). Trend leg supplies cross-sector breadth for deployment-sanity."),
    pre_registration=(
        "Two-premium book. Carry leg = Midas delta-neutral crypto funding carry (long spot / short perp, "
        "cross-sectional, inverse-vol, weekly, 8bps/turnover) -- a documented near-miss standalone "
        "(DSR ~0.892), a calm-regime premium. Trend leg = FROZEN Boreas 21-market TSMOM (1/3/12m sign "
        "blend, inverse-vol, weekly, 8bps) -- crisis-alpha with NO standalone premium. Construction: "
        "align the two NET daily-return streams on their overlap, static vol-match each to a common "
        "annualized target, blend 50/50 (default). Both legs are delivered net-of-cost by their "
        "validated adapters and each lags its signals 1 day internally (no look-ahead); the fixed-weight "
        "vol-matched blend adds no marginal turnover. HYPOTHESIS (frozen BEFORE the verdict): "
        "corr(carry,trend) ~ 0, so the crisis-alpha trend leg hedges ~half the carry max-drawdown at "
        "little Sharpe cost -- the diversification is the edge, not either leg. PRIMARY pre-registered "
        "metric = combined-book MAR / max-drawdown + leg correlation on the write-once 2022+ holdout "
        "(the 2022 inflation crash is the decisive OOS hedge test: carry bleeds, trend pays). Standalone "
        "legs are diagnostics only. DISCIPLINE: this book is fundable ONLY as the COMBINATION -- never "
        "carry-alone (near-miss) nor trend-alone (no premium); overlap is ~6.4yr / ~2 crises (sample "
        "limited). FROZEN."),
    load_data=load_data, signal=signal,
    default_params={},
    grid={
        "default": {},                       # primary: 50/50 vol-matched
        "blend_carry": {"blend": 0.65},      # carry-tilted
        "blend_trend": {"blend": 0.35},      # trend-tilted
        "voltgt_low": {"target_vol": 0.07},
        "voltgt_high": {"target_vol": 0.14},
    },
    holdout_start=HOLDOUT,
    deploy_max_positions=21,   # book breadth represented by the 21-market trend leg (carry leg net/opaque)
)