"""Agent-proposed strategy: Credit-spread carry + Trend two-premium book (NEW pro-cyclical leg).

Carry leg  = CREDIT / default-risk premium. Long HYG (high-yield corporate ETF) financed by a SHORT,
             duration-matched Treasury (IEI 3-7yr). The hedge ratio is FROZEN ex-ante from current ETF
             effective durations (HY ~3.5yr / IEI ~4.5yr -> 0.78) to strip the rates/term factor and
             isolate the credit-spread return (compensation for systematic default + illiquidity risk —
             the documented "credit-spread puzzle"). PRO-CYCLICAL: earns the spread in calm regimes and
             crashes when spreads blow out in stress. Dividend-ADJUSTED (total-return) series are used
             because the COUPON IS THE CARRY — un-adjusted Close would zero the entire premium.
Trend leg  = the FROZEN validated Boreas 21-market cross-asset TSMOM (crisis-alpha "smile" that pays the
             big moves / bleeds chop). Vol-matched 50/50 against the credit leg.

THE THESIS (pre-registered, FROZEN). Credit-carry and trend are COMPLEMENTARY opposites: credit earns the
calm and crashes exactly when spreads gap out (GFC-2008, COVID-2020, 2022); trend is crisis-alpha that
pays those same tails. With corr(credit,trend) ~ 0 the trend leg acts as a near-mechanical hedge that cuts
the credit drawdown (the analogue of the crypto-carry 26.5%->14.5% DD-cut result) at little Sharpe cost.
The DIVERSIFICATION is the edge, not either standalone leg. PRIMARY pre-registered metric = the combined
book's MAR / max-drawdown and leg correlation on the write-once 2022+ holdout (the 2022 spread-widening is
the key OOS hedge test), via the standard CPCV/DSR/PBO/FDR rails.

WHY-NOT-DUPLICATE. The credit/default premium is absent from prior tests: crypto funding-carry (Midas
near-miss), DM FX+bond TERM carry (CLOSED), commodity term-structure carry (FAIL). Credit-spread carry is
compensation for systematic default/illiquidity risk — structurally distinct from bond roll/term carry and
from cross-sectional equity factors. Pairs with the validated trend hedge; also a pro-cyclical candidate
for the multi-premium book if the crypto-carry forward verdict (2026-08-28) fails.

GATE-0 / DATA NOTE (honest, frozen BEFORE the verdict). yf_panel returns UNADJUSTED Close, which would
zero the coupon — fatal for a carry book — so the credit leg loads dividend-ADJUSTED (auto_adjust=True)
total-return Close via yfinance directly (the same direct-yfinance pattern vrp_trend.py uses). HYG inception
Apr-2007 covers the full GFC; IEI/IEF/LQD reach 2007. The frozen 0.78 HY/IEI duration hedge ratio is locked
from ETF fact-sheet effective durations. Both legs' signals are lagged 1 day (no look-ahead); 8bps frozen
cost on (1+h)*turnover (both pair legs; deliberately conservative vs real ETF spread+commission). The
harness owns ALL rails; this module only produces (daily_returns, trades). FROZEN.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import trend_returns, inv_vol_position

COST = 8.0 / 1e4          # 8bps per unit turnover (same frozen, conservative micro-cost as Boreas)
START = "2007-01-01"
TICKERS = ["HYG", "LQD", "IEI", "IEF"]   # credit (HY/IG) + duration-matched Treasuries (3-7y / 7-10y)


def load_data() -> pd.DataFrame:
    """Panel = dividend-ADJUSTED (total-return) Close for the credit + Treasury ETFs on a B-day grid.
    Uses yfinance auto_adjust=True directly because yf_panel's UNADJUSTED Close would strip the coupon
    (the carry itself). The 21-market trend leg is loaded internally in signal() (frozen adapter)."""
    import yfinance as yf
    raw = yf.download(TICKERS, start=START, progress=False, auto_adjust=True, group_by="ticker")
    cols = {}
    for t in TICKERS:
        s = None
        try:
            s = raw[t]["Close"].dropna() if t in raw.columns.get_level_values(0) else None
        except Exception:
            s = None
        if s is None:
            try:
                c = raw["Close"]; s = (c[t] if t in getattr(c, "columns", []) else c).dropna()
            except Exception:
                s = None
        if s is not None and len(s) > 200:
            cols[t] = s
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.to_datetime(px.index).normalize()
    bidx = pd.date_range(px.index.min(), px.index.max(), freq="B")
    return px.reindex(bidx).ffill(limit=3)


def _vol_scale(r, tgt=0.10, ann=252):
    """Static vol-match to a common annualized vol (a portfolio-construction constant; the standard
    vol-matched-blend convention — fixed weight, adds no marginal turnover/cost)."""
    v = float(pd.Series(r).std() * np.sqrt(ann))
    return r * (tgt / v) if v > 0 else r


def _monthly_trades(pos: pd.Series, credit_tk, treas_tk, h, r_credit, r_treas, lo) -> list:
    """One trade per monthly held run per leg (long credit / short duration-matched Treasury), within the
    overlap window — supplies deployment-sanity breadth in the 'Credit' sector."""
    out, p = [], pos[pos.index >= lo]
    # (sign, notional-scale, leg returns): the credit pair = +1*HYG  and  -h*IEI
    legs = {credit_tk: (+1.0, 1.0, r_credit), treas_tk: (-1.0, float(h), r_treas)}
    for nm, (sgn, scale, rr) in legs.items():
        for _, g in p.groupby([p.index.year, p.index.month]):
            if len(g) == 0:
                continue
            rseg = rr.reindex(g.index).fillna(0.0)
            out.append({"ticker": nm, "sector": "Credit",
                        "entry_date": str(g.index[0].date()), "exit_date": str(g.index[-1].date()),
                        "hold_days": int(len(g)), "position_value": float(abs(g).mean() * scale),
                        "pnl": float((sgn * scale * g * rseg).sum())})
    return out


def signal(panel, credit_tk="HYG", treas_tk="IEI", hedge_ratio=0.78, credit_vol=0.10,
           max_pos=2.0, vol_lb=60, blend=0.5, rebalance="W-FRI"):
    """(daily_returns, trades) for the vol-matched credit-carry + trend book. Causal: signals lagged 1 day."""
    px = panel[[credit_tk, treas_tk]].dropna(how="all")
    r = px.pct_change()
    # --- CREDIT-CARRY signal: long credit, short duration-matched Treasury (strips rates/term) ---
    cr = (r[credit_tk] - hedge_ratio * r[treas_tk]).rename("CREDIT").to_frame()
    sig = pd.DataFrame(1.0, index=cr.index, columns=["CREDIT"])          # always-long the credit premium
    pos = inv_vol_position(sig, cr, target_vol=credit_vol, vol_lb=vol_lb,
                           max_pos=max_pos, rebalance=rebalance)["CREDIT"]   # weekly-held, 1-day lagged
    # turnover hits BOTH legs of the pair (notional traded = (1+h)*|Δw|); cost lagged with the position
    turn = pos.diff().abs().fillna(0.0) * (1.0 + hedge_ratio)
    credit = (pos * cr["CREDIT"] - turn * COST).dropna()
    credit.index = pd.to_datetime(credit.index).normalize(); credit.name = "credit"

    # --- TREND leg (frozen Boreas 21-market TSMOM crisis-alpha hedge; arrives net-of-8bps) ---
    trend, ttrades = trend_returns()
    trend = pd.Series(trend).copy(); trend.index = pd.to_datetime(trend.index).normalize()
    trend.name = "trend"

    # --- align overlap, vol-match each leg, blend 50/50 (fixed weights => no extra turnover/cost) ---
    df = pd.concat([credit, trend], axis=1).dropna()
    combo = blend * _vol_scale(df["credit"]) + (1.0 - blend) * _vol_scale(df["trend"])
    combo = combo.dropna(); combo.name = "credit_carry_trend"

    # --- trades = trend sign-runs (4 sectors) + credit monthly runs (Credit sector), within overlap ---
    lo = df.index.min()
    trades = [t for t in ttrades if pd.Timestamp(t["entry_date"]) >= lo]
    trades += _monthly_trades(pos, credit_tk, treas_tk, hedge_ratio, r[credit_tk], r[treas_tk], lo)
    return combo, trades


SPEC = StrategySpec(
    id="credit-carry-trend-book",
    family="credit_carry_trend_combo",
    title=("Credit-spread carry (HY minus duration-matched Treasuries) hedged by the validated "
           "21-market trend leg — a two-premium book with a NEW pro-cyclical leg"),
    markets=["credit", "futures"],
    data_desc=("FREE: dividend-ADJUSTED total-return HYG/LQD (credit) + IEI/IEF (Treasuries) via yfinance "
               "auto_adjust=True (2007+, covers GFC); Boreas 21-market trend (yfinance). yf_panel's "
               "UNADJUSTED Close would zero the coupon carry, so the adjusted-close path is used. Frozen "
               "0.78 HY/IEI duration hedge ratio locked from ETF effective durations."),
    pre_registration=(
        "Credit leg = CREDIT/default-risk premium. Long HYG total return financed by SHORT duration-"
        "matched IEI; credit_excess = r_HYG - 0.78*r_IEI (0.78 = HY eff-dur ~3.5y / IEI ~4.5y, FROZEN "
        "ex-ante from ETF fact-sheets) to strip rates/term and isolate the spread. Dividend-ADJUSTED "
        "total returns (the coupon IS the carry). Always-long the premium, inverse-vol to a 10% target, "
        "WEEKLY rebalance, 8bps on (1+h)*turnover (both pair legs; deliberately conservative vs real ETF "
        "spread+commission), signals lagged 1 day (no look-ahead). PRO-CYCLICAL: earns the spread in calm, "
        "crashes when spreads gap out. Trend leg = FROZEN Boreas 21-market cross-asset TSMOM crisis-alpha "
        "hedge. Vol-matched 50/50 (static portfolio-construction constant). FROZEN. PRIMARY pre-registered "
        "metric = combined book MAR / max-drawdown + leg correlation on the write-once 2022+ holdout (the "
        "2022 spread-widening is the key OOS hedge test) via CPCV/DSR/PBO/FDR; standalone legs are "
        "diagnostics only. HYPOTHESIS: corr(credit,trend)~0 lets the crisis-alpha trend leg cut the credit "
        "drawdown (analogue of crypto-carry 26.5%->14.5%) at little Sharpe cost. Crisis decomposition "
        "reported for GFC-2008, COVID-2020, 2022. Credit-spread carry is structurally distinct from prior "
        "crypto funding-carry, FX/bond term carry, and commodity term-structure carry tests."),
    load_data=load_data, signal=signal,
    default_params={},
    grid={
        "default": {},                                              # HYG vs IEI, h=0.78 (primary)
        "treas_ief":    {"treas_tk": "IEF", "hedge_ratio": 0.47},   # 7-10y Treasury hedge (HY3.5/IEF7.5)
        "ig_lqd":       {"credit_tk": "LQD", "treas_tk": "IEF", "hedge_ratio": 1.10},  # IG variant
        "blend_credit": {"blend": 0.65},
        "blend_trend":  {"blend": 0.35},
        "credvol_low":  {"credit_vol": 0.07},
        "vollb_120":    {"vol_lb": 120},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=23,  # 21 trend markets + HYG/IEI credit pair
)