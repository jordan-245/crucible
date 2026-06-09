"""Agent-proposed strategy: Short-term Reversal (liquidity-provision) + Trend two-premium book.

Reversal leg = cross-sectional SHORT-TERM REVERSAL on the validated Boreas 21-market liquid-futures
core (equity-index / bond / FX / commodity). Economic claim: get *paid to provide immediacy* and
absorb the prior week's order-flow overshoot (a pro-cyclical liquidity-provision premium). Each
Friday close: per market compute the trailing 5-day return standardized by its 60-day realized vol,
z-score it cross-sectionally, then SHORT the top-quartile recent winners / LONG the bottom-quartile
recent losers (reversal weight = -z in the tails, demeaned to market-neutral), inverse-vol sized,
gross scaled to ~10% annualized vol via a CAUSAL vol-target overlay, held one week, 8bps round-trip
micro-cost on turnover.

Trend leg = the FROZEN validated Boreas 21-market TSMOM (1/3/12m sign blend, inverse-vol, weekly) —
crisis-alpha hedge. Vol-matched 50/50 against the reversal leg.

WHY THE PAIRING: reversal is trend's pro-cyclical opposite-tail partner — it earns in calm
mean-reverting markets exactly when trend whipsaws, and gets run over in sustained crisis moves
exactly when trend pays. Pre-registered success bar mirrors the validated carry+trend pattern:
leg correlation <= 0 AND combined maxDD materially below trend-only maxDD at no Sharpe cost.

GATE-0 KILL CRITERION (frozen BEFORE the verdict, per the proposal): the reversal must be a genuine
liquidity premium on liquid DAILY closes that SURVIVES realistic turnover cost — NOT a sub-daily
bid-ask-bounce mirage. Weekly cross-sectional reversal is high-turnover; if 8bps round-trip on that
turnover erases the gross signal, daily-close reversal is infeasible and the leg is dead. This is the
honest free-data ceiling the harness is built to adjudicate.

The harness owns ALL rails (split / CPCV / DSR / PBO / FDR / write-once holdout / deployment-sanity /
verdict / wiki). This module only produces (daily_returns, trades). FROZEN.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, trend_returns, inv_vol_position

COST = 8.0 / 1e4  # 8bps per unit turnover, round trip (same frozen micro-cost as Boreas)

# Canonical PRE-REGISTERED Boreas 21-market liquid CTA core: yf ticker -> (friendly name, sector).
# Same universe the validated trend leg trades, so the reversal leg is a true same-panel partner.
UNI = {
    "ES=F": ("S&P500", "Equity"), "NQ=F": ("Nasdaq", "Equity"), "YM=F": ("Dow", "Equity"), "RTY=F": ("Russell", "Equity"),
    "ZT=F": ("2Y", "Rates"), "ZF=F": ("5Y", "Rates"), "ZN=F": ("10Y", "Rates"), "ZB=F": ("30Y", "Rates"),
    "CL=F": ("Crude", "Commod"), "GC=F": ("Gold", "Commod"), "SI=F": ("Silver", "Commod"), "HG=F": ("Copper", "Commod"),
    "NG=F": ("NatGas", "Commod"), "ZC=F": ("Corn", "Commod"), "ZS=F": ("Soybean", "Commod"), "ZW=F": ("Wheat", "Commod"),
    "6E=F": ("EUR", "FX"), "6J=F": ("JPY", "FX"), "6B=F": ("GBP", "FX"), "6A=F": ("AUD", "FX"), "6C=F": ("CAD", "FX"),
}
SECTOR = {name: sec for _, (name, sec) in UNI.items()}
START = "2005-01-01"


def load_data() -> pd.DataFrame:
    """Close panel for the 21-market Boreas core via the sanctioned yfinance adapter (FREE),
    renamed to friendly market names. This is exactly the panel the trend leg also trades."""
    px = yf_panel(list(UNI), start=START).rename(columns={t: UNI[t][0] for t in UNI})
    px = px[[n for (_, (n, _)) in UNI.items() if n in px.columns]]
    px.index = pd.to_datetime(px.index).normalize()
    return px


def _vol_scale(r, tgt=0.10, ann=252):
    v = float(pd.Series(r).std() * np.sqrt(ann))
    return r * (tgt / v) if v > 0 else r


def _sign_run_trades(pos: pd.DataFrame, rets: pd.DataFrame, suffix: str, lo) -> list:
    """One trade per held-position run per market (for deployment-sanity), within the overlap."""
    trades = []
    for nm in pos.columns:
        s = np.sign(pos[nm]).fillna(0.0)
        cur, ent = 0.0, None
        for dt, sg in s.items():
            if sg != cur:
                if cur != 0.0 and ent is not None and dt >= lo:
                    seg = pos[nm].loc[ent:dt]
                    rseg = rets[nm].loc[ent:dt].fillna(0.0)
                    trades.append({"ticker": f"{nm}{suffix}", "sector": SECTOR.get(nm, "Unknown"),
                                   "entry_date": str(ent.date()), "exit_date": str(dt.date()),
                                   "hold_days": int(len(seg)),
                                   "position_value": float(abs(seg).mean() if len(seg) else 0.0),
                                   "pnl": float((seg.fillna(0) * rseg).sum())})
                cur, ent = sg, dt
    return trades


def signal(panel, blend=0.5, rev_vol=0.10, vol_lb=60, q=0.25, max_pos=2.0, rebalance="W-FRI"):
    """(daily_returns, trades) for the vol-matched reversal + trend book. Causal: signals lagged 1 day."""
    px = panel.copy()
    px.index = pd.to_datetime(px.index).normalize()
    rets = px.pct_change()

    # --- REVERSAL signal: standardized trailing 5d return, cross-sectional z, short winners/long losers ---
    r5 = px.pct_change(5)
    vold = rets.rolling(vol_lb, min_periods=vol_lb // 2).std()
    stand = r5 / (vold.replace(0, np.nan) * np.sqrt(5))          # 5d return standardized by 60d realized vol
    z = stand.sub(stand.mean(axis=1), axis=0).div(stand.std(axis=1).replace(0, np.nan), axis=0)
    hi = z.quantile(1 - q, axis=1)                                # top-quartile cut (recent winners)
    lo_q = z.quantile(q, axis=1)                                  # bottom-quartile cut (recent losers)
    tail = z.ge(hi, axis=0) | z.le(lo_q, axis=0)
    sig = (-z).where(tail, 0.0)                                   # reversal: -z in the tails, 0 in the middle
    cnt = tail.sum(axis=1).replace(0, np.nan)                     # market-neutral: demean over active names
    sig = sig.sub(sig.where(tail).sum(axis=1) / cnt, axis=0).where(tail, 0.0).fillna(0.0)

    # inverse-vol sizing + weekly hold + 1d lag (no look-ahead) via the shared building block
    pos = inv_vol_position(sig, rets, target_vol=rev_vol, vol_lb=vol_lb, max_pos=max_pos, rebalance=rebalance)
    # CAUSAL gross-leverage overlay -> scale book to ~rev_vol annualized (uses only past realized vol)
    g = (pos * rets).sum(axis=1)
    pv = g.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252)
    ovl = (rev_vol / pv.replace(0, np.nan)).shift(1).clip(0, 3)
    ovl = ovl.resample(rebalance).last().reindex(rets.index, method="ffill").fillna(0.0)
    pos = pos.mul(ovl, axis=0)

    gross = (pos * rets).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1).fillna(0.0)              # high-turnover weekly reversal
    rev = (gross - turn * COST).dropna()                        # net of the GATE-0 cost test
    rev.index = pd.to_datetime(rev.index).normalize(); rev.name = "reversal"

    # --- TREND leg (frozen Boreas 21-market TSMOM crisis-alpha hedge) ---
    trend, ttrades = trend_returns()
    trend = pd.Series(trend).copy(); trend.index = pd.to_datetime(trend.index).normalize()
    trend.name = "trend"

    # --- align overlap, vol-match each leg, blend 50/50 ---
    df = pd.concat([rev, trend], axis=1).dropna()
    combo = blend * _vol_scale(df["reversal"]) + (1 - blend) * _vol_scale(df["trend"])
    combo = combo.dropna(); combo.name = "reversal_trend"

    # --- trades = trend sign-runs (4 sectors) + reversal sign-runs (4 sectors), within the overlap ---
    lo = df.index.min()
    trades = [t for t in ttrades if pd.Timestamp(t["entry_date"]) >= lo]
    trades += _sign_run_trades(pos.loc[pos.index >= lo], rets, "-rev", lo)
    return combo, trades


SPEC = StrategySpec(
    id="reversal-trend-book",
    family="reversal_trend_combo",
    title="Short-term reversal (liquidity-provision) + Trend two-premium futures book",
    markets=["futures"],
    data_desc=("FREE / already-owned: Boreas 21-market liquid-futures core (yfinance daily closes, "
               "2005+, incl. 2008/2020/2022) for the cross-sectional 5d reversal leg + the validated "
               "Boreas 21-market TSMOM trend leg. $0 marginal cost; all markets IB-micro-tradable."),
    pre_registration=(
        "Reversal leg = cross-sectional SHORT-TERM REVERSAL (liquidity-provision premium) on the 21 "
        "Boreas markets. Each Friday close: per market compute trailing 5-day return standardized by its "
        "60-day realized vol; z-score cross-sectionally; SHORT top-quartile winners / LONG bottom-quartile "
        "losers (reversal weight = -z in the tails, demeaned to market-neutral); inverse-vol sized; gross "
        "scaled to ~10% annualized vol via a CAUSAL trailing-vol overlay; held one week; 8bps round-trip on "
        "turnover; signals lagged 1 day (no look-ahead). Trend leg = FROZEN Boreas 21-market TSMOM "
        "(1/3/12m sign blend, inverse-vol, weekly). Vol-matched 50/50. FROZEN. "
        "GATE-0 KILL CRITERION (frozen before the verdict): weekly cross-sectional reversal is "
        "high-turnover; the leg must SURVIVE 8bps round-trip cost on liquid DAILY closes — if cost on "
        "turnover erases the gross signal, daily-close reversal is a sub-daily microstructure mirage and "
        "is infeasible (KILL). PRIMARY pre-registered metric = the COMBINED book's net Sharpe and "
        "max-drawdown vs each leg standalone, plus inter-leg correlation and 2008/2020/2022 stress "
        "behavior, on the write-once 2022+ holdout via CPCV/DSR/PBO/FDR; standalone legs are diagnostics. "
        "HYPOTHESIS (mirrors validated carry+trend): corr(reversal,trend) <= 0 AND combined maxDD "
        "materially below trend-only maxDD at no Sharpe cost — reversal's calm-market premium plugging "
        "trend's whipsaw tail. This leg lives in FUTURES (sidestepping the CLOSED equity-factor decision) "
        "and is a distinct premium TYPE (liquidity provision), tested as a COMBINATION, not a standalone edge."),
    load_data=load_data, signal=signal,
    default_params={},
    grid={
        "default": {},
        "blend_rev": {"blend": 0.65},
        "blend_trend": {"blend": 0.35},
        "revvol_low": {"rev_vol": 0.07},
        "quintile": {"q": 0.20},
        "tercile": {"q": 0.33},
        "vollb_120": {"vol_lb": 120},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=31,  # ~10 reversal quartile positions + 21 trend markets
)