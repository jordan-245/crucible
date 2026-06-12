"""
two_premium_illiq_trend_v1
==========================
Two-premium book: the VALIDATED Amihud illiquidity premium leg (frozen amihud_illiq_tranched_v3
design: long illiquid quintile / short top-15-most-liquid per ADV tercile in US small caps,
4-week tranched weekly rebalance, inverse-vol sizing, residual IWM beta trim, declared hedge)
plus a SMALL pre-registered tail overlay of the validated 21-market trend leg sized at 25% of
book risk (both legs scaled to equal trailing-60d vol BEFORE the 25/100 weighting — explicitly
NOT a reflexive 50/50, per the 2026-06-08 anti-pattern that sank credit-carry).

NO new alpha is searched here: the Amihud leg is reproduced as-frozen, the trend leg is consumed
as a finished return stream, and the ONLY claim under test is book-level complementarity at one
pre-registered sizing. grid contains the single primary spec plus a non-selectable standalone
baseline needed by the machine-checkable expectations (DSR effective-N stays honest and tiny).

Lag discipline: all signals (illiq, ADV, vols, hedge beta) use data up to and including the
rebalance date; positions are applied via W.shift(1) before net_of_cost / trades_from_weights.
Trend overlay vol-matching ratio is also shift(1)-lagged.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel, trend_returns
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

# ---- frozen design constants (amihud_illiq_tranched_v3 — DO NOT TUNE) ----
START = "2010-01-01"          # overlap of Sharadar small-cap depth + trend stream; covers 2015-16 & 2020 stress
ILLIQ_LB = 63                 # Amihud rolling window
VOL_LB = 60                   # trailing vol window (sizing, hedge beta, leg vol-matching)
N_TRANCHES = 4                # 4-week tranching of the weekly target book
N_SHORT_PER_TERCILE = 15      # short the 15 most-liquid names per ADV tercile (borrowable, liquid)
LONG_QUINTILE = 0.20          # long the most-illiquid quintile per tercile
PRICE_FLOOR = 2.0             # no sub-$2 names
HEDGE_CAP = 0.35              # residual IWM trim cap (declared hedge sleeve)
COST_BPS = 8.0

_UNIV = {}


def _universe():
    """Cached sector-spread small-cap universe + sector map (survivorship-clean)."""
    if not _UNIV:
        tickers, sector_map = sector_universe(marketcap="Small", top_n_per_sector=120)
        sector_map = dict(sector_map)
        sector_map["IWM"] = "ETF-Hedge"   # declared hedge sleeve; judged separately by the gate
        _UNIV["tickers"] = list(tickers)
        _UNIV["sector_map"] = sector_map
    return _UNIV["tickers"], _UNIV["sector_map"]


def load_data() -> pd.DataFrame:
    tickers, _ = _universe()
    closeadj = sep_panel(tickers, START, field="closeadj")
    close = sep_panel(tickers, START, field="close")
    volume = sep_panel(tickers, START, field="volume")
    iwm = yf_panel(["IWM"], START).reindex(closeadj.index).ffill()
    panel = pd.concat(
        {"closeadj": closeadj, "close": close, "volume": volume, "hedge": iwm[["IWM"]]},
        axis=1,
    )
    return panel


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' by design: both legs already completed their own generalization
    # (Amihud: 3/3 untouched universes; trend: 21 markets / 3 stress regimes).
    # The only new claim is book-level complementarity -> forward validation, not breadth.
    raise NotImplementedError("local-scope combination book; no generalization universes")


def signal(panel, trend_frac=0.25, illiq_lb=ILLIQ_LB, vol_lb=VOL_LB, n_tranches=N_TRANCHES):
    _, sector_map = _universe()

    px = panel["closeadj"]
    close = panel["close"]
    volume = panel["volume"]
    iwm_px = panel["hedge"]["IWM"]

    rets = px.pct_change()
    iwm_ret = iwm_px.pct_change()

    dollar_vol = (close * volume).replace(0.0, np.nan)
    illiq = (rets.abs() / dollar_vol).rolling(illiq_lb, min_periods=40).mean()
    adv = dollar_vol.rolling(illiq_lb, min_periods=40).median()
    dvol = rets.rolling(vol_lb, min_periods=30).std()

    # ---- weekly target book (frozen v3 construction) ----
    rebal_dates = (
        pd.Series(rets.index, index=rets.index).resample("W-FRI").last().dropna().tolist()
    )
    weekly_rows, weekly_idx = [], []
    for d in rebal_dates:
        il, ad, vv, pr = illiq.loc[d], adv.loc[d], dvol.loc[d], close.loc[d]
        ok = il.notna() & ad.notna() & vv.notna() & (vv > 0) & (pr >= PRICE_FLOOR)
        names = il.index[ok]
        w = pd.Series(0.0, index=il.index)
        if len(names) >= 150:
            ad_ok = ad[names]
            terciles = pd.qcut(ad_ok.rank(method="first"), 3, labels=False)
            for t in range(3):
                tn = ad_ok.index[terciles == t]
                il_t = il[tn]
                n_long = max(int(len(tn) * LONG_QUINTILE), 5)
                longs = il_t.nlargest(n_long).index                       # most illiquid
                n_short = min(N_SHORT_PER_TERCILE, max(len(tn) // 5, 1))
                shorts = il_t.nsmallest(n_short).index                    # most liquid (borrowable)
                iv_l = 1.0 / vv[longs]
                iv_s = 1.0 / vv[shorts]
                if iv_l.sum() > 0:
                    w[longs] += (iv_l / iv_l.sum()) / 3.0                 # long leg sums to +1
                if iv_s.sum() > 0:
                    w[shorts] -= (iv_s / iv_s.sum()) / 3.0                # short leg sums to -1
        weekly_rows.append(w)
        weekly_idx.append(d)

    W_weekly = pd.DataFrame(weekly_rows, index=pd.DatetimeIndex(weekly_idx))
    W_tranched = W_weekly.rolling(n_tranches, min_periods=1).mean()       # 4-week tranching
    W = W_tranched.reindex(rets.index).ffill().fillna(0.0)

    # ---- residual IWM beta trim (declared hedge sleeve, capped) ----
    g_raw = (W.shift(1) * rets).sum(axis=1)
    beta = (g_raw.rolling(vol_lb).cov(iwm_ret) / iwm_ret.rolling(vol_lb).var()).shift(1)
    beta = beta.clip(lower=0.0, upper=HEDGE_CAP)
    beta_weekly = beta.reindex(W_tranched.index).ffill()                  # update hedge weekly
    W["IWM"] = -beta_weekly.reindex(rets.index).ffill().fillna(0.0)

    rets_full = rets.copy()
    rets_full["IWM"] = iwm_ret

    # W built from same-day data -> lag is OUR responsibility: shift(1) here.
    amihud_net = net_of_cost(W.shift(1), rets_full, cost_bps=COST_BPS, name="amihud_tranched_v3")
    trades = list(trades_from_weights(W.shift(1), rets_full, sector_map))

    # ---- trend tail overlay (validated leg, consumed as-is; 25% of book risk) ----
    if trend_frac > 0:
        t_ret, t_trades = trend_returns()
        t_ret = t_ret.reindex(amihud_net.index).fillna(0.0)
        a_vol = amihud_net.rolling(vol_lb, min_periods=30).std()
        t_vol = t_ret.rolling(vol_lb, min_periods=30).std()
        scale = (a_vol / t_vol).shift(1)                                  # lagged vol-matching
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=5.0)
        combined = amihud_net + trend_frac * (t_ret * scale)
        trades = trades + list(t_trades)
    else:
        combined = amihud_net

    combined = combined.dropna()
    combined.name = "illiq_trend_two_premium"
    return combined, trades


# ---------------- machine-checkable pre-registered expectations ----------------

def _mdd(r):
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _sharpe(r):
    r = r.dropna()
    if len(r) < 60 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _legs(ctx):
    comb = ctx["search"].dropna()
    base = ctx["grid"]["amihud_standalone"].reindex(comb.index).fillna(0.0)
    return comb, base


def _chk_maxdd(ctx):
    comb, base = _legs(ctx)
    mc, mb = abs(_mdd(comb)), abs(_mdd(base))
    ratio = mc / max(mb, 1e-9)
    return {"pass": bool(mb > 0 and ratio <= 0.80),
            "observed": f"combined MaxDD {mc:.3f} vs standalone {mb:.3f} (ratio {ratio:.2f}, need <=0.80)"}


def _chk_sharpe(ctx):
    comb, base = _legs(ctx)
    sc, sb = _sharpe(comb), _sharpe(base)
    ok = (sb <= 0) or (sc >= 0.90 * sb)
    return {"pass": bool(ok), "observed": f"combined Sharpe {sc:.2f} vs standalone {sb:.2f} (need >=90%)"}


def _chk_leg_corr(ctx):
    comb, base = _legs(ctx)
    overlay = comb - base                       # = trend_frac * vol-scaled trend stream
    mask = overlay.abs() > 0
    if int(mask.sum()) < 60:
        return {"pass": False, "observed": "overlay inactive (<60 active days)"}
    c = float(base[mask].corr(overlay[mask]))   # corr invariant to overlay scaling
    return {"pass": bool(c <= 0.10), "observed": f"leg correlation {c:.3f} (need <=+0.10)"}


def _chk_covid_crash(ctx):
    # stress decomposition diagnostic (2020 crash sits in the SEARCH window; 2022 is holdout
    # and 2008 predates the overlap, both excluded by construction)
    comb, base = _legs(ctx)
    seg = (comb - base).loc["2020-02-19":"2020-04-30"]
    tot = float(seg.sum())
    return {"pass": bool(tot >= 0.0), "observed": f"trend-overlay PnL in 2020 crash window: {tot:+.4f}"}


SPEC = StrategySpec(
    id="two_premium_illiq_trend_v1",
    family="multi_premium_combination",
    title="Illiquidity-Premium x Trend Crisis-Alpha Two-Premium Book (frozen Amihud tranched_v3 leg + 25%-risk trend tail overlay)",
    markets=["US small-cap equities (Amihud long-short, IWM residual trim)",
             "21-market futures trend (tail overlay via trend_returns)"],
    data_desc="OWNED Sharadar SEP/TICKERS small-cap panel (delisted included, survivorship-clean); "
              "IWM hedge via yfinance ETF panel; trend leg consumed as the validated trend_returns() stream. "
              "$0 incremental data.",
    pre_registration=(
        "PRE-REGISTERED PRIMARY: combined book = frozen amihud_illiq_tranched_v3 stream at 100% risk "
        "+ trend_returns() at 25% of book risk, both legs equalised to trailing-60d vol (lagged) before "
        "the 25/100 weighting. NO sizing grid, NO re-tuning of either leg — any parameter touch on the "
        "Amihud construction is meta-overfitting on a closed question; the only grid entry beyond default "
        "is the standalone-Amihud baseline required by the machine-checkable expectations (not a selectable "
        "candidate). Success criteria vs standalone Amihud on the identical window: combined MaxDD reduced "
        ">=20%, Sharpe degradation <=10%, leg correlation <= +0.10 (a realized correlation > +0.3 falsifies "
        "the complementarity premise outright), AND the combined book clears the standard gate stack "
        "(MCPT market-neutral absolute null on the Amihud panel, holdout from 2022-01-01, DSR). Stress "
        "decomposition is a secondary diagnostic, not selectable: 2020 crash overlay contribution sign is "
        "machine-checked; 2008 is excluded (Amihud-leg history starts 2010, confirmed pre-build per gate0) "
        "and 2022 is excluded from checks because it is holdout. scope='local': both legs already passed "
        "their own generalization batteries; the only new claim is book-level complementarity, confirmed "
        "by forward validation alongside the live amihud_illiq_tranched_v3 shadow (verdict aligned with the "
        "leg's forward evidence gate, ~Q4 2026). Hedge sleeve declared: IWM residual beta trim, cap 0.35."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                          # the 25% overlay IS the verdict
        "amihud_standalone": {"trend_frac": 0.0},  # baseline for expectations only, not selectable
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=10,
    hedge_tickers=["IWM"],
    hedge_cap=HEDGE_CAP,
    expectations=[
        {"name": "maxdd_reduced_20pct",
         "claim": "combined book MaxDD <= 80% of standalone Amihud MaxDD (search window)",
         "check": _chk_maxdd},
        {"name": "sharpe_degradation_le_10pct",
         "claim": "combined Sharpe >= 90% of standalone Amihud Sharpe (overlay sized to minimise drag)",
         "check": _chk_sharpe},
        {"name": "leg_correlation_le_0p10",
         "claim": "realized correlation between Amihud leg and trend overlay <= +0.10",
         "check": _chk_leg_corr},
        {"name": "trend_pays_in_2020_crash",
         "claim": "trend overlay contribution is non-negative in the 2020-02-19..2020-04-30 liquidity crisis",
         "check": _chk_covid_crash},
    ],
)