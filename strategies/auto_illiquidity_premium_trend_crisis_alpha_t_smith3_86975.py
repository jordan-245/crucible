"""
Amihud Illiquidity Premium (tranched) x 10-ETF Cross-Asset Trend Tail-Overlay
=============================================================================
Frozen two-premium book. Leg A reproduces the validated Amihud illiquidity
premium design (long the most-illiquid tradable small-caps, inverse-vol sized,
weekly cohorts held as 4 staggered tranches, residual IWM beta trim — declared
hedge sleeve). Leg B is the canonical time-series-trend rule (12m sign, per-line
vol budgeting, weekly held) mapped onto a FIXED, pre-registered 10-ETF list
spanning 6 asset-class buckets — the ONLY mutated object vs the 5-ETF parent.
Overlay sized at 25% of book risk (tail overlay, NOT 50/50); both legs scaled
to a common trailing-60d vol before blending. All signals lagged 1 day via
W.shift(1) passed to net_of_cost/trades_from_weights (the lag is taken HERE).
No file writes, no config, no side effects.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel, trend_returns
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

START = "2008-01-01"

# PRE-REGISTERED, FIXED sleeve list (written down before any backtest; the only
# changed object vs the 5-ETF parent). 6 buckets: equity x3, duration x2,
# credit x2, metals x2, broad commodity x1.
TREND_ETFS = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "DBC"]
PARENT_5 = ["SPY", "EFA", "TLT", "GLD", "DBC"]          # comparison line ONLY (grid variant)
HEDGE_ETF = "IWM"                                        # declared hedge sleeve (whitelist)

ETF_SECTOR = {
    "SPY": "ETF-Equity-US", "EFA": "ETF-Equity-Intl", "EEM": "ETF-Equity-EM",
    "TLT": "ETF-Duration-Long", "IEF": "ETF-Duration-Int",
    "LQD": "ETF-Credit-IG", "HYG": "ETF-Credit-HY",
    "GLD": "ETF-Metals", "SLV": "ETF-Metals", "DBC": "ETF-Commodity",
    "IWM": "ETF-Hedge",
}

# ----------------------------------------------------------------------------- universe
_UNIV = {}

def _universe():
    """Sector-spread small-cap universe (survivorship-clean, delisted included)."""
    if not _UNIV:
        tickers, sector_map = sector_universe(marketcap="Small", top_n_per_sector=80)
        _UNIV["tickers"] = tickers
        _UNIV["sectors"] = sector_map
    return _UNIV["tickers"], _UNIV["sectors"]


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """MultiIndex-column panel: px/close/vol for the small-cap leg, etf closes for sleeve+hedge."""
    tickers, _ = _universe()
    px = sep_panel(tickers, START, field="closeadj")      # adjusted, for returns
    close = sep_panel(tickers, START, field="close")      # raw close, for dollar volume
    vol = sep_panel(tickers, START, field="volume")
    etf = yf_panel(TREND_ETFS + [HEDGE_ETF], START)       # all inceptions <= Apr-2007
    panel = pd.concat({"px": px, "close": close, "vol": vol, "etf": etf}, axis=1).sort_index()
    return panel


def load_gen_data(label) -> pd.DataFrame:
    """scope='local' by design: both legs' standalone validation is settled; the new
    claim (book-level complementarity on tradable instruments + breadth sufficiency)
    is forward-validated in paper, per the pre-registration. No gen universes."""
    raise ValueError("local-scope spec: no generalization universes declared (label=%r)" % (label,))


# ----------------------------------------------------------------------------- signal
def signal(panel,
           amihud_lb=63, top_n=50, n_tranches=4, dvol_floor=2.5e5,
           ivol_lb=60, trend_lb=252, overlay_risk=0.25, leg_vol=0.10,
           hedge_ratio=0.35, cost_bps=8.0, sleeve_etfs=None):
    sleeve = list(sleeve_etfs) if sleeve_etfs else TREND_ETFS

    px = panel["px"]
    close = panel["close"]
    vol = panel["vol"]
    etf = panel["etf"]
    idx = px.index

    rets_eq = px.pct_change(fill_method=None)
    rets_etf = etf.pct_change(fill_method=None)

    # ---- Leg A: Amihud illiquidity, long-only most-illiquid tradable small-caps
    dvol = (close * vol).replace(0.0, np.nan)
    illiq = (rets_eq.abs() / dvol).rolling(amihud_lb, min_periods=int(amihud_lb * 0.7)).mean()
    illiq = illiq.where(illiq > 0)
    med_dvol = dvol.rolling(21, min_periods=10).median()
    illiq = illiq.where(med_dvol >= dvol_floor)            # tradability floor at $5K book
    z = xs_zscore(np.log(illiq))                           # log tames the heavy right tail

    ivol = rets_eq.rolling(ivol_lb, min_periods=int(ivol_lb * 0.7)).std() * np.sqrt(252)

    # weekly rebalance dates = last trading day of each week
    wk = pd.Series(idx, index=idx).groupby(pd.Grouper(freq="W-FRI")).last().dropna()
    rb_dates = pd.DatetimeIndex(wk.values)

    weekly_tgt = {}
    for d in rb_dates:
        row = z.loc[d].dropna()
        if len(row) < top_n * 2:
            continue
        picks = row.nlargest(top_n).index
        iv = (1.0 / ivol.loc[d, picks]).replace([np.inf, -np.inf], np.nan).dropna()
        if iv.empty:
            continue
        weekly_tgt[d] = iv / iv.sum()                      # gross 1, inverse-vol within cohort

    if not weekly_tgt:
        empty = pd.Series(0.0, index=idx, name="amihud_trend10_book")
        return empty, []

    tgt_df = pd.DataFrame(weekly_tgt).T.sort_index().fillna(0.0)
    # tranched_v3 mechanic: daily book = average of last n_tranches weekly cohorts
    tranched = tgt_df.rolling(n_tranches, min_periods=1).mean()
    g = tranched.sum(axis=1).replace(0.0, np.nan)
    tranched = tranched.div(g, axis=0).fillna(0.0)         # renormalize gross to 1

    W_eq = tranched.reindex(idx).ffill().fillna(0.0)
    W_eq = W_eq * px[W_eq.columns].notna().astype(float)   # never hold a ghost post-delisting

    on = W_eq.abs().sum(axis=1) > 0
    W_a = W_eq.copy()
    W_a[HEDGE_ETF] = np.where(on, -hedge_ratio, 0.0)       # declared residual IWM trim

    # ---- Leg B: canonical trend rule on the FIXED ETF list (no new parameters)
    sig = np.sign(etf[sleeve].pct_change(trend_lb, fill_method=None))
    evol = rets_etf[sleeve].rolling(ivol_lb, min_periods=int(ivol_lb * 0.7)).std() * np.sqrt(252)
    w_line = (sig * (leg_vol / evol.clip(lower=0.02)) / len(sleeve)).fillna(0.0)  # equal risk/line
    W_t = w_line.loc[w_line.index.intersection(rb_dates)].reindex(idx).ffill().fillna(0.0)

    # ---- vol-match legs to a common target before blending (scales lagged: shift(1))
    rets_all = pd.concat([rets_eq, rets_etf], axis=1)
    rets_all = rets_all.loc[:, ~rets_all.columns.duplicated()]

    r_a = (W_a.shift(1) * rets_all.reindex(columns=W_a.columns)).sum(axis=1)
    r_t = (W_t.shift(1) * rets_all.reindex(columns=W_t.columns)).sum(axis=1)
    s_a = (leg_vol / (r_a.rolling(60).std() * np.sqrt(252))).clip(upper=3.0).shift(1)
    s_t = (leg_vol / (r_t.rolling(60).std() * np.sqrt(252))).clip(upper=3.0).shift(1)

    W = pd.concat([W_a.mul(s_a.fillna(0.0), axis=0),
                   W_t.mul(s_t.fillna(0.0), axis=0) * overlay_risk], axis=1)
    W = W.T.groupby(level=0).sum().T                       # merge any duplicate columns (e.g. IWM)

    gross = W.abs().sum(axis=1)
    W = W.mul((2.0 / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0), axis=0)  # gross cap 2x

    # ---- THE LAG: weights built same-day, shifted 1 day here, as the contract requires
    Wl = W.shift(1).fillna(0.0)
    R = rets_all.reindex(columns=Wl.columns)

    daily = net_of_cost(Wl, R, cost_bps=cost_bps, name="amihud_trend10_book")

    sector_map = dict(_universe()[1])
    sector_map.update(ETF_SECTOR)
    trades = trades_from_weights(Wl, R, sector_map)        # kit stamps entry_regime (contract)

    return daily, trades


# ----------------------------------------------------------------------------- soft expectations
def _stats(r):
    r = pd.Series(r).dropna()
    sd = r.std()
    sh = (r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return float(sh), abs(dd)


def _chk_maxdd(ctx):
    _, dd_c = _stats(ctx["search"])
    _, dd_s = _stats(ctx["grid"]["no_overlay"])
    return {"pass": bool(dd_c <= 0.80 * dd_s),
            "observed": "combined_dd=%.3f standalone_dd=%.3f ratio=%.2f" % (dd_c, dd_s, dd_c / max(dd_s, 1e-9))}


def _chk_sharpe(ctx):
    sh_c, _ = _stats(ctx["search"])
    sh_s, _ = _stats(ctx["grid"]["no_overlay"])
    return {"pass": bool(sh_c >= 0.90 * sh_s),
            "observed": "combined_sharpe=%.2f standalone_sharpe=%.2f" % (sh_c, sh_s)}


def _chk_leg_corr(ctx):
    base = pd.Series(ctx["grid"]["no_overlay"]).dropna()
    comb = pd.Series(ctx["search"]).dropna()
    both = pd.concat([base, comb], axis=1, join="inner")
    overlay_contrib = both.iloc[:, 1] - both.iloc[:, 0]    # overlay's marginal return stream
    c = float(both.iloc[:, 0].corr(overlay_contrib))
    return {"pass": bool(c <= 0.10), "observed": "leg_corr=%.3f" % c}


def _chk_tracking_breadth(ctx):
    """Breadth thesis: 10-ETF sleeve must track the validated 21-market trend stream
    >=0.5 AND strictly better than the parent's 5-ETF sleeve (computed from grid, free)."""
    tr, _ = trend_returns()
    tr = pd.Series(tr).dropna()
    tr = tr[tr.index < pd.Timestamp(ctx["holdout_start"])]
    base = pd.Series(ctx["grid"]["no_overlay"]).dropna()

    def _contrib(label_returns):
        both = pd.concat([base, pd.Series(label_returns).dropna()], axis=1, join="inner")
        return both.iloc[:, 1] - both.iloc[:, 0]

    o10 = _contrib(ctx["search"])
    o5 = _contrib(ctx["grid"]["sleeve_5"])
    c10 = float(pd.concat([o10, tr], axis=1, join="inner").corr().iloc[0, 1])
    c5 = float(pd.concat([o5, tr], axis=1, join="inner").corr().iloc[0, 1])
    return {"pass": bool(c10 >= 0.50 and c10 > c5),
            "observed": "corr10=%.3f corr5=%.3f (vs canonical trend_returns)" % (c10, c5)}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="amihud_tranched_x_trend10_etf",
    family="combination",
    title="Amihud illiquidity (tranched) x 10-ETF cross-asset trend tail overlay (breadth variant)",
    markets=["US small-cap equities", "cross-asset ETFs (SPY/EFA/EEM/TLT/IEF/LQD/HYG/GLD/SLV/DBC)"],
    data_desc=("Sharadar SEP small-cap panel (closeadj/close/volume; survivorship-clean, delisted "
               "included) + yfinance daily adjusted closes for the fixed 10-ETF trend sleeve and "
               "the declared IWM hedge; canonical trend_returns() used only as a tracking diagnostic."),
    pre_registration=(
        "FROZEN BEFORE BACKTEST. Two-premium book: (A) Amihud illiquidity premium — long top-50 "
        "most-illiquid small-caps passing a $250k median-dollar-volume tradability floor, 63d Amihud "
        "measure, inverse-vol sized, weekly cohorts averaged over 4 staggered tranches, residual "
        "-0.35 IWM trim declared as a hedge sleeve; (B) canonical 12-month-sign trend rule mapped "
        "unchanged onto the FIXED 10-ETF list above (equal risk per line) — the ONLY object mutated "
        "vs the 5-ETF parent is this pre-named instrument list. Overlay at 25% of book risk; both "
        "legs vol-matched to 10% trailing-60d ann. vol; gross capped at 2x; 8bps costs on turnover; "
        "all weights lagged 1 day. SUCCESS CRITERIA (machine-checked as soft expectations): vs the "
        "standalone Amihud grid variant on the identical search window — MaxDD reduced >=20%, Sharpe "
        "degradation <=10%, overlay-vs-leg correlation <= +0.10. BREADTH THESIS (the variant's point): "
        "the 10-ETF sleeve must track trend_returns() at >=0.5 daily correlation AND strictly better "
        "than the parent's 5-ETF sleeve; if it does not, the breadth thesis is falsified and recorded. "
        "Per-bucket stress-window sign decomposition (2015-16/2020/2022) and $5K commission drag are "
        "reported in review, not machine-checked: they require window-level slicing and broker fee "
        "schedules outside ctx, and they gate nothing. Holdout 2022+ is write-once; no parameter "
        "search beyond the 4 declared grid variants."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                                   # primary: 10-ETF sleeve, 25% overlay
        "no_overlay": {"overlay_risk": 0.0},             # standalone Amihud baseline (criteria anchor)
        "overlay_50": {"overlay_risk": 0.5},             # declared sizing variant (DSR burden, honest)
        "sleeve_5": {"sleeve_etfs": PARENT_5},           # parent's 5-ETF sleeve, comparison line ONLY
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=65,                              # 50 equity cohort names + 10 sleeve + IWM
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
    expectations=[
        {"name": "maxdd_cut_20pct",
         "claim": "combined book MaxDD <= 80% of standalone Amihud MaxDD (search window)",
         "check": _chk_maxdd},
        {"name": "sharpe_dilution_le_10pct",
         "claim": "combined Sharpe >= 90% of standalone Amihud Sharpe (search window)",
         "check": _chk_sharpe},
        {"name": "leg_correlation_le_0p1",
         "claim": "overlay marginal returns correlate <= +0.10 with the standalone Amihud leg",
         "check": _chk_leg_corr},
        {"name": "breadth_improves_tracking",
         "claim": "10-ETF sleeve tracks canonical trend_returns() >=0.5 and strictly better than 5-ETF",
         "check": _chk_tracking_breadth},
    ],
)