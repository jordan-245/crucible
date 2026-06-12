"""
earnings_window_premium_v1
==========================
Frazzini-Lamont (2007) earnings-announcement premium, harvested as a predicted-event
long book in US small/mid caps, index-hedged with a declared IWM sleeve.

Mechanism: compensation for BEARING concentrated idiosyncratic event risk through the
scheduled announcement window — paid risk-bearing, not a direction forecast.

Point-in-time discipline: the event calendar is built ONLY from past SF1 datekeys
(filing dates). At each filing j (with >=4 prior filings and a frozen cadence-regularity
filter), the NEXT announcement is predicted as datekey_j + 91 calendar days. The trading
window opens ~86 days AFTER the information (datekey_j) became public — no look-ahead.
Execution weights are additionally shift(1)-lagged before costing.

FROZEN per proposal: EQUAL-WEIGHT in-window book (5% per-name cap, capped residual NOT
redistributed) and an index hedge sized to the FULL trailing-60d book beta (per unit of
gross, scaled by current gross) — no half-beta cap, per the beta-confound lesson.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2005-01-01"
_CACHE = {}


# ----------------------------------------------------------------------------- universe
def _universe():
    """Small+Mid cap, sector-spread, survivorship-clean. ~1200 names."""
    if "univ" not in _CACHE:
        t_s, s_s = sector_universe(marketcap="Small", top_n_per_sector=55)
        t_m, s_m = sector_universe(marketcap="Mid", top_n_per_sector=55)
        sector_map = {**s_s, **s_m}
        sector_map["IWM"] = "ETF-Hedge"  # declared hedge sleeve (hedge_tickers on spec)
        _CACHE["univ"] = (sorted(set(t_s) | set(t_m)), sector_map)
    return _CACHE["univ"]


# ----------------------------------------------------------------------- event calendar
def _event_point_mask(idx, cols, f):
    """1.0 on the first trading day >= (last datekey + 91cd), per ticker.

    FROZEN regularity filter: a prediction is only made at filing j if the median
    |gap - 91d| of the last <=4 inter-filing gaps is <= 12 days (irregular filers
    are excluded at that point in time, not globally — PIT-safe).
    """
    evt = pd.DataFrame(0.0, index=idx, columns=cols)
    f = f.dropna(subset=["datekey"])
    for tic, grp in f.groupby("ticker"):
        if tic not in evt.columns:
            continue
        dks = pd.DatetimeIndex(sorted(pd.to_datetime(grp["datekey"]).unique()))
        if len(dks) < 5:
            continue
        gaps = np.diff(dks.values).astype("timedelta64[D]").astype(float)
        col = evt.columns.get_loc(tic)
        for j in range(3, len(dks)):  # need >=4 prior filings (>=3 observed gaps)
            recent = gaps[max(0, j - 4):j]
            if np.median(np.abs(recent - 91.0)) > 12.0:
                continue
            pred = dks[j] + pd.Timedelta(days=91)
            loc = idx.searchsorted(pred)
            if loc >= len(idx):
                continue
            evt.iat[loc, col] = 1.0
    return evt


# ------------------------------------------------------------------------------- panels
def _build_panel(tickers):
    px = sep_panel(tickers, START, field="closeadj")          # survivorship-clean returns
    cl = sep_panel(tickers, START, field="close").reindex(px.index)
    vol = sep_panel(tickers, START, field="volume").reindex(px.index)

    f = sf1(tickers, fields=["eps"], dimension="ARQ")          # datekey = filing date (PIT)
    evt = _event_point_mask(px.index, px.columns, f)

    # Deployable-liquidity filter (Amihud-survivor lesson): $5+ price, top-2/3 by 60d
    # dollar volume. shift(1) -> eligibility uses only prior-day information.
    dv = (cl * vol).rolling(60, min_periods=40).mean()
    elig = ((cl >= 5.0) & (dv.rank(axis=1, pct=True) >= (1.0 / 3.0)))
    elig = elig.shift(1).fillna(False).astype(float)

    iwm = yf_panel(["IWM"], START).reindex(px.index).ffill()
    iwm.columns = ["IWM"]

    return pd.concat({"px": px, "evt": evt, "elig": elig, "hpx": iwm}, axis=1)


def load_data():
    tickers, _ = _universe()
    return _build_panel(tickers)


def load_gen_data(label):
    # scope='local' — harness does not run stage-2 breadth; these splits exist for the
    # pre-registered manual robustness checks (small-vs-mid depth validation).
    if label == "small_only":
        t, _ = sector_universe(marketcap="Small", top_n_per_sector=55)
    elif label == "mid_only":
        t, _ = sector_universe(marketcap="Mid", top_n_per_sector=55)
    else:
        raise KeyError(f"unknown generalization universe: {label}")
    return _build_panel(sorted(set(t)))


# ------------------------------------------------------------------------------- signal
def signal(panel, win_pre=5, win_post=1, cap=0.05, placebo_shift=0, **params):
    """Long names inside their predicted announcement window [-win_pre, +win_post]
    trading days, EQUAL-WEIGHT (frozen per proposal), per-name cap, full trailing-beta
    IWM hedge.

    Rebalance is DAILY (frozen deviation from the weekly default: event windows open
    and close within ~7 trading days; a weekly grid would miss entries entirely).
    placebo_shift (trading days) exists ONLY for the pre-registered placebo
    expectation check — never a grid variant.
    """
    _, sector_map = _universe()
    px = panel["px"]
    evt = panel["evt"].fillna(0.0)
    elig = panel["elig"].fillna(0.0)
    iwm_px = panel["hpx"]["IWM"]

    rets = px.pct_change()
    r_iwm = iwm_px.pct_change()

    e = evt.shift(placebo_shift).fillna(0.0) if placebo_shift else evt
    # mark trading days [event - win_pre, event + win_post]; the event date itself was
    # predictable ~86 days earlier (from the prior filing), so the pre-window is PIT-safe
    ind = sum(e.shift(k) for k in range(-win_pre, win_post + 1))
    in_win = (ind > 0) & (elig > 0)

    # FROZEN: equal-weight across the in-window book, 5% per-name cap;
    # capped residual NOT redistributed -> thin event days run lower gross (no mirage books)
    n_in = in_win.sum(axis=1)
    w = (in_win.astype(float)
         .div(n_in.where(n_in > 0), axis=0)
         .clip(upper=cap)
         .fillna(0.0))

    # declared index hedge: trailing-60d book beta vs IWM, computed per unit of gross
    # (so it is a true beta of the long book), lagged, then scaled by CURRENT gross.
    # No half-beta cap: under-hedging would reintroduce the beta confound that killed
    # the ETF-hedged Amihud sibling.
    gross = w.sum(axis=1)
    g_lag = gross.shift(1)
    r_book = (w.shift(1) * rets).sum(axis=1)
    r_unit = r_book.div(g_lag.where(g_lag > 0))
    beta = (r_unit.rolling(60, min_periods=40).cov(r_iwm)
            / r_iwm.rolling(60, min_periods=40).var()).clip(0.0, 1.5).shift(1).fillna(0.0)
    hedge = -(beta * gross)

    W = w.copy()
    W["IWM"] = hedge
    rets_all = pd.concat([rets, r_iwm.rename("IWM")], axis=1)

    W_lag = W.shift(1)  # explicit execution lag: weights formed on day t trade at t+1
    daily = net_of_cost(W_lag, rets_all, cost_bps=8.0, name="earnings_window_premium_v1")
    trades = trades_from_weights(W_lag, rets_all, sector_map)
    return daily, trades


# ------------------------------------------------------------------- soft expectations
def _check_placebo(ctx):
    """Pre-registered placebo: identical book on pseudo-events shifted +31 trading days
    (~45 calendar) must show no premium — proves the return loads on the event window,
    not a hidden universe tilt. Uses the ONE allowed extra signal() call."""
    hs = pd.Timestamp(ctx["holdout_start"])
    pl, _ = signal(ctx["panel"], placebo_shift=31)
    pl = pl[pl.index < hs]
    base_ann = float(ctx["search"].mean() * 252)
    plc_ann = float(pl.mean() * 252)
    ok = (base_ann > 0) and (plc_ann < 0.5 * base_ann)
    return {"pass": bool(ok), "observed": f"base={base_ann:.4f}/yr placebo={plc_ann:.4f}/yr"}


def _check_breadth(ctx):
    """Deployment-sanity pre-check: median concurrent in-window eligible names >= 15."""
    hs = pd.Timestamp(ctx["holdout_start"])
    p = ctx["panel"]
    evt = p["evt"].fillna(0.0)
    elig = p["elig"].fillna(0.0)
    ind = sum(evt.shift(k) for k in range(-5, 2))
    n = ((ind > 0) & (elig > 0)).sum(axis=1)
    n = n[n.index < hs].iloc[260:]  # skip warmup
    med = float(n.median())
    return {"pass": med >= 15.0, "observed": med}


def _check_short_holds(ctx):
    """Mechanism check: this is an event book — median equity hold must be ~the window
    (<=12 trading days), the opposite holding pattern of every continuous-characteristic
    book in the experiment history."""
    hd = [t["hold_days"] for t in ctx["trades"] if t["ticker"] != "IWM"]
    if not hd:
        return {"pass": False, "observed": 0}
    med = float(np.median(hd))
    return {"pass": med <= 12.0, "observed": med}


# --------------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="earnings_window_premium_v1",
    family="event_premium",
    title="Earnings-announcement-window risk premium (predicted-event long book, IWM-hedged, US small/mid)",
    markets=["us_equity_small_mid"],
    data_desc=("OWNED Sharadar: SF1 ARQ datekey (filing date) as the PIT event calendar; "
               "SEP closeadj/close/volume (survivorship-clean, delisted incl.); "
               "FREE yfinance IWM for the declared hedge sleeve."),
    pre_registration=(
        "HYPOTHESIS: stocks earn a premium through their scheduled earnings-announcement "
        "window as compensation for bearing concentrated idiosyncratic event risk "
        "(Frazzini-Lamont 2007); decayed in large caps, plausibly alive in small/mid. "
        "FROZEN DESIGN: predict next announcement = prior SF1 datekey + 91cd (>=4 prior "
        "filings, median |gap-91|<=12d regularity filter, all PIT); hold [-5,+1] trading "
        "days around the predicted date. The window is deliberately PRE-heavy because "
        "datekey is the SEC filing date, which trails the earnings press release by 0-5 "
        "days for most small/mid caps — [-5,+1] around the predicted filing straddles the "
        "actual announcement. EQUAL-WEIGHT book (frozen per proposal), 5% per-name cap, "
        "capped residual not redistributed; FULL trailing-60d-beta IWM hedge (beta of the "
        "unit-gross book, scaled by current gross, beta clipped [0,1.5]) declared on the "
        "spec — no half-beta cap, per the beta-confound gate that killed the ETF-hedged "
        "Amihud sibling. DAILY rebalance (frozen deviation from the weekly default: "
        "windows open/close within ~7 trading days). 8bps costs on turnover. "
        "MACHINE-CHECKED CLAIMS: (1) placebo book on +31-trading-day pseudo-events shows "
        "no premium; (2) median concurrent book breadth >=15 names; (3) median equity "
        "hold <=12 trading days. NOT machine-checked (needs per-name abnormal-volume "
        "study, too expensive for one signal call): the empirical datekey-vs-press-release "
        "lag distribution — covered qualitatively by the asymmetric window choice above. "
        "This is paid risk-bearing through the window, NOT PEAD and NOT a surprise/"
        "direction forecast. Standalone test; no trend blend (2026-06-08 dilution lesson)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "wide_window": {"win_pre": 7, "win_post": 2},
        "tight_window": {"win_pre": 3, "win_post": 1},
        "cap8": {"cap": 0.08},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=40,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
    expectations=[
        {"name": "placebo_no_premium",
         "claim": "shifted pseudo-event book (+31td) earns <50% of the real-window premium",
         "check": _check_placebo},
        {"name": "book_breadth",
         "claim": "median concurrent in-window eligible names >= 15 (no mirage book)",
         "check": _check_breadth},
        {"name": "event_holding_pattern",
         "claim": "median equity hold <= 12 trading days (event book, not a characteristic tilt)",
         "check": _check_short_holds},
    ],
)