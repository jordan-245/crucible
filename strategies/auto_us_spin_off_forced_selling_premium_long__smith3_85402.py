"""
US New-Listing Neglect / Forced-Selling Premium — long NEWLY LISTED small/mid-cap
names after a one-month seasoning window, 12-month hold, equal-weight, monthly
rebalance, declared IWM beta hedge. Survivorship-clean (Sharadar SEP, delisted incl.).

CORRECTION vs prior draft: the prior version tried to build the spin-off-child
cohort from SHARADAR ACTIONS via a non-existent `actions_table` adapter and a
`nasdaqdatalink` fallback — NEITHER exists in this harness (ImportError /
ModuleNotFoundError). The ACTIONS table is NOT reachable through the tested SDK
adapters, so the registered "ACTIONS spin-off children" cohort is NOT constructible
here. This module therefore registers the closest IMPLEMENTABLE cohort from owned
data, and says so honestly: post-1999 NEW LISTINGS in the liquid small/mid universe
(first SEP print after an 18-month burn-in past data start), screened to be
REVENUE-POSITIVE point-in-time at entry (excludes SPACs / blank-check shells, which
share no economics with the neglect mechanism). This cohort contains spin-offs,
carve-outs and operating-company IPOs; it is a DIFFERENT, broader premium than the
pure spin-off cohort and the pre-registration states that explicitly — including
the bear case (Ritter long-run new-issue underperformance). The test is allowed
to fail.

LOOK-AHEAD HYGIENE: cohort membership comes from the first SEP print (public);
entry is the 2nd month-end on/after listing (>= ~1 full calendar month of
seasoning); the revenue screen uses pit_panel on sf1 datekey (filing-date PIT,
never calendardate); all weights are built at month-end from trailing information
and shift(1)-lagged before net_of_cost / trades_from_weights; the hedge beta uses
a trailing 60d window only.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

# ---- frozen constants (single pre-registered config) ----
START = "1998-01-01"
LISTING_MIN = "1999-07-01"     # 18m burn-in: a "first print" at data start is NOT a new listing
PRICE_MIN = 3.0                # unadjusted close at entry
DV_MIN = 1_000_000.0           # 20d median dollar volume at entry
COST_BPS = 25.0                # realistic small/mid-cap cost level
BETA_LB = 60                   # trailing window for hedge beta
HEDGE_CAP = 0.5                # max IWM short, capped (whitelisted ETF)


def load_data() -> pd.DataFrame:
    """Panel with MultiIndex columns: ('px', t)=closeadj, ('close', t)=unadjusted,
    ('vol', t)=share volume, ('rev', t)=PIT quarterly revenue — new-listing cohort
    only — plus ('hedge','IWM')."""
    uni = sorted(
        set(us_universe(marketcap="Small", include_delisted=True, top_n=1200))
        | set(us_universe(marketcap="Mid", include_delisted=True, top_n=1200))
    )
    px_all = sep_panel(uni, START, field="closeadj")
    lm = pd.Timestamp(LISTING_MIN)
    cohort = sorted(
        t for t in px_all.columns
        if px_all[t].first_valid_index() is not None
        and px_all[t].first_valid_index() >= lm
    )
    px = px_all[cohort]
    close = sep_panel(cohort, START, field="close")
    vol = sep_panel(cohort, START, field="volume")
    fund = sf1(cohort, ["revenue"], dimension="ARQ")
    rev = pit_panel(fund, "revenue", px.index, cohort)  # datekey-based PIT, ffilled
    iwm = yf_panel(["IWM"], START)  # ETF hedge sleeve — declared on the spec
    panel = pd.concat(
        {"px": px, "close": close[cohort], "vol": vol[cohort], "rev": rev, "hedge": iwm},
        axis=1,
    )
    return panel


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    me = idx.to_series().groupby(idx.to_period("M")).max()
    return pd.DatetimeIndex(me.values)


def _entries(panel: pd.DataFrame, hold_months: int = 12):
    """Frozen entry schedule. For each new listing: anchor = first SEP print;
    entry = the 2nd month-end trading day on/after the anchor (>= ~1 full calendar
    month of seasoning, skipping the heaviest flow window). One-shot eligibility AT
    entry (unadjusted close >= $3, 20d median dollar volume >= $1M, PIT quarterly
    revenue > 0 — SPAC/shell exclusion); no retry — frozen. Exit = hold_months
    month-ends after entry (or held to sample end)."""
    px, close, vol, rev = panel["px"], panel["close"], panel["vol"], panel["rev"]
    mes = _month_ends(px.index)
    dv = (close * vol).rolling(20, min_periods=10).median()
    out = []
    for t in px.columns:
        f = px[t].first_valid_index()
        if f is None:
            continue
        i = int(mes.searchsorted(f, side="left")) + 1  # 2nd month-end => ~1 month seasoning
        if i >= len(mes):
            continue  # too recent to season
        e = mes[i]
        c = close.at[e, t] if e in close.index else np.nan
        d = dv.at[e, t] if e in dv.index else np.nan
        r = rev.at[e, t] if (e in rev.index and t in rev.columns) else np.nan
        if not (np.isfinite(c) and c >= PRICE_MIN and np.isfinite(d) and d >= DV_MIN):
            continue
        if not (np.isfinite(r) and r > 0):
            continue  # no filed revenue by entry -> SPAC/shell/blank-check: excluded
        j = i + hold_months
        x = mes[j] if j < len(mes) else None
        out.append({"ticker": t, "entry": e, "exit": x, "entry_dv": float(d)})
    return out, mes


def signal(panel, hold_months: int = 12, hedge: bool = True,
           hedge_cap: float = HEDGE_CAP, cost_bps: float = COST_BPS):
    px = panel["px"]
    rets = px.pct_change()
    entries, mes = _entries(panel, hold_months=hold_months)

    # --- monthly membership -> equal-weight long book ---
    W_me = pd.DataFrame(0.0, index=mes, columns=px.columns)
    for e in entries:
        if e["exit"] is not None:
            sl = (mes >= e["entry"]) & (mes < e["exit"])
        else:
            sl = mes >= e["entry"]
        W_me.loc[sl, e["ticker"]] = 1.0
    n = W_me.sum(axis=1)
    W_me = W_me.div(n.replace(0.0, np.nan), axis=0).fillna(0.0)

    # monthly weights held until next month-end; zero out anything past its last print
    W = W_me.reindex(px.index, method="ffill").fillna(0.0)
    W = W.where(px.notna(), 0.0)  # delisted mid-hold -> weight goes to cash, no re-norm

    rets_all = rets.copy()
    if hedge:
        iwm_px = panel["hedge"]["IWM"]
        iwm_ret = iwm_px.pct_change()
        # trailing 60d book beta vs IWM (data through t only), sampled at month-ends,
        # held for the month; the final shift(1) makes it effective t+1.
        book = (W.shift(1).fillna(0.0) * rets).sum(axis=1)
        beta = book.rolling(BETA_LB).cov(iwm_ret) / iwm_ret.rolling(BETA_LB).var()
        h_me = beta.reindex(mes).clip(lower=0.0, upper=hedge_cap).fillna(0.0)
        W["IWM"] = (-h_me).reindex(px.index, method="ffill").fillna(0.0)
        rets_all["IWM"] = iwm_ret

    # LAG: weights built at t are tradeable t+1 — the shift is ours and it is here.
    Wl = W.shift(1).fillna(0.0)

    daily = net_of_cost(Wl, rets_all, cost_bps=cost_bps, name="new_listing_neglect")
    smap = {t: "NewListing" for t in px.columns}
    smap["IWM"] = "ETF-Hedge"
    trades = trades_from_weights(Wl, rets_all, smap)
    return daily, trades


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': the mechanism (post-listing neglect/forced flows in small/mid
    # new listings) has no disjoint generalization universe by construction —
    # cohort membership IS the universe definition.
    raise NotImplementedError("scope='local' — no generalization universes declared")


# ---------------- machine-checkable soft expectations ----------------

def _chk_subperiods(ctx):
    s = ctx["search"].dropna()
    h = len(s) // 2
    a = float(s.iloc[:h].mean() * 252)
    b = float(s.iloc[h:].mean() * 252)
    return {"pass": bool(a > 0 and b > 0), "observed": f"ann_ret half1={a:.2%}, half2={b:.2%}"}


def _chk_hold_days(ctx):
    hd = [t["hold_days"] for t in ctx["trades"] if t["ticker"] != "IWM"]
    med = float(np.median(hd)) if hd else 0.0
    return {"pass": bool(med >= 150), "observed": f"median hold_days={med:.0f} (n={len(hd)})"}


def _chk_size_monotonic(ctx):
    # forced-selling/neglect fingerprint: least-liquid entries should earn the most.
    panel = ctx["panel"]
    px = panel["px"]
    hs = pd.Timestamp(ctx["holdout_start"])
    entries, _ = _entries(panel)
    rows = []
    for e in entries:
        if e["entry"] >= hs or e["exit"] is None or e["exit"] >= hs:
            continue  # search-window-complete events only
        seg = px.loc[e["entry"]:e["exit"], e["ticker"]].dropna()
        if len(seg) < 60 or seg.iloc[0] <= 0:
            continue
        rows.append((e["entry_dv"], float(seg.iloc[-1] / seg.iloc[0] - 1.0)))
    if len(rows) < 30:
        return {"pass": False, "observed": f"insufficient events ({len(rows)})"}
    df = pd.DataFrame(rows, columns=["dv", "ret"])
    q = pd.qcut(df["dv"], 3, labels=False, duplicates="drop")
    lo, hi = float(df["ret"][q == q.min()].mean()), float(df["ret"][q == q.max()].mean())
    return {"pass": bool(lo > hi), "observed": f"12m ret small-tercile={lo:.2%} vs large-tercile={hi:.2%}"}


def _chk_hedge_cuts_beta(ctx):
    iwm = ctx["panel"]["hedge"]["IWM"].pct_change()

    def beta(s):
        s = s.dropna()
        i = iwm.reindex(s.index)
        m = i.notna() & s.notna()
        if m.sum() < 100:
            return np.nan
        return float(np.cov(s[m], i[m])[0, 1] / np.var(i[m]))

    bh, bu = beta(ctx["grid"]["default"]), beta(ctx["grid"]["unhedged"])
    ok = np.isfinite(bh) and np.isfinite(bu) and abs(bh) < abs(bu)
    return {"pass": bool(ok), "observed": f"IWM beta hedged={bh:.2f} vs unhedged={bu:.2f}"}


SPEC = StrategySpec(
    id="new_listing_neglect_v1",
    family="event_premia",
    title="US New-Listing Neglect Premium — revenue-screened new listings, 1m seasoning, 12m hold, IWM-hedged",
    markets=["US small/mid-cap equities (post-1999 new-listing cohort)"],
    data_desc=("Sharadar SEP survivorship-clean closeadj/close/volume (delisted incl.) on the "
               "liquid Small+Mid universe (us_universe, top_n-bounded); cohort = tickers whose "
               "first SEP print is on/after 1999-07-01 (18m burn-in past data start so data-start "
               "artifacts are not 'listings'); SF1 ARQ revenue via pit_panel (datekey PIT) for the "
               "SPAC/shell exclusion at entry; IWM Close via yf_panel for the declared hedge sleeve."),
    pre_registration=(
        "HONEST COHORT SUBSTITUTION, stated up front: the original proposal's cohort (spin-off "
        "CHILD tickers from SHARADAR ACTIONS) is NOT constructible in this harness — the ACTIONS "
        "table is not exposed by any tested SDK adapter. This registration therefore tests the "
        "closest implementable cohort from owned data: post-1999 NEW LISTINGS (first SEP print) "
        "in the liquid small/mid universe, revenue-positive point-in-time at entry (excludes "
        "SPACs/blank-check shells). This is a BROADER premium than pure spin-offs (it includes "
        "operating-company IPOs and carve-outs) and the known bear case is Ritter long-run "
        "new-issue underperformance — a clean FAIL here is informative and acceptable. FROZEN "
        "design: entry at the 2nd month-end trading day on/after the first SEP print (~1 full "
        "month seasoning skips the peak flow window and stale-price entries); one-shot "
        "eligibility at entry: unadjusted close >= $3, 20d median dollar volume >= $1M, PIT "
        "quarterly revenue > 0 (no retry). Hold 12 months of month-ends, then exit; equal-weight "
        "across all held names; monthly rebalance only. Declared hedge sleeve: short IWM sized "
        "to trailing-60d book beta, clipped to [0, 0.5] gross, rebalanced monthly "
        "(hedge_tickers/hedge_cap on spec). Costs 25bps on turnover. All weights shift(1)-lagged. "
        "PRIMARY = the single default config; the ONLY grid variant is 'unhedged', declared "
        "solely so the hedge-beta expectation is machine-checkable (no parameter search). "
        "Mechanism claims are machine-checked: (a) positive in both search-window halves, "
        "(b) median hold >= 150 trading days, (c) size-tercile monotonicity — least-liquid "
        "entries outperform most-liquid (forced-selling/neglect fingerprint; if this fails the "
        "premium story is wrong even if returns pass), (d) hedge reduces |IWM beta| vs the "
        "unhedged variant. Local scope by economics: cohort membership IS the universe "
        "definition, so there is no disjoint generalization universe; validation is sub-period "
        "robustness + beta-aware MCPT + holdout."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "unhedged": {"hedge": False},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=40,
    hedge_tickers=["IWM"],
    hedge_cap=HEDGE_CAP,
    expectations=[
        {"name": "positive_both_halves",
         "claim": "net returns positive in both halves of the search window",
         "check": _chk_subperiods},
        {"name": "low_turnover_book",
         "claim": "12-month holds -> median trade hold >= 150 trading days",
         "check": _chk_hold_days},
        {"name": "size_tercile_monotonic",
         "claim": "smallest-dollar-volume entries outperform largest (forced-selling/neglect fingerprint)",
         "check": _chk_size_monotonic},
        {"name": "hedge_cuts_beta",
         "claim": "IWM sleeve reduces |market beta| vs the unhedged variant",
         "check": _chk_hedge_cuts_beta},
    ],
)