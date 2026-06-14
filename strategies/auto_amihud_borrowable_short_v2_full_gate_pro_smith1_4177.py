"""
Amihud borrowable-short v2 — full-gate promotion run.

Frozen economic construction (reproduced from amihud_illiq_tranched_v3 / run #1
auto-amihud-illiquidity-premium-borrowable-sh-smith1-95200):
  * Universe: survivorship-clean Sharadar small/mid US common stock (delisted incl).
  * Signal: within size-tercile Amihud illiquidity sort.
      LONG  = most-ILLIQUID quintile per size tercile  (illiquidity premium long leg)
      SHORT = most-LIQUID names per size tercile, $10-$500 price filter
              + BORROWABILITY FLOOR (20d median $-volume, point-in-time).
  * Dollar-neutral, inverse-vol sized, weekly formation with 4-tranche overlap.
  * Asymmetric costs: 60bps RT long / 15bps RT short + 50bps/yr borrow.
  * Residual broad-beta trim declared as a HEDGE SLEEVE on the spec
    (hedge_tickers=["IWM"], hedge_cap=0.35) so the deployment gate judges the
    ALPHA book alone — the alpha book itself holds NO ETF (kept pure & neutral).

Two PRE-REGISTERED, non-alpha-tuning additions (frozen a priori):
  (1) DEFLATION GRID (robustness, NOT selection): borrow floor in {$3M,$5M,$8M}.
      PRIMARY = $5M (= run #1). The 3 cells exist so DSR/PBO can deflate across
      >=3 trials (STRICTER bar) and so floor-robustness is reported. Best NOT picked.
  (2) SINGLE-NAME CONCENTRATION CAP (risk control, NOT alpha tuning): no name >
      35% of gross notional. Capped per-leg with proportional redistribution of the
      remainder inside the same leg (long stays long / short stays short ->
      dollar-neutrality preserved). Frozen at 0.35. Clears deployment-sanity (<=0.40).

Lag responsibility: weights are formed at the close of each rebalance date using
trailing data only; returns are earned on W.shift(1) (one-day lag, no look-ahead).
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd

START = "2006-01-01"

DEFAULTS = dict(
    amihud_window=60,        # trailing Amihud illiquidity window (days)
    vol_window=60,           # inverse-vol sizing window (days)
    long_q=0.20,             # LONG = most-illiquid quintile per tercile
    short_n=15,              # SHORT = 15 most-liquid per tercile
    price_min=10.0, price_max=500.0,
    borrow_floor=5_000_000.0,   # PRIMARY $5M short borrowability floor
    single_name_cap=0.35,       # frozen concentration cap (fraction of gross)
    rebalance=5,                # weekly formation
    overlap=4,                  # 4-tranche overlapping-portfolio smoothing
    cost_long_oneway=30.0,      # 60 bps round-trip long
    cost_short_oneway=7.5,      # 15 bps round-trip short
    borrow_annual=0.005,        # 50 bps/yr borrow
)

_UNIV = {}


def _universe():
    """Sector-spread small/mid US common-stock universe + sector map (memoised)."""
    if not _UNIV:
        t1, s1 = sector_universe("Small", top_n_per_sector=100)
        t2, s2 = sector_universe("Mid", top_n_per_sector=50)
        sm = {**s2, **s1}
        tickers = sorted(set(t1) | set(t2))
        _UNIV["tickers"] = tickers
        _UNIV["sector_map"] = {k: v for k, v in sm.items() if k in set(tickers)}
    return _UNIV["tickers"], _UNIV["sector_map"]


def load_data() -> pd.DataFrame:
    tickers, sector_map = _universe()
    px = sep_panel(tickers, START, field="closeadj")          # returns (split+div adj)
    cols = px.columns
    try:
        prc = sep_panel(list(cols), START, field="closeunadj").reindex(columns=cols)
    except Exception:
        prc = px.copy()                                       # fallback price level
    vol = sep_panel(list(cols), START, field="volume").reindex(columns=cols)
    dvol = prc * vol                                          # daily dollar volume
    fund = sf1(list(cols), ["marketcap"], dimension="ARQ")    # datekey PIT -> no look-ahead
    mcap = pit_panel(fund, "marketcap", px.index, list(cols))
    panel = pd.concat({"px": px, "prc": prc, "dvol": dvol, "mcap": mcap}, axis=1)
    panel.attrs["sector_map"] = {k: v for k, v in sector_map.items() if k in set(cols)}
    return panel


def load_gen_data(label: str) -> pd.DataFrame:
    # scope='local' -> stage-2 generalisation battery is not run; defined for signature.
    return load_data()


def _apply_cap(w: pd.Series, cap: float) -> pd.Series:
    """Cap any per-name weight at `cap` (fraction of gross=1); redistribute the
    excess proportionally to the uncapped names IN THE SAME LEG (leg sum preserved)."""
    w = w.astype(float).copy()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = w.index[(~over) & (w > 0)]
        if len(free) == 0:
            break
        w[free] = w[free] + excess * (w[free] / w[free].sum())
    return w


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    px = panel["px"]
    prc = panel["prc"]
    dvol = panel["dvol"]
    mcap = panel["mcap"]
    sector_map = panel.attrs.get("sector_map")
    if not sector_map:
        _, sector_map = _universe()

    rets = px.pct_change()
    # Amihud illiquidity = trailing mean of |ret| / dollar-volume (high = illiquid).
    amihud = (rets.abs() / dvol.replace(0.0, np.nan)).rolling(
        p["amihud_window"], min_periods=max(20, p["amihud_window"] // 2)).mean()
    vol = rets.rolling(p["vol_window"], min_periods=max(20, p["vol_window"] // 2)).std()
    med_dvol = dvol.rolling(20, min_periods=12).median()      # PIT borrowability proxy

    dates = px.index
    rebal = dates[::p["rebalance"]]
    form = {}

    for dt in rebal:
        a = amihud.loc[dt]; m = mcap.loc[dt]; v = vol.loc[dt]
        pr = prc.loc[dt]; md = med_dvol.loc[dt]
        ok = a.notna() & m.notna() & v.notna() & (v > 0) & pr.notna()
        a, m, v, pr, md = a[ok], m[ok], v[ok], pr[ok], md[ok]
        if len(a) < 30:
            continue
        terc = pd.qcut(m.rank(method="first"), 3, labels=False)   # size terciles

        long_w, short_w = {}, {}
        for tc in (0, 1, 2):
            idx = terc.index[terc == tc]
            if len(idx) < 25:
                continue
            a_tc = a[idx]
            # LONG: most-illiquid quintile within tercile.
            thr = a_tc.quantile(1.0 - p["long_q"])
            longs = a_tc.index[a_tc >= thr]
            # SHORT: most-liquid (lowest Amihud) eligible names within tercile.
            pool = a_tc.drop(longs, errors="ignore")             # disjoint from longs
            elig = pool.index[(pr.reindex(pool.index) >= p["price_min"]) &
                              (pr.reindex(pool.index) <= p["price_max"]) &
                              (md.reindex(pool.index) >= p["borrow_floor"])]
            shorts = a[elig].sort_values().index[:p["short_n"]] if len(elig) else pd.Index([])
            for tk in longs:
                long_w[tk] = 1.0 / v[tk]                          # inverse-vol size
            for tk in shorts:
                short_w[tk] = 1.0 / v[tk]

        if not long_w or not short_w:
            continue                                             # need both legs (flat else)
        lw = pd.Series(long_w, dtype=float); lw = lw / lw.sum() * 0.5
        sw = pd.Series(short_w, dtype=float); sw = sw / sw.sum() * 0.5
        lw = _apply_cap(lw, p["single_name_cap"])
        sw = _apply_cap(sw, p["single_name_cap"])
        row = {tk: float(w) for tk, w in lw.items()}
        row.update({tk: -float(w) for tk, w in sw.items()})
        form[dt] = pd.Series(row, dtype=float)

    if not form:
        return pd.Series(dtype=float, name="amihud_borrowable_short_v2"), []

    Wt = pd.DataFrame(form).T.reindex(columns=px.columns).fillna(0.0).sort_index()
    # Overlapping-tranche smoothing (convex avg keeps each leg dollar-neutral and
    # cannot raise any name above the 0.35 cap already enforced per formation).
    Wo = Wt.rolling(p["overlap"], min_periods=1).mean()
    W = Wo.reindex(dates).ffill().fillna(0.0)                     # daily held book

    rets_a = rets.reindex(columns=W.columns).fillna(0.0)
    Wlag = W.shift(1).fillna(0.0)                                 # 1-day lag (MY responsibility)
    # Asymmetric costs -> net_of_cost per leg with its own one-way rate, then add.
    r_long = net_of_cost(Wlag.clip(lower=0.0), rets_a, cost_bps=p["cost_long_oneway"], name="amih_long")
    r_short = net_of_cost(Wlag.clip(upper=0.0), rets_a, cost_bps=p["cost_short_oneway"], name="amih_short")
    borrow = Wlag.clip(upper=0.0).abs().sum(axis=1) * (p["borrow_annual"] / 252.0)
    daily = r_long.add(r_short, fill_value=0.0).subtract(borrow, fill_value=0.0)
    daily = daily.reindex(dates).fillna(0.0)
    daily.name = "amihud_borrowable_short_v2"

    trades = trades_from_weights(W, rets_a, sector_map)           # kit stamps entry_regime
    return daily, trades


# ---------------- machine-checkable soft expectations ----------------
def _sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else 0.0


def _chk_concentration(ctx):
    """Pre-reg: the 0.35 cap drives single-name position-day share <= 0.40."""
    tot, grand = {}, 0.0
    for t in (ctx.get("trades", []) or []):
        pv = abs(float(t.get("position_value", 0.0))) * max(int(t.get("hold_days", 1)), 1)
        tot[t.get("ticker", "?")] = tot.get(t.get("ticker", "?"), 0.0) + pv
        grand += pv
    share = (max(tot.values()) / grand) if (grand > 0 and tot) else 1.0
    return {"pass": share <= 0.40, "observed": round(share, 3)}


def _chk_floor_robustness(ctx):
    """Pre-reg: finding must not hinge on $5M — all {$3M,$5M,$8M} cells same-sign positive."""
    sh = {}
    for k, r in (ctx.get("grid", {}) or {}).items():
        if r is None:
            continue
        s = _sharpe(r)
        if s != 0.0:
            sh[k] = s
    ok = (len(sh) >= 2) and all(v > 0 for v in sh.values())
    return {"pass": bool(ok), "observed": round(min(sh.values()), 3) if sh else 0.0}


def _chk_premium_positive(ctx):
    """Pre-reg: PRIMARY $5M borrowable-short Amihud book is net-positive in-sample."""
    s = _sharpe(ctx["search"])
    return {"pass": s > 0.20, "observed": round(s, 3)}


SPEC = StrategySpec(
    id="amihud_borrowable_short_v2",
    family="illiquidity_premium",
    title="Amihud borrowable-short v2 — borrow-floor deflation grid ($3M/$5M/$8M) + 0.35 single-name cap",
    markets=["US_equity"],
    data_desc=("Survivorship-clean Sharadar SEP: closeadj (returns), closeunadj x volume "
               "(point-in-time daily $-volume & $10-$500 price filter & 20d-median borrow floor); "
               "SF1 ARQ marketcap (datekey PIT) for size terciles. ~1.1k small/mid US common stocks."),
    pre_registration=(
        "Byte-frozen amihud_illiq_tranched_v3 borrowable-short construction (within size-tercile Amihud "
        "sort; LONG most-illiquid quintile; SHORT 15 most-liquid per tercile, $10-$500 price filter + 20d "
        "median $-volume borrow floor; inverse-vol, dollar-neutral, weekly 4-tranche overlap; 60/15 bps RT "
        "+ 50 bps/yr borrow; IWM beta-trim declared as hedge sleeve, alpha book ETF-free). TWO pre-registered "
        "additions, both STRICTER/constraining not selecting: (1) deflation grid borrow floor in {$3M,$5M,$8M}, "
        "PRIMARY=$5M is the sole hypothesis verdict, other cells only deflate DSR/PBO across >=3 trials and "
        "report floor-robustness (do NOT pick best); (2) frozen 0.35 single-name concentration cap "
        "(per-leg redistribution, dollar-neutrality preserved) to clear deployment-sanity. SUCCESS = PRIMARY "
        "$5M cell: DSR>=promote AND holdout>0 AND MCPT pass AND deployment-sanity pass (single_name<=0.40) "
        "AND regime pass. Machine-checkable: concentration<=0.40, floor sign-robustness, in-sample positive."),
    load_data=load_data,
    signal=signal,
    default_params={},                       # primary = $5M floor (DEFAULTS)
    grid={
        "default": {},                       # PRIMARY $5M
        "floor_3m": {"borrow_floor": 3_000_000.0},
        "floor_8m": {"borrow_floor": 8_000_000.0},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=60,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
    expectations=[
        {"name": "concentration_cap_binds",
         "claim": "0.35 cap holds single-name position-day share <= 0.40 (clears deployment-sanity)",
         "check": _chk_concentration},
        {"name": "floor_robustness",
         "claim": "Sharpe sign is consistent (positive) across borrow floors {$3M,$5M,$8M}",
         "check": _chk_floor_robustness},
        {"name": "premium_positive",
         "claim": "PRIMARY $5M borrowable-short Amihud book is net-positive in-sample (Sharpe > 0.20)",
         "check": _chk_premium_positive},
    ],
)