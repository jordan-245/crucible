"""
Amihud illiquidity premium — BORROWABLE-SHORT variant.

ONE question under test (frozen, single spec, no search): does the validated Amihud
illiquidity premium (deployed as amihud_illiq_tranched_v3) SURVIVE when the short leg is
restricted to plausibly-borrowable names? The construction reproduces v3 (size-tercile
Amihud sort; LONG = most-illiquid quintile per size tercile; SHORT = top-15 most-liquid
per size tercile, $10-$500 price band, 10% single-name cap, inverse-vol sizing, weekly
rebalance, asymmetric costs 60bps RT long / 15bps RT short + 50bps/yr borrow, residual
IWM beta-trim sleeve) with EXACTLY ONE CHANGE: every short candidate must clear a FROZEN
borrowability proxy = 20-day trailing median dollar volume >= $5,000,000, computed strictly
point-in-time (data up to the formation date only). Failing shorts are DROPPED (not replaced
by reaching deeper down the liquidity rank); the dollar-neutral target re-balances across the
surviving shorts. Long leg UNCHANGED. PRIMARY VERDICT = borrowable-short book.

No external side effects. Owned Sharadar data only ($0).
"""
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd

# ----------------------------- FROZEN CONSTANTS ------------------------------
START               = "2004-01-01"
HOLDOUT_START       = "2022-01-01"
TOP_N_PER_SECTOR    = 50          # Small + Mid -> ~1000-1100 liquid names (bounded for CPCV)

AMIHUD_WINDOW       = 63          # ~3m trailing Amihud illiquidity (|ret|/$vol)
AMIHUD_MINP         = 40
DV_WINDOW           = 20          # 20d trailing median $-volume = borrowability proxy
DV_MINP             = 12
VOL_WINDOW          = 20          # trailing vol for inverse-vol sizing
VOL_MINP            = 12

BORROW_FLOOR        = 5_000_000.0 # FROZEN a-priori $5M ADV borrowability floor (NOT searched)
N_SIZE_TERCILES     = 3
LONG_Q              = 0.20        # most-illiquid quintile per size tercile (LONG)
LONG_MIN            = 2
SHORT_TOP_N         = 15          # top-15 most-liquid per tercile (SHORT candidates)
PRICE_MIN, PRICE_MAX = 10.0, 500.0
SINGLE_NAME_CAP     = 0.10
REBAL_DAYS          = 5           # weekly
MIN_NAMES           = 75          # min valid names on a formation date (else skip)
MIN_TERCILE         = 20

COST_LONG_BPS       = 30.0        # 60 bps round-trip on long turnover  (one-way = RT/2)
COST_SHORT_BPS      = 7.5         # 15 bps round-trip on short turnover (one-way = RT/2)
BORROW_BPS_YR       = 50.0        # 50 bps/yr financing on short notional

_STATE = {"sector_map": {}}       # sector map for the trade ledger (kit contract)
_DIAG_CACHE = {}                  # cache for soft-expectation diagnostics


# ------------------------------- UNIVERSE/DATA -------------------------------
def _build_universe():
    """Sector-spread Small+Mid US common stock (survivorship-clean, delisted incl)."""
    tickers, smap = [], {}
    for cap in ("Small", "Mid"):
        try:
            t, m = sector_universe(marketcap=cap, top_n_per_sector=TOP_N_PER_SECTOR)
        except Exception:
            continue
        tickers += list(t)
        smap.update(dict(m))
    return sorted(set(tickers)), smap


def load_data() -> pd.DataFrame:
    """Wide panel with column MultiIndex (field, ticker). Fields: px (closeadj, returns),
    close (unadjusted, $-volume + price band), vol (volume), mcap (PIT marketcap)."""
    tickers, smap = _build_universe()
    _STATE["sector_map"] = smap

    px    = sep_panel(tickers, START, field="closeadj")   # split+div adjusted -> returns
    close = sep_panel(tickers, START, field="close")      # as-traded price -> $vol, band
    vol   = sep_panel(tickers, START, field="volume")     # as-traded volume -> $vol
    cols  = px.columns

    sf    = sf1(tickers, ["marketcap"], dimension="ARQ")
    mcap  = pit_panel(sf, "marketcap", px.index, list(cols))  # datekey-based, no lookahead

    panel = pd.concat({
        "px":   px,
        "close": close.reindex(columns=cols),
        "vol":   vol.reindex(columns=cols),
        "mcap":  mcap.reindex(columns=cols),
    }, axis=1)
    return panel


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': single frozen universe, no stage-2 generalization battery.
    raise NotImplementedError("scope='local': no generalization universes for this stress test")


# ------------------------------- SIGNAL CORE ---------------------------------
def _features(panel):
    px    = panel["px"]
    close = panel["close"]
    vol   = panel["vol"]
    mcap  = panel["mcap"]
    rets  = px.pct_change()
    dvol  = (close * vol).replace(0.0, np.nan)
    illiq = rets.abs() / dvol
    amihud = illiq.rolling(AMIHUD_WINDOW, min_periods=AMIHUD_MINP).mean()      # higher = illiquid
    med_dv = dvol.rolling(DV_WINDOW, min_periods=DV_MINP).median()            # borrowability proxy
    vol20  = rets.rolling(VOL_WINDOW, min_periods=VOL_MINP).std()
    return {"px": px, "close": close, "rets": rets,
            "amihud": amihud, "med_dv": med_dv, "vol20": vol20, "mcap": mcap}


def _rebal_dates(index):
    warm_idx = max(AMIHUD_WINDOW, DV_WINDOW) + 5
    if len(index) <= warm_idx:
        return []
    warm = index[warm_idx]
    return [d for d in index[::REBAL_DAYS] if d >= warm]


def _targets_at(date, f, borrow_floor):
    """Target dollar-neutral weights at one formation date (all inputs trailing-only).
    Returns (weight Series or None, per-tercile surviving-short counts [small,mid,large])."""
    a  = f["amihud"].loc[date]
    mc = f["mcap"].loc[date]
    pr = f["close"].loc[date]
    dv = f["med_dv"].loc[date]
    iv = (1.0 / f["vol20"].loc[date]).replace([np.inf, -np.inf], np.nan)

    base  = a.notna() & mc.notna() & iv.notna() & (iv > 0)
    names = base.index[base.values]
    if len(names) < MIN_NAMES:
        return None, []

    a_, mc_, pr_, iv_ = a[names], mc[names], pr[names], iv[names]
    q1, q2 = mc_.quantile(1.0 / 3.0), mc_.quantile(2.0 / 3.0)
    bounds = [(-np.inf, q1), (q1, q2), (q2, np.inf)]   # small, mid, large

    w = pd.Series(0.0, index=names)
    short_counts = []
    for lo, hi in bounds:
        mem = names[((mc_ > lo) & (mc_ <= hi)).values]
        if len(mem) < MIN_TERCILE:
            short_counts.append(0)
            continue
        am_mem = a_[mem]
        # LONG: most-illiquid quintile (highest Amihud), inverse-vol weighted
        n_long = max(LONG_MIN, int(np.ceil(LONG_Q * len(mem))))
        long_names = am_mem.sort_values(ascending=False).index[:n_long]
        # SHORT candidates: most-liquid (lowest Amihud) within price band; THEN borrow floor
        prm = pr_[mem]
        price_ok = mem[((prm >= PRICE_MIN) & (prm <= PRICE_MAX)).values]
        if len(price_ok):
            cand = a_[price_ok].sort_values(ascending=True).index[:SHORT_TOP_N]
            short_names = [n for n in cand
                           if pd.notna(dv.get(n, np.nan)) and dv[n] >= borrow_floor]
        else:
            short_names = []
        short_counts.append(len(short_names))

        lv = iv_[long_names]
        if lv.sum() > 0:
            w.loc[long_names] += (lv / lv.sum()) * (1.0 / N_SIZE_TERCILES)
        if len(short_names):
            sv = iv_[short_names]
            if sv.sum() > 0:
                w.loc[short_names] -= (sv / sv.sum()) * (1.0 / N_SIZE_TERCILES)

    # dollar-neutralise (long->+1, short->-1) and enforce 10% single-name cap (iterate)
    for _ in range(6):
        w = w.clip(-SINGLE_NAME_CAP, SINGLE_NAME_CAP)
        pos = w[w > 0].sum(); neg = -w[w < 0].sum()
        if pos > 0: w[w > 0] = w[w > 0] / pos
        if neg > 0: w[w < 0] = w[w < 0] / neg
    return w, short_counts


def signal(panel, **params):
    """borrow_floor: $-ADV borrowability floor on the SHORT leg.
       default (primary book) = $5M. borrow_floor=0 reproduces the unrestricted v3 baseline.
       Weights are formed at each date's close from TRAILING data, then lagged 1 day
       (W.shift(1)) before any return/turnover/borrow accounting -> no look-ahead."""
    borrow_floor = float(params.get("borrow_floor", BORROW_FLOOR))
    f    = _features(panel)
    px, rets = f["px"], f["rets"]
    cols, dates = px.columns, px.index

    rows = {}
    for d in _rebal_dates(dates):
        w, _ = _targets_at(d, f, borrow_floor)
        if w is not None:
            rows[d] = w.reindex(cols).fillna(0.0)

    if not rows:
        empty = pd.Series(dtype=float, name="amihud_borrowable_short")
        return empty, []

    Wt = pd.DataFrame(rows).T.reindex(columns=cols).fillna(0.0)
    W  = Wt.reindex(dates).ffill().fillna(0.0)
    Wlag = W.shift(1).fillna(0.0)                 # 1-day execution lag (no look-ahead)

    Wl, Ws = Wlag.clip(lower=0.0), Wlag.clip(upper=0.0)
    # gross is linear in W -> split sleeves to charge asymmetric (long vs short) turnover costs
    r_long  = net_of_cost(Wl, rets, cost_bps=COST_LONG_BPS,  name="amihud_long").fillna(0.0)
    r_short = net_of_cost(Ws, rets, cost_bps=COST_SHORT_BPS, name="amihud_short").fillna(0.0)
    daily   = r_long + r_short
    borrow  = (BORROW_BPS_YR / 1e4) / 252.0 * Ws.abs().sum(axis=1)   # short-notional financing
    daily   = (daily - borrow.reindex(daily.index).fillna(0.0))
    daily.name = "amihud_borrowable_short"

    trades = trades_from_weights(Wlag, rets, _STATE.get("sector_map", {}))  # kit stamps entry_regime
    return daily, trades


# --------------------------- SOFT EXPECTATIONS -------------------------------
def _sharpe(r):
    r = r.dropna()
    if len(r) < 60 or r.std() == 0:
        return 0.0
    return float(np.sqrt(252.0) * r.mean() / r.std())


def _short_diag(panel, holdout_start):
    """Pre-holdout short-basket diagnostics: avg surviving shorts per tercile-formation and
    per-tercile drop induced by the floor. Recomputed strictly on dates < holdout_start."""
    key = (id(panel), str(holdout_start))
    if key in _DIAG_CACHE:
        return _DIAG_CACHE[key]
    f = _features(panel)
    h = pd.Timestamp(holdout_start)
    pre = f["px"].index[f["px"].index < h]
    rebal = _rebal_dates(pre)[::4]   # sample to keep the soft check cheap
    floor_counts, drop_s, drop_l = [], [], []
    for d in rebal:
        _, scf = _targets_at(d, f, BORROW_FLOOR)
        _, scu = _targets_at(d, f, 0.0)
        if len(scf) == 3:
            floor_counts.extend(scf)
            if len(scu) == 3:
                drop_s.append(scu[0] - scf[0])   # small tercile drop
                drop_l.append(scu[2] - scf[2])   # large tercile drop
    res = {
        "avg_floor":  float(np.mean(floor_counts)) if floor_counts else 0.0,
        "drop_small": float(np.mean(drop_s)) if drop_s else 0.0,
        "drop_large": float(np.mean(drop_l)) if drop_l else 0.0,
    }
    _DIAG_CACHE[key] = res
    return res


def _check_premium_survives(ctx):
    try:
        h = pd.Timestamp(ctx["holdout_start"])
        base = ctx["search"]; base = base[base.index < h]
        unr, _ = signal(ctx["panel"], borrow_floor=0.0)   # ONE extra signal() call
        unr = unr[unr.index < h]
        s_b, s_u = _sharpe(base), _sharpe(unr)
        ratio = (s_b / s_u) if s_u > 0 else 0.0
        return {"pass": bool(ratio >= 0.70 and s_b > 0), "observed": round(ratio, 3)}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _check_short_basket_nondegenerate(ctx):
    try:
        d = _short_diag(ctx["panel"], ctx["holdout_start"])
        return {"pass": bool(d["avg_floor"] >= 6.0), "observed": round(d["avg_floor"], 2)}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _check_floor_hits_small(ctx):
    try:
        d = _short_diag(ctx["panel"], ctx["holdout_start"])
        ok = d["drop_small"] >= d["drop_large"]
        return {"pass": bool(ok),
                "observed": f"small_drop={d['drop_small']:.2f},large_drop={d['drop_large']:.2f}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


# --------------------------------- SPEC --------------------------------------
SPEC = StrategySpec(
    id="amihud_illiq_borrowable_short_v1",
    family="liquidity_premium",
    title="Amihud illiquidity premium — borrowable-short variant ($5M ADV floor on the short leg)",
    markets=["US small/mid-cap equities (Sharadar SEP/TICKERS, survivorship-clean, delisted incl)"],
    data_desc=("Owned Sharadar SEP daily closeadj/close/volume + SF1 marketcap (PIT via datekey, "
               "ffilled). Amihud illiquidity = mean(|ret|/$vol) over 63d. Borrowability proxy = "
               "20d trailing median dollar volume, computed point-in-time at each formation date."),
    pre_registration=(
        "FROZEN single spec, NO search. Reproduce deployed amihud_illiq_tranched_v3 "
        "(size-tercile Amihud sort; LONG=most-illiquid quintile per size tercile, inverse-vol; "
        "SHORT=top-15 most-liquid per size tercile, $10-$500 price band; 10% single-name cap; "
        "dollar-neutral; weekly rebalance; costs 60bps RT long / 15bps RT short + 50bps/yr borrow; "
        "residual IWM beta-trim sleeve, hedge_cap=0.35) with EXACTLY ONE CHANGE: each short "
        "candidate must clear a FROZEN borrowability floor = 20d median dollar volume >= $5,000,000 "
        "(point-in-time, zero look-ahead). Failing shorts are DROPPED (not back-filled deeper down the "
        "liquidity rank); the dollar-neutral target re-balances across surviving shorts. Long leg "
        "UNCHANGED. PRIMARY VERDICT = this borrowable-short book; default_params borrow_floor=$5M, and "
        "the same code with borrow_floor=0 reproduces the unrestricted v3 baseline apples-to-apples. "
        "SUCCESS (decision-critical): borrowable-short book retains >=70% of unrestricted-v3 search "
        "Sharpe (machine-checked) AND holdout>0 AND MCPT passes; mechanism: the floor must leave a "
        "non-degenerate short basket (>=~6 names/tercile-formation, machine-checked) and bite mainly "
        "in the SMALL size tercile (machine-checked). The $5M floor was chosen a priori (borrow desks "
        "broadly carry names above a few $M ADV) and is NOT a selection knob. "
        "NOT machine-checkable here (reported, not gated): the look-ahead-tainted gauge of how many of "
        "the book's historical shorts are in TODAY's Alpaca shortable set — the harness has no live "
        "broker access offline, so it stays prose-only and forward-looking. If the premium does NOT "
        "survive, that is itself the fundable finding: the Amihud short alpha concentrates in "
        "unborrowable small-caps and the long-only / borrowable economics must be reconsidered."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"borrow_floor": BORROW_FLOOR},
    grid={"default": {}},                  # frozen single spec -> honest effective-N = 1
    scope="local",                         # universe-specific deployment-constraint stress test
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT_START,
    deploy_max_positions=50,
    hedge_tickers=["IWM"],                 # declared residual beta-trim sleeve (whitelist + cap)
    hedge_cap=0.35,
    expectations=[
        {"name": "premium_survives",
         "claim": "borrowable-short book retains >= 70% of unrestricted-v3 search Sharpe",
         "check": _check_premium_survives},
        {"name": "short_basket_nondegenerate",
         "claim": "the $5M floor leaves >= 6 surviving short names per size-tercile-formation (avg, pre-holdout)",
         "check": _check_short_basket_nondegenerate},
        {"name": "floor_bites_small_tercile",
         "claim": "borrowability floor drops more shorts in the SMALL size tercile than the LARGE one",
         "check": _check_floor_hits_small},
    ],
)