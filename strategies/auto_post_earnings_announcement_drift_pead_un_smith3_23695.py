"""
Strategy module: PEAD-SUE small-cap long/short (event-driven under-reaction premium)
====================================================================================
Family   : event_underreaction_pead
Axis     : earnings-surprise drift (NOT static XS price-momentum / value / quality —
           those were retired on a clean null; this is a different, pre-flagged axis).
Premium  : behavioral earnings under-reaction + limits-to-arbitrage, isolated to the
           less-efficient corner (small-cap, low-coverage, ADV>=$1M for cost-honesty).

----------------------------------------------------------------------------------
WHY THIS MODULE WAS REWRITTEN  (root-cause fix, not a pivot)
----------------------------------------------------------------------------------
The prior version shipped a GATE-0 *NO-GO* claiming the point-in-time, survivorship-
correct EPS surface SUE needs "is not reachable on the tested/free adapters" and so
`load_data()` *raised* — which simply breaks the harness contract (the traceback the
run produced). That premise was FALSE. It probed `sdk.adapters` for adapter names
`sf1_eps`/`sf1_fundamentals`/`pit_eps` that never existed, missed the ones that DO,
and refused to look on disk.

The sanctioned adapter module exposes EXACTLY the survivorship-correct, point-in-time
inputs this signal requires (all OWNED / zero marginal cost, data present on disk):
  * `sf1(tickers, fields=["eps"], dimension="ARQ")` -> as-reported quarterly EPS keyed
    on the FILING date `datekey` (the adapter's own docstring: "use `datekey` ... NOT
    calendardate ... to avoid look-ahead"). This is the point-in-time SUE input.
  * `sep_panel(tickers, field="closeadj"|"volume")` -> SURVIVORSHIP-CLEAN daily prices
    (delisted names included, split+div adjusted) — the docstring literally says
    "PREFER THIS over yf_panel ... yfinance has survivorship bias".
  * `us_universe(category="Domestic Common Stock", marketcap="Small",
    include_delisted=True)` -> the delisted-inclusive small-cap common-stock universe.

So both biases the prior author feared (restated EPS look-ahead; survivor truncation)
are SOLVED by the available adapters, not by refusing to run. The honest move is to
RUN the real, look-ahead-free, survivorship-clean PEAD backtest and let the rails
deliver the verdict — which is what this module now does. (Whatever the verdict, it is
the truthful one; we do not pre-judge it.)

DESIGN (frozen below; see SPEC.pre_registration)
  SUE_t = (EPS_q - EPS_{q-4}) / sigma(dEPS over trailing 8q), evaluated at each name's
  as-reported `datekey` (strictly point-in-time). On each announcement day, restrict to
  the less-efficient corner (closeadj >= $2, 21d dollar-ADV >= $1M), require >= 25
  announcers, rank into quintiles -> LONG top / SHORT bottom. Enter T+1 (1-day signal
  lag) and additionally earn returns only from the day AFTER weights are set (a second
  structural lag -> no entry-day look-ahead). Hold 60 trading days, overlapping daily
  tranches, inverse-vol weights, dollar-neutral +/-0.5 gross, target-vol scaled with a
  LAGGED realized-vol estimate. Costs = 15 bps one-way on turnover (small-cap honest;
  30 bps round trip) + daily short-borrow accrual. Write-once holdout from 2022-01-01.

The PEAD leg is tested STANDALONE first (per the proposal); the validated 21-market
trend crisis-hedge is NOT bundled into this verdict.
"""

from sdk.harness import StrategySpec
from sdk.adapters import (  # tested, owned, zero-marginal-cost adapters
    yf_panel, fred_series, trend_returns, carry_returns, inv_vol_position,
    sf1, sep_panel, us_universe,
)
import glob
import numpy as np
import pandas as pd

RET_NAME = "pead_sue_smallcap_ls"
_SHARADAR_DIR = "/root/atlas/data/sharadar"

# ---- Frozen primary spec --------------------------------------------------------
DEFAULTS = dict(
    # universe / SUE
    universe_start="2005-01-01",
    min_quarters=8,         # >=8 quarterly EPS for seasonal-RW std
    sue_std_lb=8,           # trailing quarters for sigma(dEPS)
    quintiles=5,            # top vs bottom bucket of the day's announcers
    min_names_day=25,       # >= ~5 names per bucket per announce-day
    # less-efficient corner (cost-honest)
    price_floor=2.0,        # avoid sub-$2 microcaps
    adv_floor=1_000_000.0,  # 21d dollar-ADV >= $1M so it is tradable
    adv_lb=21,
    # event window / sizing
    hold_days=60,           # canonical drift window (trading days)
    vol_lb=60,              # trailing window for inverse-vol & target-vol scaling
    target_vol=0.10,        # 10% annualized book target
    max_leverage=4.0,
    # costs (small-cap honest)
    oneway_bps=15.0,        # 30 bps round-trip on turnover
    round_trip_bps=30.0,
    borrow_annual=0.03,     # short borrow accrual on small-cap shorts
    notional=1.0e5,
    ret_name=RET_NAME,
)


def _sector_map():
    p = glob.glob(f"{_SHARADAR_DIR}/SHARADAR_TICKERS_*.csv")
    if not p:
        return {}
    tk = pd.read_csv(p[0], usecols=["ticker", "sector"]).dropna()
    return dict(tk.drop_duplicates("ticker").values)


def _empty():
    return pd.Series(dtype=float, name=RET_NAME), []


# ---- SUE (seasonal-random-walk standardized unexpected earnings), vectorized -----
def _build_events(P):
    """Point-in-time SUE events for the less-efficient corner.

    Returns (events_df, attrs) where events_df rows are one per qualifying announcement
    [ticker, sector, di(int announce index into `dates`), sue, px, adv] and attrs carries
    the heavy shared matrices the (param-dependent) signal reuses across the grid.
    """
    uni = us_universe(category="Domestic Common Stock", marketcap="Small",
                      include_delisted=True)
    if not uni:
        return None, None

    # 1) Point-in-time as-reported quarterly EPS -> SUE (seasonal random walk).
    f = sf1(uni, fields=["eps"], dimension="ARQ")
    f = f.dropna(subset=["eps", "datekey"]).sort_values(["ticker", "datekey"])
    n_q = f.groupby("ticker")["ticker"].transform("size")
    f = f[n_q >= P["min_quarters"]]
    if f.empty:
        return None, None
    yoy = f["eps"] - f.groupby("ticker")["eps"].shift(4)           # vs same q prior year
    sig = yoy.groupby(f["ticker"]).transform(
        lambda s: s.rolling(P["sue_std_lb"], min_periods=6).std())
    f = f.assign(sue=yoy / sig.replace(0.0, np.nan))
    ev = f.dropna(subset=["sue"])[["ticker", "datekey", "sue"]].copy()
    ev.columns = ["ticker", "adate", "sue"]
    ev["adate"] = pd.to_datetime(ev["adate"])
    cand = sorted(ev["ticker"].unique())
    if not cand:
        return None, None

    # 2) Survivorship-clean prices (delisted incl, split+div adj) -> returns, vol, ADV.
    close = sep_panel(cand, start=P["universe_start"], field="closeadj").astype("float32")
    close = close.dropna(axis=1, how="all")
    if close.shape[1] == 0:
        return None, None
    vol_raw = sep_panel(cand, start=P["universe_start"], field="volume").reindex(
        index=close.index, columns=close.columns).astype("float32")
    cols = close.columns
    dates = close.index
    R = close.pct_change().astype("float32")
    dvol = (close * vol_raw).rolling(P["adv_lb"], min_periods=P["adv_lb"] // 2).mean()
    # Per-name daily realized vol, LAGGED 1 day -> inverse-vol sizing uses past only.
    rv_lag = R.rolling(P["vol_lb"]).std().shift(1).astype("float32")

    # 3) Restrict announcements to the tradable, less-efficient corner (as-of announce).
    ev = ev[ev["ticker"].isin(cols)].copy()
    di = np.searchsorted(dates.values, ev["adate"].values, side="left")
    ev["di"] = di
    ev = ev[(ev["di"] >= 0) & (ev["di"] < len(dates))]
    ii = ev["di"].values.clip(0, len(dates) - 1)
    jj = cols.get_indexer(ev["ticker"].values)
    ev["px"] = close.values[ii, jj]
    ev["adv"] = dvol.values[ii, jj]
    ev = ev[(ev["px"] >= P["price_floor"]) & (ev["adv"] >= P["adv_floor"])]
    if ev.empty:
        return None, None
    ev["sector"] = ev["ticker"].map(_sector_map()).fillna("NA")

    attrs = {
        "dates": dates,
        "cols": list(cols),
        "colpos": {c: i for i, c in enumerate(cols)},
        "R": np.nan_to_num(R.values).astype("float32"),
        "rv": rv_lag.values.astype("float32"),
    }
    return ev[["ticker", "sector", "di", "sue", "px", "adv"]].reset_index(drop=True), attrs


def load_data() -> pd.DataFrame:
    """Assemble the point-in-time, survivorship-clean PEAD panel signal() consumes.

    Heavy, param-independent work (universe, SUE, price/vol/ADV matrices) happens ONCE
    here; the returned object is the qualifying-announcement event table, with the
    shared matrices stashed in `.attrs` for the lean, param-dependent signal()/grid.
    """
    P = dict(DEFAULTS)
    ev, attrs = _build_events(P)
    if ev is None or ev.empty:
        panel = pd.DataFrame(columns=["ticker", "sector", "di", "sue", "px", "adv"])
        panel.attrs["matrices"] = None
        return panel
    panel = ev
    panel.attrs["matrices"] = attrs
    return panel


def signal(panel, **params):
    """PEAD-SUE long/short, look-ahead-free. -> (daily net-of-cost returns, trades).

    Param-dependent step only (the heavy SUE/price work is cached in load_data):
      * each announce-day, rank announcers into `quintiles`; LONG top / SHORT bottom;
      * enter T+1, hold `hold_days`, overlapping daily tranches, inverse-vol weights;
      * dollar-neutral +/-0.5 gross; target-vol scaled with a LAGGED estimate;
      * returns earned on weights lagged 1 day (no entry-day look-ahead);
      * costs = one-way bps on turnover + daily short-borrow accrual.
    """
    P = dict(DEFAULTS)
    P.update(params or {})
    if panel is None or getattr(panel, "empty", True):
        return _empty()
    A = panel.attrs.get("matrices")
    if not A:
        return _empty()

    dates = A["dates"]
    cols = A["cols"]
    colpos = A["colpos"]
    R = A["R"]
    rv = A["rv"]
    T, N = R.shape
    QN = int(P["quintiles"])
    HOLD = int(P["hold_days"])

    # 1) Per-day quintile sort -> long top / short bottom, T+1 entry, build positions.
    #    Work on an attrs-free frame so per-day slices don't drag the heavy ndarray
    #    `.attrs` into pd.concat (which would try to compare arrays for equality).
    work = pd.DataFrame({
        "di": panel["di"].to_numpy(),
        "sue": panel["sue"].to_numpy(),
        "ticker": panel["ticker"].to_numpy(),
        "sector": panel["sector"].to_numpy(),
    })
    entry_i, exit_i, col_i, w_i, meta = [], [], [], [], []
    for d, grp in work.groupby("di"):
        if len(grp) < P["min_names_day"]:
            continue
        try:
            q = pd.qcut(grp["sue"].rank(method="first"), QN, labels=False)
        except ValueError:
            continue
        sel = pd.concat([grp[q == QN - 1].assign(side=1.0),
                         grp[q == 0].assign(side=-1.0)])
        ei = int(d) + 1                                   # T+1 entry (the 1-day lag)
        if ei >= T:
            continue
        xi = min(ei + HOLD, T - 1)
        for tk, sec, side in zip(sel["ticker"], sel["sector"], sel["side"]):
            j = colpos[tk]
            v = rv[ei, j]
            if not (np.isfinite(v) and v > 0):
                continue
            iv = 1.0 / v
            entry_i.append(ei); exit_i.append(xi); col_i.append(j)
            w_i.append(side * iv); meta.append((tk, sec, ei, xi, side))
    if not entry_i:
        return _empty()

    entry_i = np.asarray(entry_i)
    exit_i = np.asarray(exit_i)
    col_i = np.asarray(col_i)
    w_i = np.asarray(w_i, dtype="float64")

    # 2) Overlapping tranches -> held inverse-vol weights via a difference array.
    D = np.zeros((T, N), dtype="float64")
    np.add.at(D, (entry_i, col_i), w_i)
    inb = exit_i < T
    np.add.at(D, (exit_i[inb], col_i[inb]), -w_i[inb])
    Wraw = np.cumsum(D, axis=0)                            # active over [entry, exit)

    # 3) Dollar-neutral +/-0.5 gross per side.
    posM = np.clip(Wraw, 0.0, None)
    negM = np.clip(Wraw, None, 0.0)
    ls = posM.sum(1, keepdims=True)
    ss = -negM.sum(1, keepdims=True)
    Wn = (np.divide(posM, ls, out=np.zeros_like(posM), where=ls > 0) * 0.5
          + np.divide(negM, ss, out=np.zeros_like(negM), where=ss > 0) * 0.5)

    # 4) Target-vol scaling with a LAGGED estimate (no look-ahead).
    gross_raw = (Wn * R).sum(1)
    rvv = pd.Series(gross_raw, index=dates).rolling(P["vol_lb"]).std().shift(1)
    daily_tgt = P["target_vol"] / np.sqrt(252.0)
    scale = (daily_tgt / rvv).clip(upper=P["max_leverage"]).fillna(1.0).values[:, None]
    Wn = Wn * scale

    # 5) Net-of-cost daily returns. Earn on weights lagged 1 day -> no entry-day edge.
    Wn_lag = np.vstack([np.zeros((1, N)), Wn[:-1]])
    gross = (Wn_lag * R).sum(1)
    turn = np.abs(np.diff(Wn, axis=0, prepend=np.zeros((1, N)))).sum(1)
    cost = (P["oneway_bps"] / 1e4) * turn
    borrow = (P["borrow_annual"] / 252.0) * np.clip(-Wn_lag, 0.0, None).sum(1)
    net = pd.Series(gross - cost - borrow, index=dates).fillna(0.0)
    net = net.iloc[P["vol_lb"] + 1:]                      # drop inverse-vol warmup
    net.name = P["ret_name"]

    # 6) Trade ledger: one row per held position run (deployment-sanity input).
    trades = []
    notion = P["notional"]
    rt = P["round_trip_bps"] / 1e4
    bps_day = P["borrow_annual"] / 252.0
    for (tk, sec, ei, xi, side) in meta:
        j = colpos[tk]
        if xi <= ei:
            continue
        w = Wn_lag[ei:xi, j]
        r = R[ei:xi, j]
        hold = int(xi - ei)
        pos_val = float(notion * np.abs(w).mean()) if hold else 0.0
        gross_pnl = float(notion * (w * r).sum())
        tc = pos_val * rt
        bc = bps_day * pos_val * hold if side < 0 else 0.0
        trades.append({
            "ticker": tk,
            "sector": sec,
            "entry_date": dates[ei].strftime("%Y-%m-%d"),
            "exit_date": dates[xi].strftime("%Y-%m-%d"),
            "hold_days": hold,
            "position_value": round(pos_val, 2),
            "pnl": round(gross_pnl - tc - bc, 2),
        })

    return net, trades


SPEC = StrategySpec(
    id="pead_sue_smallcap_ls_v1",
    family="event_underreaction_pead",
    title="PEAD under-reaction premium — SUE-sorted small-cap/low-coverage long-short (owned SF1)",
    markets=["us_equities_smallcap"],
    data_desc=(
        "OWNED Sharadar, zero marginal cost. Point-in-time as-reported quarterly EPS "
        "via sf1(dimension='ARQ') keyed on the FILING date `datekey` (-> SUE, strictly "
        "point-in-time). Survivorship-CLEAN daily prices via sep_panel(closeadj/volume) "
        "(delisted names included, split+div adjusted) for returns, a 21d dollar-ADV "
        "liquidity filter, and per-name vol. Delisted-inclusive small-cap common-stock "
        "universe via us_universe(category='Domestic Common Stock', marketcap='Small', "
        "include_delisted=True). No yfinance fundamentals (restated/survivor-biased) are "
        "used anywhere."
    ),
    pre_registration=(
        "HYPOTHESIS: a behavioral earnings under-reaction / limits-to-arbitrage premium "
        "(PEAD) — a DIFFERENT AXIS from the retired static XS factors (no price ranking, "
        "no valuation ratio) — survives net of small-cap costs in the less-efficient "
        "corner. SIGNAL: SUE=(EPS_q-EPS_{q-4})/sigma(dEPS,8q) at each as-reported "
        "`datekey` (strictly point-in-time, >=8 quarters required); universe = "
        "delisted-inclusive small-cap US common stock, closeadj>=$2, 21d dollar-ADV>=$1M; "
        "each announce-day with >=25 qualifying announcers, rank into quintiles, LONG top "
        "/ SHORT bottom, ENTER T+1 (1-day signal lag) and earn returns only from the day "
        "AFTER weights are set (no entry-day look-ahead), HOLD 60 trading days, "
        "overlapping daily tranches, inverse-vol dollar-neutral +/-0.5 gross, target-vol "
        "scaled (10% ann.) with a LAGGED realized-vol estimate; costs = 15bps one-way on "
        "turnover (30bps round-trip, small-cap honest) + daily short borrow. VERDICT = "
        "rails write-once holdout (>=2022-01-01). GRID = the canonical drift-window and "
        "bucket-count robustness checks only (honest, small search burden); 'default' is "
        "primary. PAIRS with the validated 21-market trend crisis-hedge ONLY if this leg "
        "passes net-of-cost on the holdout (tested STANDALONE first; not bundled into "
        "this verdict). NOTE: this replaces a prior false GATE-0 'NO-GO' — the "
        "point-in-time, survivorship-clean inputs SUE needs ARE provided by the tested "
        "adapters (sf1 datekey EPS, sep_panel delisted-inclusive prices, us_universe "
        "delisted-inclusive), so we run the honest backtest and let the rails decide."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                  # PRIMARY: quintiles, 60-day drift window
        "hold_45": {"hold_days": 45},   # shorter drift window (robustness)
        "hold_90": {"hold_days": 90},   # longer drift window (robustness)
        "tercile": {"quintiles": 3},    # coarser surprise buckets (robustness)
        "adv_2m": {"adv_floor": 2_000_000.0},  # stricter liquidity corner (robustness)
    },
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)
