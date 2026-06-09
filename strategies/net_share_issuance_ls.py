"""
Net share issuance factor — long low-issuance (buybacks) / short high-issuance, US equities.

PREMIUM (distinct family, NOT value/quality/momentum/size — all already nulled here):
  the corporate-financing / capital-allocation premium. Net share ISSUERS systematically
  underperform; net REPURCHASERS outperform (Pontiff-Woodgate 2008; McLean-Pontiff-Watanabe
  2016 show it survives in 68 markets 1976-2022 and is NOT subsumed by the standard factors).
  Economic story: managers issue equity when it is over-valued / buy back when cheap, and
  issuance dilutes per-share claims — a slow, ~annual, well-powered cross-sectional signal.

DATA — OWNED, $0 (no paid/free-web downloads):
  * Sharadar SF1 `sharesbas` (basic shares outstanding) x `sharefactor` -> a clean,
    SPLIT-ADJUSTED share count, read POINT-IN-TIME via `datekey` (the FILING/available date,
    NEVER `calendardate`) so the signal uses only what was public on each rebalance day.
    (Verified: SF1 `sharesbas` is already split-adjusted — AAPL is smooth through its 2014/2020
    splits — and `sharefactor` is 1.0 for 99.8% of rows; multiplying is ratio-invariant in the
    common case and corrects the rare ADR/recap names where it is a constant != 1.)
  * Sharadar SEP `closeadj` (split+dividend adjusted) + `volume`, SURVIVORSHIP-CLEAN with
    delisted names INCLUDED (us_universe(include_delisted=True)) — kills the yfinance
    survivorship bias the wiki flags as an anti-pattern.

FROZEN PRE-REGISTERED DESIGN (written before running):
  Universe = us_universe(category='Domestic Common Stock', include_delisted=True). Keep names
  with median 63-day dollar volume (closeadj*volume) >= $2M AND price >= $5 AS-OF each rebalance
  (time-varying eligibility — no peeking). At each MONTH-END compute net share issuance
  iss = log( adjShares_t / adjShares_{t-12m} ) from the most-recent point-in-time filing
  (datekey-based forward fill; names with no filing 12m earlier are dropped -> IPO guard).
  Cross-sectionally: LONG the bottom quintile (lowest issuance / net buybacks), SHORT the top
  quintile (highest issuance / dilution). DOLLAR-NEUTRAL (long leg +1, short leg -1), INVERSE-VOL
  weighted within each leg (60d realized daily vol, lagged -> no look-ahead). MONTHLY rebalance
  (a 12-month signal does not move week-to-week; weekly churn would only burn cost), positions
  LAGGED 1 trading day, ~8bps cost on |Δweight| turnover. PRIMARY = net-of-cost Sharpe + MAR via
  the harness DSR/PBO/CPCV + write-once 2022+ holdout; 'default' is THE primary. The grid declares
  the honest effective-N search burden only (quintile/decile/tercile cut, equal- vs inverse-vol,
  $5M liquidity) — it is NOT optimised over. STANDALONE: the premium is tested ALONE (no trend /
  crisis-alpha overlay) per the construction rule — a ~0-Sharpe hedge blended 50/50 would just
  halve a real edge. A small tail overlay would only be considered IF a real standalone edge appears.
  KNOWN LIMIT (pre-registered): delisting returns are taken at last adjusted close (Sharadar has no
  explicit delist-to-zero return); the dominant survivorship correction — delisted names being IN
  the universe — is applied.
"""

from sdk.harness import StrategySpec
from sdk.adapters import us_universe, sep_panel, sf1
import os, glob
import numpy as np, pandas as pd

# ---------------------------- frozen primary parameters ------------------------------
_PRIMARY = dict(
    quintile=0.20,        # LONG bottom 20% issuance, SHORT top 20%
    weighting="invvol",   # inverse-vol within each leg (lagged 60d realized vol)
    vol_lb=60,            # daily-return vol lookback for inverse-vol sizing
    iss_lb_months=12,     # net-issuance measured over the trailing 12 months
    liq_min=2.0e6,        # min median 63d dollar volume (as-of, time-varying)
    price_min=5.0,        # min price (as-of)
    min_names=100,        # need a well-populated cross-section to form quintiles
    min_leg=5,            # min names per leg to trade that month
    rebalance_freq="BME", # MONTHLY (business month-end) — appropriate to an annual signal
    cost_bps=8.0,         # ~8bps on |Δweight| turnover
    start="2005-01-01",   # evaluation start (build buffer from 2003 for vol/issuance)
    build_start="2003-01-01",
    capital=1_000_000.0,  # notional for the deployment-sanity ledger only
)

_SHARADAR_TICKERS = "/root/atlas/data/sharadar/SHARADAR_TICKERS_*.csv"


def _sector_map() -> dict:
    p = glob.glob(_SHARADAR_TICKERS)[0]
    tk = pd.read_csv(p, usecols=["ticker", "sector"]).dropna(subset=["ticker"])
    return dict(zip(tk["ticker"], tk["sector"].fillna("Unknown")))


# =====================================================================================
def load_data() -> pd.DataFrame:
    """Load OWNED survivorship-clean SEP prices + point-in-time SF1 shares, then PRE-COMPUTE
    (once) the heavy, param-independent pieces signal() consumes: daily returns, the month-end
    net-issuance matrix, time-varying eligibility inputs, and inverse-vol inputs. The expensive
    work lives here (load_data runs once); signal() is then cheap across the grid."""
    p = _PRIMARY
    U = us_universe(category="Domestic Common Stock", include_delisted=True)

    px = sep_panel(U, start=p["build_start"], field="closeadj")
    vol = sep_panel(U, start=p["build_start"], field="volume")

    # restrict to names that were EVER liquid enough to matter (memory/speed) — eligibility
    # is still applied time-varying per rebalance below, so this only drops perennial illiquids.
    dvol = (px * vol)
    med63 = dvol.rolling(63, min_periods=21).median()
    ever = med63.max(axis=0) >= p["liq_min"]
    keep = med63.columns[ever.fillna(False).values]
    px, vol, med63 = px[keep], vol[keep], med63[keep]
    rets = px.pct_change()

    # --- point-in-time split-adjusted shares (datekey = filing date -> no look-ahead) ---
    shp = sf1(list(keep), fields=["sharesbas", "sharefactor"], dimension="ARQ")
    shp = shp.dropna(subset=["sharesbas", "datekey"]).copy()
    shp["adj"] = shp["sharesbas"] * shp["sharefactor"].fillna(1.0)
    shp = shp[shp["adj"] > 0]
    shares_wide = (shp.pivot_table(index="datekey", columns="ticker", values="adj", aggfunc="last")
                      .sort_index())
    # as-of each business day: most recent filing forward-filled (point-in-time)
    sd = shares_wide.reindex(shares_wide.index.union(px.index)).ffill().reindex(px.index)

    # month-end rebalance dates that actually exist in the price grid
    form = pd.date_range(px.index.min(), px.index.max(), freq=p["rebalance_freq"])
    form = pd.DatetimeIndex([d for d in form if d in px.index])

    # net issuance = log( shares_now / shares_12m_ago ), both point-in-time
    sh_now = sd.reindex(form)
    lag = form - pd.DateOffset(months=p["iss_lb_months"])
    sh_prev = sd.reindex(sd.index.union(lag)).ffill().reindex(lag)
    sh_prev.index = form
    with np.errstate(divide="ignore", invalid="ignore"):
        iss = np.log(sh_now / sh_prev)
    iss = iss.replace([np.inf, -np.inf], np.nan)

    # inputs for time-varying eligibility + inverse-vol sizing, sampled at rebalance dates
    nvol = rets.rolling(p["vol_lb"], min_periods=p["vol_lb"] // 2).std()
    price_f = px.reindex(form)
    med_f = med63.reindex(form)
    nvol_f = nvol.reindex(form)

    panel = rets                       # the panel signal() consumes is the daily-return matrix
    panel.attrs.update(dict(
        iss=iss, price_f=price_f, med_f=med_f, nvol_f=nvol_f,
        form=form, sectors=_sector_map(), capital=p["capital"],
    ))
    return panel


# ----------------------------------- helpers -----------------------------------------
def _build_weights(panel: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Month-end dollar-neutral quintile weights (long low issuance, short high), inverse-vol
    or equal weighted within each leg, using only as-of (point-in-time) information."""
    A = panel.attrs
    iss, price_f, med_f, nvol_f, form = (A["iss"], A["price_f"], A["med_f"], A["nvol_f"], A["form"])
    elig = (price_f >= p["price_min"]) & (med_f >= p["liq_min"]) & iss.notna() & (nvol_f > 0)
    issE = iss.where(elig)

    W = pd.DataFrame(0.0, index=form, columns=panel.columns)
    q = p["quintile"]
    for d in form:
        s = issE.loc[d].dropna()
        if len(s) < p["min_names"]:
            continue
        ql, qh = s.quantile(q), s.quantile(1.0 - q)
        longs = s.index[s <= ql]
        shorts = s.index[s >= qh]
        if len(longs) < p["min_leg"] or len(shorts) < p["min_leg"]:
            continue
        if p["weighting"] == "invvol":
            il = 1.0 / nvol_f.loc[d, longs]; wl = il / il.sum()
            isr = 1.0 / nvol_f.loc[d, shorts]; ws = isr / isr.sum()
        else:  # equal weight
            wl = pd.Series(1.0 / len(longs), index=longs)
            ws = pd.Series(1.0 / len(shorts), index=shorts)
        W.loc[d, longs] = wl.values
        W.loc[d, shorts] = -ws.values
    return W


def _trade_ledger(W: pd.DataFrame, Wd: pd.DataFrame, rets: pd.DataFrame,
                  sectors: dict, cap: float) -> list:
    """One trade per held position RUN (consecutive months a name stays in a leg). pnl = realised
    over the run's daily span; position_value = avg |weight| x capital."""
    pnl_daily = Wd * rets * cap
    held_cols = W.columns[(W.abs() > 0).any(axis=0)]
    trades = []
    for tk in held_cols:
        w = Wd[tk]
        held = w.abs() > 1e-12
        if not held.any():
            continue
        grp = (held != held.shift()).cumsum()[held]
        for _, idx in w[held].groupby(grp):
            dts = idx.index
            trades.append(dict(
                ticker=str(tk),
                sector=sectors.get(tk, "Unknown"),
                entry_date=dts[0].strftime("%Y-%m-%d"),
                exit_date=dts[-1].strftime("%Y-%m-%d"),
                hold_days=int(len(dts)),
                position_value=float(w.loc[dts].abs().mean() * cap),
                pnl=float(pnl_daily.loc[dts, tk].sum()),
            ))
    return trades


# =====================================================================================
def signal(panel: pd.DataFrame, **params):
    """Net-share-issuance long/short factor. Returns (daily_returns net-of-cost, trades)."""
    p = {**_PRIMARY, **params}
    rets = panel

    W = _build_weights(panel, p)
    Wd = W.reindex(rets.index, method="ffill").shift(1).fillna(0.0)   # hold + LAG 1 day
    gross = (Wd * rets).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).reindex(rets.index).fillna(0.0)  # turnover at rebalances
    net = (gross - turn * (p["cost_bps"] / 1e4)).rename("net_share_issuance_ls")
    net = net[net.index >= p["start"]].dropna()

    if not params.get("_emit_trades", True):       # grid variants skip the ledger (returns only)
        return net, []
    trades = _trade_ledger(W, Wd, rets, panel.attrs["sectors"], panel.attrs["capital"])
    return net, trades


# =====================================================================================
SPEC = StrategySpec(
    id="net-share-issuance-ls",
    family="corporate_financing_issuance",
    title="Net share issuance — long low-issuance / short high-issuance US equity L/S (standalone)",
    markets=["US_EQUITIES"],
    data_desc=(
        "OWNED, $0. Sharadar SF1 sharesbas x sharefactor read point-in-time via datekey "
        "(split-adjusted shares, no look-ahead) + survivorship-clean SEP closeadj/volume with "
        "delisted names included. Liquid common stock (median 63d $vol >= $2M, price >= $5)."
    ),
    pre_registration=(
        "FROZEN. Premium = corporate-financing / net-share-issuance (Pontiff-Woodgate; "
        "McLean-Pontiff-Watanabe) — a distinct family from the already-nulled value/quality/"
        "momentum/size set. Universe = us_universe('Domestic Common Stock', incl. delisted), "
        "time-varying liquidity (median 63d $vol >= $2M) & price (>= $5) eligibility AS-OF each "
        "rebalance. Signal = log(adjShares_t / adjShares_{t-12m}) from point-in-time SF1 (datekey, "
        "NEVER calendardate; split-adjusted via sharefactor; <12m-history names dropped). LONG "
        "bottom quintile (buybacks), SHORT top quintile (dilution); dollar-neutral, inverse-vol "
        "within leg (60d lagged vol), MONTHLY rebalance, 1-day lag, ~8bps on |Δweight| turnover. "
        "STANDALONE (no trend/crisis overlay — a ~0-Sharpe hedge at 50/50 would halve a real edge). "
        "PRIMARY = net Sharpe + MAR via DSR/PBO/CPCV + write-once 2022+ holdout; 'default' is THE "
        "primary, grid declares effective-N only (quintile/decile/tercile, equal- vs inverse-vol, "
        "$5M liq). KNOWN LIMIT: delist returns at last adjusted close; delisted names ARE in-universe."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},                 # primary == frozen spec
    grid={                             # honest effective-N search burden ONLY (declared, not tuned)
        "default": {},
        "decile": {"quintile": 0.10, "_emit_trades": False},
        "tercile": {"quintile": 0.3333, "_emit_trades": False},
        "equal_weight": {"weighting": "equal", "_emit_trades": False},
        "liq_5m": {"liq_min": 5.0e6, "_emit_trades": False},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=400,          # broad diversified quintile book (~hundreds of names/side)
)
