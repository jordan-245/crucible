# Net Share Issuance Factor — long low-issuance / short high-issuance, US equities (STANDALONE).
# Premium: corporate-financing / issuance premium (Pontiff-Woodgate; McLean-Pontiff-Watanabe).
# Net share ISSUERS underperform; net REPURCHASERS outperform. Distinct from value/quality/momentum/size.
# No reflexive trend hedge (issuance is a slow-moving balance-sheet premium, not a tail trade).
# OWNED data only: Sharadar SF1 `sharesbas` (point-in-time via datekey) + SEP survivorship-clean
# adjusted prices. No look-ahead (filing-date as-of + 1-day position lag), ~8bps cost, inverse-vol size,
# weekly rebalance, dollar-neutral long/short.

from sdk.harness import StrategySpec
from sdk.adapters import (
    sep_panel, us_universe, sf1, yf_panel, fred_series,
    trend_returns, carry_returns, inv_vol_position,
)
import numpy as np, pandas as pd

START = "2005-01-01"
TOP_N = 1000          # 1000 most-liquid survivorship-clean common names (delisted incl). NEVER full universe.
BOOK = 1_000_000.0    # nominal book size for trade position_value / pnl bookkeeping only.

# Sharadar sector labels (used to tag trades for deployment-sanity sector-spread check).
_SECTORS = [
    "Healthcare", "Financial Services", "Technology", "Consumer Cyclical",
    "Industrials", "Consumer Defensive", "Energy", "Basic Materials",
    "Real Estate", "Utilities", "Communication Services",
]

# In-memory side-channel for panel metadata (no external side effects; survives attrs loss).
_CACHE = {}


def _col(df, name):
    """Case-insensitive column finder."""
    for c in df.columns:
        if str(c).lower() == name:
            return c
    return None


def _sector_map(univ):
    """Map each universe ticker -> Sharadar sector by intersecting per-sector ticker lists."""
    uset = set(univ)
    smap = {}
    for s in _SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock", include_delisted=True)
        except Exception:
            ts = []
        for t in ts:
            if t in uset and t not in smap:
                smap[t] = s
    for t in univ:
        smap.setdefault(t, "Unknown")
    return smap


def load_data() -> pd.DataFrame:
    # 1) Survivorship-clean liquid universe (delisted INCLUDED -> no survivorship bias).
    univ = us_universe(category="Domestic Common Stock", top_n=TOP_N, include_delisted=True)
    univ = list(dict.fromkeys(univ))  # de-dup, keep order

    # 2) Survivorship-clean split+div-adjusted daily prices (total-return proxy).
    px = sep_panel(univ, start=START, field="closeadj").sort_index()
    px = px.loc[:, [c for c in px.columns if px[c].notna().sum() > 60]]  # drop near-empty cols

    # 3) Point-in-time shares outstanding via datekey (NEVER calendardate -> no look-ahead).
    raw = sf1(univ, fields=["sharesbas"], dimension="ARQ")
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    raw = raw.reset_index()
    tcol, dcol, vcol = _col(raw, "ticker"), _col(raw, "datekey"), _col(raw, "sharesbas")
    sh = raw[[tcol, dcol, vcol]].dropna().copy()
    sh.columns = ["ticker", "datekey", "sharesbas"]
    sh["datekey"] = pd.to_datetime(sh["datekey"])
    sh = sh[sh["sharesbas"] > 0].sort_values("datekey")

    # Wide point-in-time matrix indexed by FILING date.
    shw = sh.pivot_table(index="datekey", columns="ticker", values="sharesbas", aggfunc="last")

    # Month-end grid = last actual trading day of each month.
    s = px.index.to_series()
    grp_max = s.groupby([s.dt.year, s.dt.month]).transform("max")
    me_index = px.index[(s == grp_max).values]

    # As-of forward-fill shares to each month-end (latest filing with datekey <= month-end).
    all_dates = shw.index.union(me_index)
    sh_me = shw.reindex(all_dates).ffill().reindex(me_index)

    # 12-month log net share issuance (split-consistent: sharesbas is point-in-time count).
    # HIGH nsi = net issuance (expected underperformer); LOW/NEG = net repurchase (outperformer).
    nsi = np.log(sh_me / sh_me.shift(12))
    nsi = nsi.replace([np.inf, -np.inf], np.nan)

    # Forward-fill monthly signal to the daily price grid (point-in-time; no look-ahead).
    iss_daily = nsi.reindex(px.index).ffill()

    # Align both blocks to the common tradable column set.
    common = px.columns.intersection(iss_daily.columns)
    px = px.reindex(columns=common)
    iss_daily = iss_daily.reindex(columns=common)

    # Stash sector tags (used in signal() for trade tagging; same-process side channel).
    _CACHE["sectors"] = _sector_map(list(common))

    # Single panel: MultiIndex columns (field, ticker) so signal() gets prices + signal robustly.
    panel = pd.concat({"px": px, "iss": iss_daily}, axis=1)
    return panel


def signal(panel, **params):
    n_leg = int(params.get("n_leg", 50))          # names per side
    vol_lb = int(params.get("vol_lb", 63))        # inverse-vol / risk lookback (days)
    target_vol = float(params.get("target_vol", 0.10))  # ann. book vol target
    cost_bps = float(params.get("cost_bps", 8.0))
    max_lev = float(params.get("max_lev", 3.0))

    px = panel["px"]
    iss = panel["iss"]
    sectors = _CACHE.get("sectors", {})

    rets = px.pct_change()
    ann = np.sqrt(252.0)
    vol = rets.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std()

    # Weekly rebalance = last trading day of each ISO week.
    iso = px.index.isocalendar()
    key = pd.Series(iso["year"].astype(int).values * 100 + iso["week"].astype(int).values,
                    index=px.index)
    rebal_dates = px.index[(~key.duplicated(keep="last")).values]

    cols = px.columns
    rows = {}
    costs = {}
    prev_w = pd.Series(0.0, index=cols)

    for d in rebal_dates:
        s = iss.loc[d].dropna()
        if s.empty:
            continue
        v_d = vol.loc[d].reindex(s.index)
        s = s[v_d.notna() & (v_d > 0)]
        if len(s) < 2 * n_leg:
            continue

        ranked = s.sort_values()
        longs = ranked.index[:n_leg]            # lowest NSI -> repurchasers (overweight)
        shorts = ranked.index[-n_leg:]          # highest NSI -> issuers (underweight)

        new_w = pd.Series(0.0, index=cols)
        wL = 1.0 / vol.loc[d, longs]
        wL = wL / wL.sum()
        wS = 1.0 / vol.loc[d, shorts]
        wS = wS / wS.sum()
        new_w[longs] = 0.5 * wL.values          # long leg gross 0.5
        new_w[shorts] = -0.5 * wS.values         # short leg gross 0.5 -> total gross 1, net 0

        costs[d] = float((new_w - prev_w).abs().sum()) * cost_bps / 1e4
        rows[d] = new_w
        prev_w = new_w

    if not rows:
        out = pd.Series(dtype=float, name="net_share_issuance")
        return out, []

    # Daily held weights (held constant between rebalances), then 1-day execution lag in returns.
    Wre = pd.DataFrame(rows).T.reindex(px.index).ffill().fillna(0.0).reindex(columns=cols)
    cost_series = pd.Series(costs).reindex(px.index).fillna(0.0)

    gross = (Wre.shift(1) * rets).sum(axis=1)

    # Ex-ante vol target via TRAILING realized vol (shifted -> no look-ahead).
    roll = gross.rolling(63, min_periods=20).std().shift(1)
    lev = (target_vol / (roll * ann)).clip(upper=max_lev).fillna(0.0)

    net = (gross - cost_series) * lev
    net.name = "net_share_issuance"

    # Trim leading flat (pre-warmup) stretch.
    nz = net.ne(0)
    if nz.any():
        net = net.loc[nz.idxmax():]

    # ---- Trades: one record per held position RUN (continuous same-side holding) ----
    Cd = Wre.shift(1) * rets            # daily pnl fraction per ticker
    W_arr = Wre.values
    C_arr = Cd.values
    dates = Wre.index
    col_list = list(cols)

    trades = []
    for ci, t in enumerate(col_list):
        w = W_arr[:, ci]
        if not np.any(w != 0.0):
            continue
        sg = np.sign(w)
        n = len(w)
        i = 0
        while i < n:
            if sg[i] == 0.0:
                i += 1
                continue
            j = i
            cur = sg[i]
            while j + 1 < n and sg[j + 1] == cur:
                j += 1
            entry, exit_ = dates[i], dates[j]
            hold = int(max((exit_ - entry).days, 1))
            pos_val = float(np.nanmean(np.abs(w[i:j + 1])) * BOOK)
            pnl = float(np.nansum(C_arr[i:j + 1, ci]) * BOOK)
            trades.append({
                "ticker": t,
                "sector": sectors.get(t, "Unknown"),
                "entry_date": entry.strftime("%Y-%m-%d"),
                "exit_date": exit_.strftime("%Y-%m-%d"),
                "hold_days": hold,
                "position_value": pos_val,
                "pnl": pnl,
            })
            i = j + 1

    return net, trades


SPEC = StrategySpec(
    id="net_share_issuance_factor",
    family="equity_factor",
    title="Net Share Issuance Factor (long low-issuance / short high-issuance US equities)",
    markets=["US_EQUITY"],
    data_desc=(
        "Sharadar SF1 `sharesbas` (point-in-time via datekey) + SEP survivorship-clean split/div-"
        "adjusted closes; top-1000 most-liquid US Domestic Common Stocks (delisted incl), 2005-present. "
        "Signal = 12-month log change in shares outstanding; dollar-neutral L/S, weekly, inverse-vol."
    ),
    pre_registration=(
        "HYPOTHESIS: The net share issuance premium (Pontiff-Woodgate 2008; McLean-Pontiff-Watanabe) "
        "is a distinct, slow-moving corporate-financing anomaly: firms that net-ISSUE shares "
        "subsequently underperform, firms that net-REPURCHASE outperform, after controlling for size/"
        "value/momentum. TEST: cross-sectional dollar-neutral long-low / short-high 12m share-growth, "
        "50 names/side, weekly rebalance, inverse-vol sized within legs, ~8bps cost on turnover, 1-day "
        "execution lag, point-in-time shares via filing datekey (no look-ahead, survivorship-clean). "
        "STANDALONE — no reflexive trend hedge (this is a balance-sheet premium, not a tail trade). "
        "PASS if net-of-cost holdout (2022+) Sharpe is positive with DSR surviving the declared grid; "
        "FAIL otherwise. PRE-REGISTERED EXPECTATION: weak-to-modest standalone edge; null is a real "
        "outcome and will be banked as negative knowledge."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "n_leg_30": {"n_leg": 30},
        "n_leg_100": {"n_leg": 100},
        "vol_126": {"vol_lb": 126},
        "tv_15": {"target_vol": 0.15},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=100,
)