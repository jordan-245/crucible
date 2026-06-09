import numpy as np
import pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import (
    sep_panel, us_universe, sf1, yf_panel, fred_series,
    trend_returns, carry_returns, inv_vol_position,
)

# ----------------------------------------------------------------------------
# Net share issuance factor (long low-issuance / buybacks, short high-issuance).
# Corporate-financing premium (Pontiff-Woodgate; McLean-Pontiff-Watanabe).
# STANDALONE long/short equity factor. OWNED data only: Sharadar SEP + SF1.
# ----------------------------------------------------------------------------

START = "2004-01-01"  # need 12m lookback before the 2005 sample start
SECTORS = [
    "Healthcare", "Technology", "Financial Services", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive", "Energy",
    "Basic Materials", "Real Estate", "Utilities",
]


def load_data() -> pd.DataFrame:
    # 1) survivorship-clean universe (delisted INCLUDED) + sector map ----------
    ticker_sector = {}
    for s in SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             include_delisted=True)
        except Exception:
            ts = []
        for t in ts:
            ticker_sector[t] = s
    if not ticker_sector:  # fallback: full domestic common-stock list
        for t in us_universe(category="Domestic Common Stock",
                             include_delisted=True):
            ticker_sector[t] = "Unknown"
    tickers = sorted(ticker_sector.keys())

    # 2) adjusted prices (returns) + raw volume (liquidity) -------------------
    px = sep_panel(tickers, START, field="closeadj")
    try:
        vol = sep_panel(tickers, START, field="volume")
    except Exception:
        vol = None

    have = [t for t in px.columns if t in ticker_sector]
    px = px[have]
    if vol is not None:
        vol = vol.reindex(columns=have)

    # 3) POINT-IN-TIME shares outstanding via `datekey` (NEVER calendardate) --
    sf = sf1(have, ["sharesbas"], dimension="ARQ")
    if "datekey" not in sf.columns or "ticker" not in sf.columns:
        sf = sf.reset_index()
    sf = sf.copy()
    sf["datekey"] = pd.to_datetime(sf["datekey"])
    sf = sf.dropna(subset=["sharesbas"])
    sf = sf[sf["sharesbas"] > 0]
    shares_wide = (sf.pivot_table(index="datekey", columns="ticker",
                                  values="sharesbas", aggfunc="last")
                     .sort_index())

    # forward-fill the most recently FILED sharesbas onto every trading day
    idx = px.index
    shares_daily = (shares_wide.reindex(idx.union(shares_wide.index))
                    .ffill().reindex(idx).reindex(columns=have))

    panel = px.copy()
    panel.attrs["volume"] = vol
    panel.attrs["shares"] = shares_daily
    panel.attrs["sectors"] = ticker_sector
    return panel


def signal(panel, **params):
    q = params.get("quantile", 0.20)          # quintiles by default
    min_dv = params.get("min_dollar_vol", 2e6)
    vol_lb = params.get("vol_lb", 63)
    cost_bps = params.get("cost_bps", 8.0)
    capital = params.get("capital", 1_000_000.0)

    px = panel
    vol = panel.attrs.get("volume")
    shares_daily = panel.attrs["shares"]
    ticker_sector = panel.attrs["sectors"]

    rets = px.pct_change()

    # --- monthly rebalance dates: last trading day of each calendar month ----
    me = px.index.to_series().resample("ME").last().dropna()
    me_dates = pd.DatetimeIndex(pd.Index(me.values))
    me_dates = me_dates[me_dates.isin(px.index)]

    # --- net share issuance = log(shares_t / shares_{t-12m}) on month-ends ---
    shares_me = shares_daily.reindex(me_dates)
    iss = np.log(shares_me / shares_me.shift(12)).replace([np.inf, -np.inf], np.nan)

    # --- liquidity screen: median 63d dollar-volume > threshold -------------
    if vol is not None:
        dv = (px * vol).rolling(63, min_periods=20).median().reindex(me_dates)
        liquid = dv > min_dv
    else:
        liquid = pd.DataFrame(True, index=me_dates, columns=px.columns)

    # --- inverse-vol weights (trailing daily-return vol) --------------------
    vol_me = rets.rolling(vol_lb, min_periods=20).std().reindex(me_dates)

    # --- build target weights at each rebalance -----------------------------
    tgt = pd.DataFrame(0.0, index=me_dates, columns=px.columns)
    for d in me_dates:
        row = iss.loc[d]
        liq = liquid.loc[d].reindex(row.index).fillna(False) \
            if d in liquid.index else pd.Series(True, index=row.index)
        valid = row.notna() & liq & px.loc[d].notna()
        x = row[valid]
        if x.shape[0] < 50:
            continue
        lo, hi = x.quantile(q), x.quantile(1 - q)
        longs = x[x <= lo].index          # lowest issuance / net buybacks
        shorts = x[x >= hi].index         # highest issuance
        if len(longs) == 0 or len(shorts) == 0:
            continue

        iv = (1.0 / vol_me.loc[d]).replace([np.inf, -np.inf], np.nan)
        wl = iv.reindex(longs);  wl = wl.fillna(wl.median())
        ws = iv.reindex(shorts); ws = ws.fillna(ws.median())
        if not (np.isfinite(wl.sum()) and wl.sum() > 0
                and np.isfinite(ws.sum()) and ws.sum() > 0):
            continue
        wl = 0.5 * wl / wl.sum()          # dollar-neutral: long +0.5
        ws = -0.5 * ws / ws.sum()         #                 short -0.5
        tgt.loc[d, longs] = wl.values
        tgt.loc[d, shorts] = ws.values

    # --- daily held weights, 1-day lag, costs on turnover -------------------
    W = tgt.reindex(px.index).ffill().fillna(0.0)
    Wlag = W.shift(1).fillna(0.0)                    # NO look-ahead

    gross = (Wlag * rets).sum(axis=1)
    turn = (Wlag - Wlag.shift(1)).abs().sum(axis=1)
    cost = turn * (cost_bps / 1e4)
    net = (gross - cost).fillna(0.0)
    net.name = "net_issuance_factor"

    active_any = W.ne(0).any(axis=1)
    if active_any.any():
        net = net.loc[active_any.idxmax():]

    # --- trades: one per contiguous held run per name -----------------------
    trades = []
    held_cols = Wlag.columns[(Wlag != 0).any()]
    pnl_daily = (Wlag[held_cols] * rets[held_cols] * capital)
    for t in held_cols:
        w = Wlag[t]
        sgn = np.sign(w).fillna(0.0)
        if not (sgn != 0).any():
            continue
        grp = (sgn != sgn.shift(1)).cumsum()
        for _, sub in sgn.groupby(grp):
            if sub.iloc[0] == 0:
                continue
            dts = sub.index
            pv = float(w.reindex(dts).abs().mean() * capital)
            tpnl = float(pnl_daily[t].reindex(dts).sum())
            trades.append({
                "ticker": t,
                "sector": ticker_sector.get(t, "Unknown"),
                "entry_date": dts[0].strftime("%Y-%m-%d"),
                "exit_date": dts[-1].strftime("%Y-%m-%d"),
                "hold_days": int(len(dts)),
                "position_value": pv,
                "pnl": tpnl,
            })

    return net, trades


SPEC = StrategySpec(
    id="net_share_issuance_v1",
    family="issuance",
    title="Net share issuance factor (long buybacks / short issuers, US equities)",
    markets=["US_EQUITY"],
    data_desc=(
        "Sharadar SEP survivorship-clean split/div-adjusted prices (delisted "
        "included) + SF1 `sharesbas` point-in-time via `datekey`. Domestic "
        "common stock, median 63d dollar-vol > $2M. OWNED data, $0."
    ),
    pre_registration=(
        "FROZEN. Premium: net-share-issuance / corporate-financing "
        "(Pontiff-Woodgate; McLean-Pontiff-Watanabe) — net issuers underperform, "
        "net repurchasers outperform; a distinct family NOT subsumed by the "
        "already-nulled value/quality/momentum/size set. Universe: Sharadar "
        "domestic common stock, delisted INCLUDED (survivorship-clean), liquidity "
        "filter median 63d dollar-volume > $2M. Signal: at each month-end, net "
        "issuance = log(sharesbas_t / sharesbas_{t-12m}) using point-in-time "
        "`datekey` (NEVER calendardate), most-recent-filed value forward-filled. "
        "Cross-sectional rank: LONG bottom quintile (lowest issuance / buybacks), "
        "SHORT top quintile (highest issuance). Dollar-neutral, inverse-vol "
        "weighted within each leg (gross=1, net=0), MONTHLY rebalance, signals "
        "lagged 1 trading day, ~8bps cost on turnover. STANDALONE — no trend / "
        "hedge leg (per the construction rule); add a small crisis overlay only "
        "if a real standalone edge survives. Holdout 2022-01-01+. Hypothesis: the "
        "long-short earns positive net out-of-sample Sharpe, orthogonal to prior "
        "nulled factors."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                       # primary: quintiles, $2M, 63d vol
        "q10": {"quantile": 0.10},           # tighter deciles
        "q30": {"quantile": 0.30},           # wider tertiles
        "liq5M": {"min_dollar_vol": 5e6},    # stricter liquidity
        "vol126": {"vol_lb": 126},           # slower vol estimate
    },
    holdout_start="2022-01-01",
    deploy_max_positions=20,
)