# Amihud (2002) illiquidity premium — cross-sectional small-cap factor book.
# Mechanism: stocks with higher price-impact-per-dollar-traded (|ret|/$vol) earn a
# return premium for bearing illiquidity. Tested in liquidity-bounded SMALL caps across
# 11 sectors (survivorship-clean, delisted included). Long high-Amihud / short low-Amihud,
# inverse-vol sized, weekly rebal, signals lagged 1 day, ~8bps/turnover costs.

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
import numpy as np
import pandas as pd

START = "2001-01-01"

_SECTORS = [
    "Healthcare", "Basic Materials", "Financial Services", "Consumer Cyclical",
    "Technology", "Consumer Defensive", "Industrials", "Real Estate",
    "Energy", "Utilities", "Communication Services",
]


def load_data() -> pd.DataFrame:
    # Build a liquidity-bounded SMALL-cap universe spread across sectors (and a
    # ticker->sector map for trade tagging). top_n per sector caps total ~1.4k names
    # (well inside the rails' safe range; never the full ~16k universe).
    sector_map = {}
    tickers = []
    for s in _SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             marketcap="Small", include_delisted=True, top_n=130)
        except Exception:
            ts = []
        for t in ts:
            sector_map[t] = s
        tickers.extend(ts)
    tickers = sorted(set(tickers))

    # Returns from split+div-adjusted close (closeadj is the curated SEP field).
    px = sep_panel(tickers, START, field="closeadj")

    # Dollar volume for the Amihud denominator. Prefer raw close * volume (the genuine
    # $ traded), but the owned SEP panel exposes closeadj + volume and may NOT carry a
    # raw 'close' field (that was the read failure) -> fall back to closeadj as the
    # price level. For a purely CROSS-SECTIONAL ranking this proxy is fine.
    try:
        px_lvl = sep_panel(tickers, START, field="close").reindex(
            index=px.index, columns=px.columns)
    except Exception:
        px_lvl = px
    volume = sep_panel(tickers, START, field="volume").reindex(
        index=px.index, columns=px.columns)
    dvol = px_lvl * volume

    panel = pd.concat({"px": px, "dvol": dvol}, axis=1)
    panel = panel.sort_index()
    panel.attrs["sector_map"] = sector_map
    return panel


def signal(panel, lookback=63, n_long=75, n_short=75, vol_lb=63,
           min_dvol=2.5e5, cost_bps=8.0, gross=1.0, **params):
    px = panel["px"]
    dvol = panel["dvol"].reindex_like(px)
    idx = px.index
    rets = px.pct_change()

    # --- Amihud illiquidity: rolling mean of |daily ret| / daily dollar volume ---
    mp = max(20, lookback // 2)
    daily_illiq = rets.abs() / dvol.where(dvol > 0)
    amihud = daily_illiq.rolling(lookback, min_periods=mp).mean()
    adv = dvol.rolling(lookback, min_periods=mp).mean()          # tradeability floor
    vol = rets.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std()  # inv-vol sizing

    # --- weekly rebalance days = last trading day of each ISO week ---
    order = np.arange(len(idx))
    last_of_wk = pd.Series(order, index=idx).groupby(idx.to_period("W")).transform("max")
    rebal_dates = idx[order == last_of_wk.values]

    weights = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    half = gross / 2.0
    for d in rebal_dates:
        a, l, v = amihud.loc[d], adv.loc[d], vol.loc[d]
        elig = a.notna() & v.notna() & (v > 0) & (l > min_dvol)
        cand = a[elig]
        if cand.size < (n_long + n_short):
            continue
        longs = cand.nlargest(n_long).index    # most illiquid -> long the premium
        shorts = cand.nsmallest(n_short).index  # most liquid   -> short
        wl = 1.0 / v[longs]
        ws = 1.0 / v[shorts]
        w = pd.Series(0.0, index=px.columns)
        w[longs] = (wl / wl.sum() * half).values
        w[shorts] = (-ws / ws.sum() * half).values
        weights.loc[d] = w

    weights = weights.ffill().fillna(0.0)
    pos = weights.shift(1).fillna(0.0)  # lag 1 day: no look-ahead

    gross_ret = (pos * rets).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    net_ret = (gross_ret - turnover * (cost_bps / 1e4)).rename("amihud_illiquidity")

    active = (pos != 0).any(axis=1)
    if not active.any():
        return net_ret.iloc[0:0], []
    first = active.idxmax()
    net_ret = net_ret.loc[first:]

    # --- trades: one record per contiguous held-position run ---
    posa = pos.loc[first:]
    contriba = (posa * rets.loc[first:])
    dates = posa.index.strftime("%Y-%m-%d").to_numpy()
    pv_all = posa.to_numpy()
    cv_all = contriba.to_numpy()
    cols = posa.columns.to_numpy()
    sgn = np.sign(pv_all)
    held_any = (pv_all != 0).any(axis=0)
    sector_map = panel.attrs.get("sector_map", {})
    book = 1_000_000.0
    nrows = pv_all.shape[0]

    trades = []
    for ci in np.where(held_any)[0]:
        s = sgn[:, ci]
        pv = pv_all[:, ci]
        cv = cv_all[:, ci]
        tk = str(cols[ci])
        sec = sector_map.get(tk, "Unknown")
        i = 0
        while i < nrows:
            if s[i] == 0:
                i += 1
                continue
            cur = s[i]
            j = i
            while j < nrows and s[j] == cur:
                j += 1
            trades.append({
                "ticker": tk,
                "sector": sec,
                "entry_date": str(dates[i]),
                "exit_date": str(dates[j - 1]),
                "hold_days": int(j - i),
                "position_value": float(np.nanmean(np.abs(pv[i:j])) * book),
                "pnl": float(np.nansum(cv[i:j]) * book),
            })
            i = j

    return net_ret, trades


SPEC = StrategySpec(
    id="amihud_illiquidity_premium",
    family="illiquidity",
    title="Amihud Illiquidity Premium (small-cap cross-section)",
    markets=["us_equity"],
    data_desc=(
        "Sharadar SEP daily closeadj (returns) and (close|closeadj)*volume (dollar "
        "volume) for ~1.4k liquidity-bounded SMALL-cap US Domestic Common Stocks spread "
        "across 11 sectors, delisted INCLUDED (survivorship-clean). Amihud (2002) "
        "illiquidity = rolling mean(|daily ret| / daily $ volume)."
    ),
    pre_registration=(
        "HYPOTHESIS: the Amihud illiquidity premium is a universal microstructure "
        "mechanism — names with higher price-impact-per-dollar-traded command higher "
        "expected returns as compensation for bearing illiquidity. PREDICTION: a "
        "weekly, dollar-neutral, inverse-vol L/S book that is long the highest-Amihud "
        "and short the lowest-Amihud names within a liquidity-floored small-cap universe "
        "earns a positive net-of-cost Sharpe. SIGNAL: 63d rolling mean of |ret|/$vol, "
        "ranked cross-sectionally, signals lagged 1 day. FILTERS: only names with 63d "
        "avg $vol > $250k (tradeability). COSTS: 8bps/turnover. SCOPE: declared 'broad' "
        "as a factor premium; it must GENERALISE to other untouched ILLIQUID slices "
        "(other small/mid sub-slices, within-sector sorts). It is NOT expected in large "
        "caps (a known false null — arbitraged away), so 'large' is deliberately excluded "
        "from the generalization set. Holdout 2022+ is forward confirmation."
    ),
    load_data=load_data,
    signal=signal,
    default_params={
        "lookback": 63,
        "n_long": 75,
        "n_short": 75,
        "vol_lb": 63,
        "min_dvol": 2.5e5,
        "cost_bps": 8.0,
        "gross": 1.0,
    },
    grid={
        "default": {},
        "amihud_21": {"lookback": 21},
        "amihud_126": {"lookback": 126},
        "wide_q": {"n_long": 50, "n_short": 50},
        "deep_q": {"n_long": 120, "n_short": 120},
    },
    scope="broad",
    generalization_universes=["small", "mid", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=30,
)