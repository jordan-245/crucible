"""
Value + Momentum complementary cross-sectional equity book (mid-cap US).

Economic premise (pre-registered):
  Cross-sectional VALUE (earnings-yield + book-yield) and 12-1 MOMENTUM are each
  individually positive-premium equity anomalies that are NEGATIVELY correlated
  (cheap names are usually down, winners are usually expensive). Combining them in
  ONE inverse-vol, weekly-rebalanced, long-only mid-cap book should keep the premium
  while cutting the drawdown of either leg alone. Here the 50/50 blend is justified
  because BOTH legs are real premia (this is NOT a 0-Sharpe trend overlay) — and we
  pre-declare value-only / mom-only standalone variants in the grid so the
  combination's claim is falsifiable.

No external side-effects. Owned Sharadar data only (SEP prices, SF1 fundamentals).
"""

import warnings
import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1

START = "2005-01-01"
ID = "auto_value_momentum_complementary_equity_comb"

SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Energy", "Basic Materials",
    "Real Estate", "Communication Services", "Utilities",
]


def _xs_rank(df, min_n=20):
    """Cross-sectional pct-rank per day; rows with < min_n valid names -> NaN."""
    r = df.rank(axis=1, pct=True)
    ok = df.notna().sum(axis=1) >= min_n
    # broadcast the row-mask to the full panel shape (where() needs same shape)
    mask = np.broadcast_to(ok.to_numpy()[:, None], r.shape)
    return r.where(mask, np.nan)


def _as_panel(f, field, dates, cols):
    """Filing-date (datekey) -> daily as-of forward-filled panel. NO look-ahead."""
    f = f.reset_index()
    if "ticker" not in f.columns or "datekey" not in f.columns or field not in f.columns:
        return pd.DataFrame(np.nan, index=dates, columns=cols)
    d = f[["ticker", "datekey", field]].dropna(subset=["datekey"]).copy()
    d["datekey"] = pd.to_datetime(d["datekey"])
    piv = d.pivot_table(index="datekey", columns="ticker", values=field, aggfunc="last").sort_index()
    full = piv.index.union(dates)
    piv = piv.reindex(full).ffill().reindex(dates)
    return piv.reindex(columns=cols)


def load_data() -> pd.DataFrame:
    # ---- survivorship-clean universe, spread across sectors (gives us sector map too)
    ticker_sector = {}
    for s in SECTORS:
        try:
            ts = us_universe(sector=s, category="Domestic Common Stock",
                             marketcap="Mid", include_delisted=True, top_n=120)
        except Exception:
            ts = []
        for t in ts:
            ticker_sector[t] = s
    tickers = sorted(ticker_sector)

    # ---- prices (delisted incl, split+div adjusted)
    px = sep_panel(tickers, start=START, field="closeadj")
    px = px.sort_index()
    px = px[~px.index.duplicated()]
    cols = list(px.columns)
    idx = px.index

    # ---- fundamentals, as-of FILING date (ART = trailing-twelve-month, as reported)
    f = sf1(tickers, ["eps", "bvps"], dimension="ART")
    eps_p = _as_panel(f, "eps", idx, cols)
    bvps_p = _as_panel(f, "bvps", idx, cols)

    # ---- signals (all backward-looking)
    # momentum 12-1: return from t-252 to t-21 (skip most recent month)
    mom = px.shift(21) / px.shift(252) - 1.0
    mom_rank = _xs_rank(mom)

    # value: earnings-yield + book-yield (higher = cheaper = better)
    pxs = px.replace(0.0, np.nan)
    ey = eps_p / pxs
    by = bvps_p / pxs
    ey_rank = _xs_rank(ey)
    by_rank = _xs_rank(by)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.nanmean(np.stack([ey_rank.to_numpy(), by_rank.to_numpy()]), axis=0)
    val_rank = pd.DataFrame(v, index=ey_rank.index, columns=ey_rank.columns)

    return {"px": px, "mom_rank": mom_rank, "val_rank": val_rank, "sectors": ticker_sector}


def signal(panel, **params):
    p = dict(w_val=0.5, w_mom=0.5, max_pos=30, vol_lb=60)
    p.update(params or {})

    px = panel["px"]
    mr = panel["mom_rank"]
    vr = panel["val_rank"]
    sect = panel["sectors"]

    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)
    vol = rets.rolling(int(p["vol_lb"]), min_periods=20).std()

    # composite score (ignore zero-weight legs so *_only variants don't require the other)
    parts = []
    if p["w_mom"] > 0:
        parts.append(p["w_mom"] * mr)
    if p["w_val"] > 0:
        parts.append(p["w_val"] * vr)
    comp = parts[0].copy()
    for q in parts[1:]:
        comp = comp + q

    idx = px.index
    cols = px.columns
    iso = idx.isocalendar()
    key = pd.Series((iso["year"].astype(int).astype(str) + "_" +
                     iso["week"].astype(int).astype(str)).values, index=idx)
    rebal = idx[(key != key.shift(1)).to_numpy()]
    maxp = int(p["max_pos"])

    # ---- weekly target weights (inverse-vol among top-N composite names), long-only
    W = pd.DataFrame(np.nan, index=idx, columns=cols)
    for d in rebal:
        sc = comp.loc[d].dropna()
        if len(sc) < maxp:
            continue
        sel = sc.sort_values(ascending=False).head(maxp).index
        iv = (1.0 / vol.loc[d, sel]).replace([np.inf, -np.inf], np.nan).dropna()
        iv = iv[iv > 0]
        if iv.empty:
            continue
        w = iv / iv.sum()
        row = pd.Series(0.0, index=cols)
        row[w.index] = w.values
        W.loc[d] = row.values

    W = W.ffill().fillna(0.0)
    W = W.shift(1).fillna(0.0)  # lag signals 1 day -> no look-ahead

    # ---- net daily returns (8bps on turnover)
    gross = (W * rets).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    cost = turnover * 0.0008
    net = (gross - cost).fillna(0.0)

    active = W.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[idx[active.to_numpy().argmax()]:]
    net.name = ID

    # ---- trades: one per contiguous held run (deployment-sanity)
    CAP = 1_000_000.0
    trades = []
    dates = W.index
    n = len(dates)
    for t in cols:
        wcol = W[t].to_numpy()
        inpos = wcol > 1e-9
        if not inpos.any():
            continue
        pser = px[t]
        i = 0
        while i < n:
            if inpos[i]:
                j = i
                while j + 1 < n and inpos[j + 1]:
                    j += 1
                entry, ex = dates[i], dates[j]
                ep, xp = pser.loc[entry], pser.loc[ex]
                if np.isfinite(ep) and np.isfinite(xp) and ep > 0:
                    pv = float(np.mean(wcol[i:j + 1])) * CAP
                    r = xp / ep - 1.0
                    pnl = pv * r - pv * 0.0016  # round-trip cost
                    trades.append({
                        "ticker": t,
                        "sector": sect.get(t, "Unknown"),
                        "entry_date": entry.strftime("%Y-%m-%d"),
                        "exit_date": ex.strftime("%Y-%m-%d"),
                        "hold_days": int(j - i + 1),
                        "position_value": round(pv, 2),
                        "pnl": round(pnl, 2),
                    })
                i = j + 1
            else:
                i += 1

    return net, trades


SPEC = StrategySpec(
    id=ID,
    family="equity_factor_combo",
    title="Value + Momentum complementary mid-cap equity book",
    markets=["US_EQUITY"],
    data_desc=("Sharadar SEP survivorship-clean adj-close prices + SF1 ART fundamentals "
               "(eps, bvps) over a mid-cap US universe spread across 11 sectors (~1300 names)."),
    pre_registration=(
        "H: cross-sectional VALUE (E/P+B/P) and 12-1 MOMENTUM are each positive-premium but "
        "negatively-correlated mid-cap equity anomalies; a 50/50 inverse-vol, weekly, long-only "
        "combination should retain the premium while cutting drawdown vs either leg alone. "
        "Falsifiable: grid pre-declares value-only & mom-only standalone variants — the "
        "combination is only credible if it does not dilute standalone Sharpe while smoothing the "
        "ride. Holdout from 2022-01-01. ~8bps turnover cost, signals lagged 1 day."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"w_val": 0.5, "w_mom": 0.5, "max_pos": 30, "vol_lb": 60},
    grid={
        "default": {},
        "value_only": {"w_val": 1.0, "w_mom": 0.0},
        "mom_only": {"w_val": 0.0, "w_mom": 1.0},
        "value_tilt": {"w_val": 0.7, "w_mom": 0.3},
        "mom_tilt": {"w_val": 0.3, "w_mom": 0.7},
        "concentrated": {"max_pos": 20},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=30,
)