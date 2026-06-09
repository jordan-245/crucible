import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, inv_vol_position


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SPEC_ID = "auto_value_momentum_complementary_combination_smith3_28986"
START = "2009-01-01"
UNIVERSE_TOP_N = 1200
COST_BPS = 8.0

# Sharadar (Morningstar-style) sector labels — used only for the trades ledger.
_SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Industrials", "Consumer Defensive", "Energy", "Basic Materials",
    "Real Estate", "Communication Services", "Utilities",
]
_SECTOR_MAP = {}


def _build_sector_map(tickers):
    """Map each ticker -> sector by intersecting per-sector universe pulls."""
    global _SECTOR_MAP
    if _SECTOR_MAP:
        return _SECTOR_MAP
    tset = set(tickers)
    mp = {}
    for s in _SECTORS:
        try:
            names = us_universe(sector=s, category="Domestic Common Stock",
                                include_delisted=True, top_n=10000)
        except Exception:
            names = []
        for t in names:
            if t in tset:
                mp[t] = s
    for t in tickers:
        mp.setdefault(t, "Unknown")
    _SECTOR_MAP = mp
    return mp


def _asof_panel(fund, field, ref_index):
    """Forward-fill a Sharadar fundamental onto the daily index, AS-OF its
    FILING date (datekey) -> no look-ahead."""
    df = fund.reset_index()
    lower = {c.lower(): c for c in df.columns}
    tcol = lower.get("ticker")
    dcol = lower.get("datekey") or lower.get("date")
    fcol = lower.get(field.lower(), field)
    df = df[[tcol, dcol, fcol]].dropna()
    df.columns = ["ticker", "datekey", "val"]
    df["datekey"] = pd.to_datetime(df["datekey"])
    df = df.sort_values("datekey")
    wide = df.pivot_table(index="datekey", columns="ticker",
                          values="val", aggfunc="last").sort_index()
    return wide.reindex(wide.index.union(ref_index)).ffill().reindex(ref_index)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    # Mid-caps: value & momentum premia live OUTSIDE the arbitraged megacaps.
    tickers = us_universe(category="Domestic Common Stock", marketcap="Mid",
                          include_delisted=True, top_n=UNIVERSE_TOP_N)

    price = sep_panel(tickers, start=START, field="closeadj")
    price = price.dropna(axis=1, how="all").sort_index()

    fund = sf1(list(price.columns), ["bvps"], dimension="ARQ")
    bvps = _asof_panel(fund, "bvps", price.index)

    # Book-to-market (book value per share / price): higher = cheaper = value
    btm = bvps.reindex(columns=price.columns) / price

    panel = pd.concat({"price": price, "btm": btm}, axis=1)
    panel.attrs["sectors"] = _build_sector_map(list(price.columns))
    return panel


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def _zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)


def signal(panel, **params):
    price = panel["price"]
    btm = panel["btm"]

    mom_lb = int(params.get("mom_lb", 252))
    mom_gap = int(params.get("mom_gap", 21))
    n_long = int(params.get("n_long", 60))
    target_vol = float(params.get("target_vol", 0.10))
    vol_lb = int(params.get("vol_lb", 63))
    val_w = float(params.get("val_w", 0.5))

    rets = price.pct_change()

    # 12-1 momentum (skip most recent month) and value (log book-to-market)
    mom = price.shift(mom_gap) / price.shift(mom_lb) - 1.0
    val = btm.where(btm > 0)

    valid = mom.notna() & val.notna()
    z_mom = _zscore(mom.where(valid))
    z_val = _zscore(np.log(val.where(valid)))

    combo = val_w * z_val + (1.0 - val_w) * z_mom

    rank = combo.rank(axis=1, ascending=False)
    sig = rank.le(n_long).astype(float)
    sig = sig.where(sig > 0)  # NaN where not selected

    # inverse-vol size + weekly rebalance + 1-day lag (handled by the adapter)
    positions = inv_vol_position(sig, rets, target_vol=target_vol,
                                 vol_lb=vol_lb, max_pos=n_long, rebalance="W")
    positions = positions.reindex(index=rets.index,
                                  columns=rets.columns).fillna(0.0)

    gross = (positions * rets.fillna(0.0)).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    net = (gross - turnover * (COST_BPS / 1e4))

    # Trim leading flat period until the book is actually invested.
    active = positions.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    net = net.dropna()
    net.name = SPEC_ID

    trades = _build_trades(positions, rets, panel.attrs.get("sectors", {}))
    return net, trades


def _build_trades(positions, rets, sector_map):
    """One trade per contiguous held run of a name (factor-book convention)."""
    contrib = positions * rets.fillna(0.0)
    idx = positions.index
    trades = []
    for t in positions.columns:
        p = positions[t].values
        held = p > 1e-9
        if not held.any():
            continue
        c = contrib[t].values
        n = len(p)
        i = 0
        while i < n:
            if held[i]:
                j = i
                while j + 1 < n and held[j + 1]:
                    j += 1
                trades.append({
                    "ticker": t,
                    "sector": sector_map.get(t, "Unknown"),
                    "entry_date": idx[i].strftime("%Y-%m-%d"),
                    "exit_date": idx[j].strftime("%Y-%m-%d"),
                    "hold_days": int(j - i + 1),
                    "position_value": float(np.nanmean(p[i:j + 1])),
                    "pnl": float(np.nansum(c[i:j + 1])),
                })
                i = j + 1
            else:
                i += 1
    return trades


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id=SPEC_ID,
    family="value_momentum",
    title="Complementary Value + Momentum mid-cap composite (long-only)",
    markets=["US_EQUITY"],
    data_desc=("Sharadar SEP survivorship-clean adjusted closes + SF1 bvps "
               "(as-of filing datekey) for ~1200 liquid US mid-caps."),
    pre_registration=(
        "Value (book-to-market) and momentum (12-1) are the two most-replicated "
        "cross-sectional equity premia and are negatively correlated; a 50/50 "
        "z-score composite long book of the top mid-cap names should earn a "
        "positive net-of-cost Sharpe (>0.4) out-of-sample on the 2022+ holdout "
        "and GENERALISE to untouched large/small/sector slices. If the edge only "
        "appears in mid-caps it is an overfit outlier, not a universal premium."
    ),
    load_data=load_data,
    signal=signal,
    default_params={
        "mom_lb": 252, "mom_gap": 21, "n_long": 60,
        "target_vol": 0.10, "vol_lb": 63, "val_w": 0.5,
    },
    grid={
        "default": {},
        "mom_6m": {"mom_lb": 126},
        "value_tilt": {"val_w": 0.7},
        "momentum_tilt": {"val_w": 0.3},
        "concentrated": {"n_long": 40},
    },
    scope="broad",
    generalization_universes=["large", "small", "sectors"],
    holdout_start="2022-01-01",
    deploy_max_positions=20,
)