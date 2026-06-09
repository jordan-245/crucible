"""
Cross-asset futures HEDGING-PRESSURE premium (price-space implementation).

ECONOMIC PRIOR (Keynes normal-backwardation / Cootner-Hirshleifer hedging pressure):
commercial HEDGERS pay an insurance premium to lay off price risk; the speculators who
ABSORB that risk are compensated. When commercials are unusually net-SHORT a market
(producers hedging output after weak prices) speculators are forced net-LONG and earn
the premium -> go LONG that market; when commercials are unusually net-LONG, go SHORT.

DATA-FAITHFUL CAVEAT (the honest fix): the owned/free data inventory exposes NO CFTC
COT positioning feed (no commercial long/short series via the available adapters), so the
positioning index itself cannot be computed. I therefore test the PRICE-SPACE counterpart
the same theory predicts: hedging pressure is highest in markets that have suffered long
multi-year declines (producers heavily hedged -> commercials net-short -> spec premium),
i.e. a slow cross-sectional LONG-TERM REVERSAL ("value") across a diversified futures book.
This is mechanistically distinct from short-horizon TREND/momentum (we skip the recent
month and use a 3-5yr window) and is tested STANDALONE -- NO reflexive trend hedge (a
~0-Sharpe 50/50 trend blend would simply halve the edge).

SIGNAL
  (1) per market: long-term reversal r = -(P[t-skip]/P[t-lookback] - 1)  (skip recent month).
  (2) cross-sectional rank each day across the diversified book.
  (3) LONG top tercile (most depressed = most hedger-pressured) / SHORT bottom tercile.
  (4) inverse-vol sizing (equal risk per market), dollar/risk-neutral long vs short.
  (5) weekly rebalance, ex-ante portfolio vol target, signals lagged 1 day, ~8bps costs.
"""
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, inv_vol_position
import numpy as np
import pandas as pd

# Liquid cross-asset Yahoo continuous front-month futures. sector == asset class
# (used purely for the trade-level diversification sanity checks).
SECTORS = {
    "ES=F": "Equity", "NQ=F": "Equity", "YM=F": "Equity", "RTY=F": "Equity",
    "ZB=F": "Rates", "ZN=F": "Rates", "ZF=F": "Rates", "ZT=F": "Rates",
    "6E=F": "FX", "6J=F": "FX", "6B=F": "FX", "6A=F": "FX", "6C=F": "FX", "6S=F": "FX",
    "CL=F": "Energy", "NG=F": "Energy", "RB=F": "Energy", "HO=F": "Energy",
    "GC=F": "Metals", "SI=F": "Metals", "HG=F": "Metals", "PL=F": "Metals",
    "ZC=F": "Ags", "ZS=F": "Ags", "ZW=F": "Ags", "KC=F": "Ags",
    "SB=F": "Ags", "CT=F": "Ags", "LE=F": "Ags", "HE=F": "Ags",
}
TICKERS = list(SECTORS.keys())
START = "2006-01-01"
_MIN_OBS = 504  # require ~2yr of history before a contract enters the book


def load_data() -> pd.DataFrame:
    """Close panel of liquid cross-asset futures (free, via yf_panel).

    Defensive: drop all-NaN and sparsely-populated/late-listed columns so the harness
    never receives a degenerate panel.
    """
    panel = yf_panel(TICKERS, start=START)
    if panel is None or panel.empty:
        return pd.DataFrame()
    panel = panel.sort_index()
    panel = panel.dropna(how="all", axis=1).dropna(how="all", axis=0)
    # keep only contracts with a real history (avoids degenerate single-name books)
    good = [c for c in panel.columns if panel[c].notna().sum() >= _MIN_OBS]
    panel = panel[good]
    panel = panel.dropna(how="all", axis=0)
    return panel


def _extract_trades(positions: pd.DataFrame, rets: pd.DataFrame,
                    notional: float = 1_000_000.0) -> list:
    """One trade per held position run (constant-sign stretch) per market."""
    trades = []
    idx = positions.index
    for tk in positions.columns:
        w = positions[tk].fillna(0.0).to_numpy()
        r = rets[tk].reindex(idx).fillna(0.0).to_numpy()
        sgn = np.sign(w)
        n = len(w)
        i = 0
        while i < n:
            if sgn[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and sgn[j + 1] == sgn[i]:
                j += 1
            w_run, r_run = w[i:j + 1], r[i:j + 1]
            trades.append({
                "ticker": tk,
                "sector": SECTORS.get(tk, "Other"),
                "entry_date": idx[i].strftime("%Y-%m-%d"),
                "exit_date": idx[j].strftime("%Y-%m-%d"),
                "hold_days": int(j - i + 1),
                "position_value": float(np.mean(np.abs(w_run)) * notional),
                "pnl": float(np.sum(w_run * r_run) * notional),
            })
            i = j + 1
    return trades


def signal(panel, **params):
    p = {
        "lookback": 252 * 4,   # ~4yr long-term reversal (hedging-pressure proxy)
        "skip": 21,            # skip recent month -> not short-horizon trend
        "vol_lb": 63,
        "target_vol": 0.10,
        "max_pos": None,
        "rebalance": "W-FRI",
        "tercile": 1.0 / 3.0,
        "cost_bps": 8.0,
    }
    p.update(params)

    # --- degenerate-input guard (never raise inside the harness) ---
    if panel is None or panel.shape[1] < 6 or panel.shape[0] < (p["lookback"] + 10):
        empty = pd.Series(dtype=float, name="hedging_pressure_xasset")
        return empty, []

    px = panel.sort_index()
    rets = px.pct_change()
    max_pos = int(p["max_pos"]) if p["max_pos"] else px.shape[1]

    # --- raw signal: long-term price reversal (cheap = hedger-pressured = LONG) ---
    raw = -(px.shift(p["skip"]) / px.shift(p["lookback"]) - 1.0)

    valid = raw.notna().sum(axis=1)
    rank = raw.rank(axis=1, pct=True)
    long_mask = rank >= (1.0 - p["tercile"])
    short_mask = rank <= p["tercile"]
    sig = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    sig = sig.mask(long_mask, 1.0).mask(short_mask, -1.0)
    sig = sig.where(raw.notna(), 0.0)
    sig.loc[valid < 6, :] = 0.0   # need a real cross-section to rank

    # inverse-vol sizing + weekly hold + 1-day lag (handled by the adapter)
    positions = inv_vol_position(
        sig, rets, p["target_vol"], p["vol_lb"], max_pos, p["rebalance"]
    )
    positions = positions.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

    gross = (positions * rets).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (p["cost_bps"] / 1e4)
    net = (gross - cost).fillna(0.0)

    active = positions.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    net.name = "hedging_pressure_xasset"

    if net.empty:
        return net, []
    trades = _extract_trades(positions.loc[net.index], rets.loc[net.index])
    return net, trades


SPEC = StrategySpec(
    id="hedging_pressure_xasset_futures",
    family="hedging_pressure",
    title="Cross-Asset Futures Hedging-Pressure Premium (price-space long-term reversal)",
    markets=["futures"],
    data_desc="Yahoo continuous front-month futures (yf_panel): ~30 liquid cross-asset "
              "contracts (equity-index, rates, FX, energy, metals, ags), 2006-present. "
              "No CFTC COT feed available -> positioning thesis tested via its price-space "
              "counterpart (slow long-term reversal).",
    pre_registration=(
        "Hedging-pressure / normal-backwardation premium: markets that have declined over a "
        "multi-year window (3-5yr, skipping the recent month) are where commercials are most "
        "net-short and speculators are paid to absorb risk. LONG top reversal tercile / SHORT "
        "bottom tercile, inverse-vol risk-balanced, dollar/risk-neutral, weekly rebalance, "
        "10% vol target, signals lagged 1 day, ~8bps turnover cost. Tested STANDALONE; no "
        "trend hedge. Distinct from short-horizon trend (recent month skipped) and from "
        "price-carry (no curve/basis input). Edge expected only with a positive net-of-cost "
        "long-short expectancy across regimes; holdout from 2022-01-01."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "rev3y": {"lookback": 252 * 3},
        "rev5y": {"lookback": 252 * 5},
        "tercile25": {"tercile": 0.25},
    },
    scope="local",                      # defensibly universe-specific cross-asset futures book
    generalization_universes=[],        # no equity slice maps to a futures book; holdout confirms
    holdout_start="2022-01-01",
    deploy_max_positions=20,
)