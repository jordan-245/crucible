"""
Crypto-native time-series momentum (trend) — STANDALONE crisis-alpha premium.

Hypothesis (LOCAL, deliberately): the trend / time-series-momentum MECHANISM is
universal, but the *standalone positive* premium survives only where institutional
arbitrage has not yet fully arrived. It is empirically ~0 in efficient DM futures
([[boreas-tsmom]] -> only the HEDGE leg survived there), but is hypothesised >0 in
the young, retail-dominated, capital-constrained liquid-crypto cross-section with
strong reflexive trends and slow arb capital.

UNIVERSE (FROZEN, registered AS-IS — this is the thesis, not a placeholder):
a FIXED, fully-specified static basket of 15 persistent, liquid crypto majors
(USD spot pairs). This is a deliberately SIMPLE, completely-specified cross-section
— it is NOT a dynamically re-ranked top-N-by-ADV book and it does NOT include
dead/delisted coins. The reason is honest and pre-committed: the sanctioned crypto
source exposed to this harness is yf_panel (crypto USD pairs via yfinance), which
(a) carries NO volume — so trailing-ADV ranking is IMPOSSIBLE in-harness — and
(b) is a survivorship-biased LIVE set (dead coins/perps absent) and exposes NO perp
funding. We therefore TEST the static-liquid-major basket as the registered thesis,
and FLAG three honestly-acknowledged biases as Gate-0 gaps to close on a real
Binance/Bybit pipe (with volume + delisted perps + funding) BEFORE any capital:
  - no dynamic ADV ranking (static basket proxies "liquid majors"),
  - no dead-coin inclusion (in-harness set is survivorship-biased),
  - no explicit per-coin funding (for a long/short-balanced trend book, funding paid
    on longs ~ offsets funding earned on shorts to second order; only the realistic
    10bps taker cost on turnover is applied). Treat the in-harness result as a
    funding-agnostic SPOT-trend proxy.

Pre-registered PRIMARY = sign of an equal-weight blend of trailing 1m/3m/6m total
returns, per coin, inverse-realised-vol sized, weekly rebalance, gross capped at 2x,
10bps taker cost on turnover. Everything in `grid` is a robustness check; the primary
("default") is the verdict. The rails (CPCV / DSR / regime splits / forward) are the
arbiter; this module only produces net daily returns + trades.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Universe: FIXED static basket of 15 liquid persistent crypto majors. Registered
# AS-IS (yf_panel carries no volume -> ADV ranking is impossible in-harness, and the
# set is survivorship-biased; both flagged as Gate-0 gaps, NOT part of the thesis).
# Long history so the sample spans the 2021 bull AND 2022 bear (>=2 distinct regimes).
# "sector" tags give the deployment-sanity check a cross-sectional spread.
# ----------------------------------------------------------------------------------
SECTOR = {
    "BTC-USD": "Store-of-Value",
    "ETH-USD": "SmartContract",
    "BNB-USD": "Exchange",
    "XRP-USD": "Payments",
    "ADA-USD": "SmartContract",
    "SOL-USD": "SmartContract",
    "DOGE-USD": "Meme",
    "DOT-USD": "Interop",
    "AVAX-USD": "SmartContract",
    "MATIC-USD": "Scaling",
    "LINK-USD": "Oracle",
    "LTC-USD": "Payments",
    "BCH-USD": "Payments",
    "UNI-USD": "DeFi",
    "ATOM-USD": "Interop",
}
TICKERS = list(SECTOR.keys())
START = "2019-01-01"
NOTIONAL = 10_000.0  # nominal book $ for trade-level position_value / pnl reporting

DEFAULTS = dict(
    lb1=21, lb3=63, lb6=126,   # 1m / 3m / 6m lookbacks (crypto trades 7d/wk)
    vol_lb=30,                 # realised-vol lookback for inverse-vol sizing
    target_vol=0.20,           # per-coin annualised vol target
    per_coin_cap=3.0,          # cap on a single coin's pre-gross inv-vol weight
    max_gross=2.0,             # hard gross-leverage cap
    cost_bps=10.0,             # taker cost on turnover (bps)
)


def load_data() -> pd.DataFrame:
    """Daily Close panel for the FIXED liquid-crypto basket (FREE via yf_panel)."""
    px = yf_panel(TICKERS, start=START)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.sort_index()
    px = px.dropna(axis=1, how="all")           # drop any ticker that returned nothing
    px = px[[c for c in TICKERS if c in px.columns]]
    return px


def _runs(sign_vec):
    """Yield (i, j, sign) for each maximal run of constant non-zero sign."""
    n = len(sign_vec)
    i = 0
    while i < n:
        s = sign_vec[i]
        if not np.isfinite(s) or s == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign_vec[j + 1] == s:
            j += 1
        yield i, j, s
        i = j + 1


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    lb1, lb3, lb6 = int(p["lb1"]), int(p["lb3"]), int(p["lb6"])
    vol_lb = int(p["vol_lb"])
    target_vol = float(p["target_vol"])
    per_coin_cap = float(p["per_coin_cap"])
    max_gross = float(p["max_gross"])
    cost_bps = float(p["cost_bps"])

    px = panel.sort_index().astype(float)
    rets = px.pct_change(fill_method=None)

    # --- signal: sign of equal-weight blend of trailing 1m/3m/6m total returns ---
    r1 = px.pct_change(lb1, fill_method=None)
    r3 = px.pct_change(lb3, fill_method=None)
    r6 = px.pct_change(lb6, fill_method=None)
    blend = (r1 + r3 + r6) / 3.0
    raw_sig = np.sign(blend)                      # +1 long / -1 short / 0/NaN flat

    # --- inverse realised-vol sizing to a per-coin vol target (crypto: 365d/yr) ---
    dvol = rets.rolling(vol_lb, min_periods=max(5, vol_lb // 2)).std()
    ann_vol = dvol * np.sqrt(365.0)
    inv_vol_w = (target_vol / ann_vol).clip(upper=per_coin_cap)
    target_w = raw_sig * inv_vol_w                # signed target weights (pre-rebal/lag)

    # --- weekly rebalance: hold weights, only update on the last trading day of week ---
    idx = px.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    last_in_week = pos.groupby(idx.to_period("W")).transform("max")
    is_rebal = pd.Series(pos.values == last_in_week.values, index=idx)

    held = target_w.copy()
    held[~is_rebal.values] = np.nan
    held = held.ffill()

    # --- NO look-ahead: lag positions 1 day (signal computed at t -> traded from t+1) ---
    weights = held.shift(1)

    # --- hard gross-leverage cap at max_gross ---
    gross = weights.abs().sum(axis=1)
    scale = np.minimum(1.0, max_gross / gross.replace(0.0, np.nan)).fillna(1.0)
    weights = weights.mul(scale, axis=0).fillna(0.0)

    # --- net-of-cost daily portfolio returns ---
    rets_f = rets.fillna(0.0)
    gross_ret = (weights * rets_f).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = (gross_ret - cost).rename("crypto_tsmom")

    # trim warm-up (all-flat) prefix
    active = weights.abs().sum(axis=1).gt(0)
    if active.any():
        net = net.loc[active.idxmax():]

    # --- trades: one per held position run (sign-stable spell) ---
    contrib = weights * rets_f
    sgn = np.sign(weights).values
    cols = list(weights.columns)
    dates = weights.index
    trades = []
    for k, coin in enumerate(cols):
        wv = weights[coin].values
        cv = contrib[coin].values
        for i, j, s in _runs(sgn[:, k]):
            seg_w = np.abs(wv[i:j + 1])
            avg_w = float(np.nanmean(seg_w)) if seg_w.size else 0.0
            if avg_w <= 0:
                continue
            pnl = float(np.nansum(cv[i:j + 1])) * NOTIONAL
            trades.append({
                "ticker": coin,
                "sector": SECTOR.get(coin, "Crypto"),
                "entry_date": dates[i].strftime("%Y-%m-%d"),
                "exit_date": dates[j].strftime("%Y-%m-%d"),
                "hold_days": int(j - i + 1),
                "position_value": round(avg_w * NOTIONAL, 2),
                "pnl": round(pnl, 2),
            })

    return net, trades


SPEC = StrategySpec(
    id="crypto-native-tsmom-standalone",
    family="trend",
    title="Crypto-native time-series momentum (standalone crisis-alpha premium)",
    markets=["crypto"],
    data_desc=(
        "Daily Close panel for a FIXED static basket of 15 persistent, liquid crypto "
        "majors (USD spot pairs) via yf_panel (FREE). The basket is REGISTERED AS-IS "
        "(a simple, fully-specified cross-section) — it is NOT a dynamic top-N-by-ADV "
        "book and does NOT include dead/delisted coins, because yf_panel carries no "
        "volume (ADV ranking impossible in-harness), is a survivorship-biased LIVE set "
        "(dead coins absent), and exposes NO perp funding. Acknowledged Gate-0 gaps to "
        "close on a real Binance/Bybit pipe (volume + delisted perps + funding) before "
        "any capital: dynamic ADV ranking, dead-coin inclusion, and explicit per-coin "
        "funding. Sample spans the 2021 bull and 2022 bear (>=2 regimes)."
    ),
    pre_registration=(
        "PRIMARY (the verdict). UNIVERSE = a FIXED, fully-specified STATIC basket of 15 "
        "persistent liquid crypto majors (USD spot pairs); this static basket IS the "
        "registered thesis — NOT a dynamically re-ranked top-N-by-ADV book and NOT "
        "including dead/delisted coins (yf_panel exposes no volume and is a "
        "survivorship-biased live set, so ADV ranking and dead-coin inclusion are "
        "IMPOSSIBLE in-harness and are registered as KNOWN Gate-0 gaps, not as part of "
        "the tested claim). PER-COIN SIGNAL = sign of the equal-weight blend of trailing "
        "1m/3m/6m (21/63/126d) total returns (long if >0, short if <0); inverse-realised-"
        "vol sized to a 20% per-coin annualised vol target (365d/yr); WEEKLY POSITION "
        "rebalance (hold between the last trading day of each week); gross capped at 2x; "
        "10bps taker cost on turnover; positions lagged 1 day (no look-ahead). Funding is "
        "NOT modelled in-harness (treated as ~net-neutral for a long/short-balanced book; "
        "flagged for the real Binance/Bybit pipe pre-deployment). DIRECTION OF TEST: "
        "standalone POSITIVE net-of-cost expectancy in the liquid-crypto cross-section "
        "(unlike ~0 in efficient DM futures). Robustness checks (grid, NOT new "
        "hypotheses): faster/slower lookbacks, lower vol target, higher (20bps) cost. "
        "Confirm WITHIN crypto: holds across the alt basket (not just BTC/ETH) and is "
        "positive in BOTH 2021-bull and 2022-bear sub-samples. NO reflexive 50/50 blend "
        "with anything; the near-miss carry (Midas) leg would only be added LATER as a "
        "SMALL tail-overlay iff it cuts the tail without diluting standalone Sharpe — and "
        "never by re-opening Midas in-sample."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},  # primary = module DEFAULTS
    grid={
        "default": {},                                   # PRIMARY
        "fast":        {"lb1": 10, "lb3": 30, "lb6": 60},
        "slow":        {"lb1": 42, "lb3": 84, "lb6": 168},
        "lower_vol":   {"target_vol": 0.15},
        "higher_cost": {"cost_bps": 20.0},
    },
    scope="local",
    generalization_universes=[],  # local edge: forward-validation on fresh crypto is the arbiter
    holdout_start="2022-01-01",   # 2022 crypto bear held out as the OOS regime
    deploy_max_positions=15,
)