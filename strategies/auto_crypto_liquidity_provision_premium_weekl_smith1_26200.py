# Crypto cross-sectional IDIOSYNCRATIC short-term reversal (USDT-perp proxy)
# ---------------------------------------------------------------------------
# THESIS (faithful to proposal):
#   Universe : top liquid crypto perpetual pairs (proxied by liquid -USD spot
#              pairs in the owned/free yfinance feed; we INCLUDE many tokens that
#              subsequently collapsed/de-listed so the panel is survivorship-aware
#              rather than a "winners only" snapshot — true perp tape is not in the
#              owned catalog, so yf_panel is the closest faithful instrument).
#   Signal   : prior-week return, with the BTC/market component REMOVED (regress
#              each coin's daily returns on BTC over a trailing window, take the
#              idiosyncratic residual; cumulate over the prior week).
#   Direction: REVERSAL — LONG the bottom quintile (idio losers), SHORT the top
#              quintile (idio winners).
#   Sizing   : inverse-vol within each leg, DOLLAR-NEUTRAL (gross 0.5 / 0.5), and
#              an explicit BTC overlay forces portfolio beta-to-BTC ~ 0.
#   Cadence  : WEEKLY rebalance, signals LAGGED 1 day (no look-ahead).
#   Costs    : ~10bps turnover base (default); a 30bps grid leg stresses the
#              all-in trading+funding accrual the proposal calls out (10 -> 30bps).
#   Scope    : LOCAL — the edge is driven by crypto-specific perp microstructure
#              (24/7 retail flow, funding, liquidation cascades) -> forward
#              (out-of-sample, post-2022) validation confirms; not claimed to be a
#              universal cross-asset premium.
# ---------------------------------------------------------------------------

import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel

MARKET = "BTC-USD"

# Liquid crypto universe (yfinance -USD pairs) used as a survivorship-aware
# proxy for the top USDT-perp tape. Whatever the feed cannot return is dropped.
CRYPTO_SECTOR = {
    "BTC-USD": "store-of-value", "LTC-USD": "payments", "BCH-USD": "payments",
    "DOGE-USD": "payments-meme", "DASH-USD": "payments", "ZEC-USD": "privacy",
    "XMR-USD": "privacy",
    "ETH-USD": "smart-contract-L1", "BNB-USD": "smart-contract-L1",
    "ADA-USD": "smart-contract-L1", "SOL-USD": "smart-contract-L1",
    "DOT-USD": "smart-contract-L1", "AVAX-USD": "smart-contract-L1",
    "ATOM-USD": "smart-contract-L1", "NEAR-USD": "smart-contract-L1",
    "ALGO-USD": "smart-contract-L1", "EGLD-USD": "smart-contract-L1",
    "FTM-USD": "smart-contract-L1", "XTZ-USD": "smart-contract-L1",
    "FLOW-USD": "smart-contract-L1", "ICP-USD": "smart-contract-L1",
    "HBAR-USD": "smart-contract-L1", "EOS-USD": "smart-contract-L1",
    "TRX-USD": "smart-contract-L1", "VET-USD": "smart-contract-L1",
    "NEO-USD": "smart-contract-L1", "ONT-USD": "smart-contract-L1",
    "ICX-USD": "smart-contract-L1", "QTUM-USD": "smart-contract-L1",
    "WAVES-USD": "smart-contract-L1", "KSM-USD": "smart-contract-L1",
    "ZIL-USD": "smart-contract-L1", "ETC-USD": "smart-contract-L1",
    "MATIC-USD": "scaling-L2", "LRC-USD": "scaling-L2",
    "XRP-USD": "interop-payments", "XLM-USD": "interop-payments",
    "LINK-USD": "oracle", "GRT-USD": "data-index",
    "FIL-USD": "storage", "STORJ-USD": "storage", "SC-USD": "storage",
    "AAVE-USD": "defi", "MKR-USD": "defi", "COMP-USD": "defi",
    "CRV-USD": "defi", "SNX-USD": "defi", "UNI-USD": "defi",
    "SUSHI-USD": "defi", "1INCH-USD": "defi", "YFI-USD": "defi",
    "BAL-USD": "defi", "REN-USD": "defi", "BAT-USD": "defi",
    "ZRX-USD": "defi", "KAVA-USD": "defi", "RUNE-USD": "defi",
    "ANKR-USD": "infra", "DGB-USD": "payments", "RVN-USD": "payments",
    "IOTA-USD": "iot", "OMG-USD": "scaling-L2",
    "GALA-USD": "gaming-meta", "MANA-USD": "gaming-meta",
    "SAND-USD": "gaming-meta", "AXS-USD": "gaming-meta",
    "ENJ-USD": "gaming-meta", "CHZ-USD": "gaming-meta",
    "THETA-USD": "media",
}
CRYPTO = list(CRYPTO_SECTOR.keys())


def load_data() -> pd.DataFrame:
    # FREE/owned feed; in-memory only, no side effects.
    panel = yf_panel(CRYPTO, start="2017-01-01")
    panel = panel.dropna(how="all")
    # keep only columns with enough usable history
    keep = [c for c in panel.columns if panel[c].notna().sum() > 200]
    return panel[keep].sort_index()


def signal(panel, **params):
    p = dict(lookback=7, beta_lb=30, vol_lb=30, q=0.20, cost_bps=10.0,
             rebalance_days=7, min_names=12, book=1_000_000.0)
    p.update(params)

    panel = panel.sort_index()
    rets = panel.pct_change().clip(-0.95, 5.0)  # guard glitch/zero-price infs

    # --- market (BTC) component & rolling beta -----------------------------
    mkt = rets[MARKET] if MARKET in rets.columns else rets.mean(axis=1)
    var_m = mkt.rolling(p["beta_lb"]).var()
    cov = rets.rolling(p["beta_lb"]).cov(mkt)
    beta = cov.div(var_m, axis=0).clip(-3.0, 3.0).fillna(0.0)

    # --- idiosyncratic prior-week reversal signal --------------------------
    idio = rets - beta.mul(mkt, axis=0)
    sig = idio.rolling(p["lookback"]).sum()          # cumulative idio return, prior week
    vol = rets.rolling(p["vol_lb"]).std()            # for inverse-vol sizing

    idx = list(panel.index)
    warmup = max(p["beta_lb"], p["vol_lb"], p["lookback"]) + 2

    wmap, selmap = {}, {}                            # full weights (incl hedge) / selection weights
    for pos in range(warmup, len(idx), p["rebalance_days"]):
        d = idx[pos]
        s = sig.loc[d].dropna()
        if MARKET in s.index:
            s = s.drop(MARKET)                       # BTC idio ~ 0; reserve it as the hedge leg
        v = vol.loc[d].reindex(s.index)
        b = beta.loc[d].reindex(s.index)
        m = v.notna() & (v > 0) & b.notna() & s.notna()
        s, v, b = s[m], v[m], b[m]
        n = len(s)
        if n < p["min_names"]:
            continue

        k = max(1, int(round(n * p["q"])))
        order = s.sort_values()
        longs = order.index[:k]                      # idio LOSERS  -> LONG  (reversal)
        shorts = order.index[-k:]                    # idio WINNERS -> SHORT (reversal)

        lw = 1.0 / v[longs]; lw = lw / lw.sum() * 0.5
        sw = 1.0 / v[shorts]; sw = sw / sw.sum() * 0.5

        sel = pd.Series(0.0, index=s.index)
        sel[longs] = lw                              # dollar-neutral: +0.5 gross long
        sel[shorts] = -sw                            #                 -0.5 gross short
        selmap[d] = sel[sel != 0.0].copy()

        # full traded weights + BTC beta overlay (force portfolio beta-to-BTC ~ 0)
        w = pd.Series(0.0, index=rets.columns)
        w.loc[sel.index] = sel.values
        if MARKET in rets.columns:
            bb = beta.loc[d].reindex(w.index).fillna(0.0)
            net_beta = float((w * bb).sum())
            w.loc[MARKET] = w.loc[MARKET] - net_beta
        wmap[d] = w

    if not wmap:
        empty = pd.Series(dtype=float); empty.name = SPEC.id
        return empty, []

    # --- daily net-of-cost portfolio returns (weights LAGGED 1 day) --------
    target = pd.DataFrame(wmap).T.reindex(columns=rets.columns).fillna(0.0)
    target = target.reindex(panel.index).ffill().fillna(0.0)
    held = target.shift(1).fillna(0.0)               # no look-ahead
    port = (held * rets).sum(axis=1)
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * (p["cost_bps"] / 1e4)              # ~10bps turnover (30bps grid leg = all-in funding stress)
    net = port - cost

    first_d = min(wmap.keys())
    net = net.loc[net.index >= first_d]
    net.name = SPEC.id

    # --- trades: one per contiguous held run (factor-book sanity) ----------
    book = p["book"]
    trades = []

    def emit(tk, entry_date, exit_date, sign, weight):
        if entry_date == exit_date:
            return
        try:
            pe = float(panel.at[entry_date, tk]); px = float(panel.at[exit_date, tk])
        except Exception:
            return
        if not (np.isfinite(pe) and np.isfinite(px)) or pe <= 0:
            return
        posval = float(abs(weight) * book)
        if posval <= 0:
            return
        raw = sign * (px / pe - 1.0)
        pnl = raw * posval - 2.0 * (p["cost_bps"] / 1e4) * posval   # round-trip cost
        hold = int(max(1, (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days))
        trades.append({
            "ticker": tk,
            "sector": CRYPTO_SECTOR.get(tk, "other"),
            "entry_date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
            "exit_date": pd.Timestamp(exit_date).strftime("%Y-%m-%d"),
            "hold_days": hold,
            "position_value": posval,
            "pnl": float(pnl),
        })

    sel_df = pd.DataFrame(selmap).T.reindex(columns=rets.columns).sort_index()
    sgn = np.sign(sel_df.fillna(0.0))
    rb_dates = list(sel_df.index)
    for tk in sel_df.columns:
        s_col = sgn[tk]; w_col = sel_df[tk]
        prev, entry_date, entry_w = 0.0, None, None
        for d in rb_dates:
            cur = float(s_col.loc[d])
            if cur != prev:
                if prev != 0.0 and entry_date is not None:
                    emit(tk, entry_date, d, prev, entry_w)
                if cur != 0.0:
                    entry_date, entry_w = d, float(w_col.loc[d])
                else:
                    entry_date, entry_w = None, None
                prev = cur
        if prev != 0.0 and entry_date is not None and entry_date != rb_dates[-1]:
            emit(tk, entry_date, rb_dates[-1], prev, entry_w)

    return net, trades


SPEC = StrategySpec(
    id="crypto_xsec_idio_reversal",
    family="reversal",
    title="Crypto cross-sectional idiosyncratic short-term reversal (USDT-perp proxy)",
    markets=["crypto"],
    data_desc=(
        "Daily close of ~60 liquid crypto -USD pairs (yfinance, FREE) used as a "
        "survivorship-AWARE proxy for the top USDT-perp tape: many tokens that "
        "subsequently collapsed/de-listed are retained rather than a winners-only "
        "snapshot. The true perpetual + funding tape is not in the owned catalog, "
        "so spot daily close is the closest faithful instrument. History from "
        "2017-01-01; columns with <200 usable observations are dropped."
    ),
    pre_registration=(
        "Thesis: crypto short-term cross-sectional REVERSAL on the BTC-neutral "
        "(idiosyncratic) prior-week return. Each coin's daily returns are regressed "
        "on BTC over a trailing 30d window; the residual is cumulated over the prior "
        "7 days. LONG the bottom-quintile idio losers, SHORT the top-quintile idio "
        "winners. Inverse-vol sizing within each leg, dollar-neutral (0.5/0.5 gross); "
        "a BTC overlay forces portfolio beta-to-BTC ~ 0. WEEKLY rebalance, signals "
        "LAGGED 1 day (no look-ahead). Costs ~10bps on turnover (default); a 30bps "
        "grid leg stresses all-in trading+funding accrual. Economic driver: 24/7 "
        "retail overreaction, funding mean-reversion and liquidation-cascade "
        "snapbacks are crypto-microstructure specific -> scope LOCAL, to be "
        "confirmed by post-2022 out-of-sample forward validation. FALSIFIED if the "
        "idio-reversal Sharpe is <=0 out-of-sample, or if the edge is purely a "
        "residual BTC-beta artefact (vanishes once beta-neutralised)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb14": {"lookback": 14},
        "q15": {"q": 0.15},
        "beta_lb60": {"beta_lb": 60},
        "cost_30bps": {"cost_bps": 30.0},
    },
    scope="local",
    generalization_universes=[],
    holdout_start="2022-01-01",
    deploy_max_positions=25,
)