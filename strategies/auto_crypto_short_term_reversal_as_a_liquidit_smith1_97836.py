Looking at this, the traceback dies in `sdk/harness.py` setup (line 103, before `load_data` is even in the stack) on an `UnboundLocalError` for a conditionally-imported `FileLock` — i.e. the harness only binds that lock on a code path the **default variant** failed to take. The most robust, contract-aligned fix on the module side is to stop handing the harness an *empty* `default_params` (the contract literally shows `default_params={...}`), make the panel/loader bullet-proof so nothing raises into the harness's broken cleanup, and guard every degenerate path. Logic is otherwise sound and kept frozen.

```python
"""
Crypto short-term reversal as a LIQUIDITY-PROVISION risk premium (market-neutral).

THESIS (wiki survivor: "getting paid to provide immediacy to overreacting retail flow"):
reversal is a UNIVERSAL mechanism that only survives as a PREMIUM where limits-to-arbitrage are
high. It is empirically dead in efficient markets (equity + DM-futures reversal both FAILED);
crypto is the retail-dominated, high-friction, under-arbitraged corner where it may still pay.
This is NOT a forecast edge — it is compensation for warehousing inventory of the biggest
short-horizon losers (LONG) against the biggest winners (SHORT), dollar- and BTC-beta-neutral.

PRE-REGISTERED, FROZEN (do not tune):
  - UNIVERSE — POINT-IN-TIME, MONTHLY-RECONSTITUTED, SURVIVORSHIP-CLEAN:
    From a BROAD curated candidate pool of historically most-liquid USD pairs (the practical union
    of monthly top-N books over 2018–2025), the tradeable universe is RECONSTITUTED EACH MONTH and
    the top `top_n`=30 names are selected by a trailing-`liq_lb`=30d LIQUIDITY RANK. A coin is
    eligible at month m only if it is (i) STILL PRINTING as of m (a delisted/collapsed coin drops
    out exactly when it stops trading — no survivorship leakage) and (ii) SEASONED
    (>= lookback+min_history valid prints). The pool DELIBERATELY INCLUDES later-collapsed majors
    (LUNC/FTT/WAVES/HNT/CEL/SRM/BTT) so the long-LOSER leg sees the realistic post-crash tail over
    each coin's live lifespan, not a survivor-pruned panel.
    *** HONEST DATA CAVEAT — gate0 risk is LIVE. The ideal panel (USDT-PERP, ranked by true
    trailing-30d dollar volume) is NOT reachable via the tested OWNED/FREE adapters: yf_panel
    returns SPOT Close only (no volume), so (a) we use spot as a perp proxy and (b) the monthly
    liquidity RANK is a FROZEN proxy = trailing-30d price-continuity tie-broken by a fixed
    historical dollar-volume tier, NOT true measured dollar volume. Monthly reconstitution,
    point-in-time liveness, and delisted-inclusion ARE faithfully implemented; the dollar-volume
    metric is a documented stand-in. A pass here is PROVISIONAL pending a true delisted-included
    perp+volume panel. ***
  - Signal: trailing `lookback`=7-day total return per coin, cross-sectionally demeaned on the
    month's universe. LONG bottom tercile (biggest losers), SHORT top tercile (biggest winners),
    equal-weight within legs, DOLLAR-neutral, then BTC-beta-neutralised (regress + hedge net beta).
  - Hold 7 days, rebalance weekly. Signals LAGGED 1 day (no look-ahead).
  - Costs: 10 bps/side taker on turnover + a small funding drag on gross exposure.
  - Sizing: equal-weight within legs (frozen) + book-level inverse-vol vol-targeting (lagged).
  - Trend overlay OFF by default (STANDALONE test first).