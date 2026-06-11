"""Hedge-sleeve exemption in deployment_sanity (designed 2026-06-12).

Origin: forge night 2026-06-11 — two genuine Amihud near-misses (holdout Sharpe 1.44/1.31,
DSR ~1.0, 15/15 CPCV paths positive) were killed by single_name_share because their
PRE-REGISTERED IWM residual-beta hedge dominates position-days. The gate could not tell a
declared hedge instrument from an accidental concentration bet.

Anti-loophole invariants (each has a test):
  1. hedge tickers must be on the frozen ETF whitelist — declaring an alpha stock as a
     "hedge" is a forced fail, not an exemption.
  2. the ALPHA book is judged alone: excluding the hedge must not weaken any existing
     check (trade count, peak concurrency, single-name share all computed ex-hedge).
  3. hedge share of total position-days must be <= the declared cap — an oversized
     "hedge" is the falsified ETF-substitution variant in disguise.
  4. no declaration -> byte-identical behavior to the old gate (frozen designs unaffected).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_integrity import deployment_sanity  # noqa: E402


def _trade(tk, entry, exit_, pv=1000.0, sector="Tech"):
    return {"ticker": tk, "entry_date": entry, "exit_date": exit_,
            "position_value": pv, "sector": sector}


def _alpha_book(n_names=20, n_rounds=4):
    """A healthy multi-name book: 20 names x 4 monthly rounds = 80 trades, equal weight."""
    trades = []
    months = [("2020-01-02", "2020-01-30"), ("2020-02-03", "2020-02-27"),
              ("2020-03-02", "2020-03-30"), ("2020-04-01", "2020-04-29")][:n_rounds]
    sectors = ["Tech", "Energy", "Health", "Finance", "Retail"]
    for i in range(n_names):
        for e, x in months:
            trades.append(_trade(f"C{i:03d}", e, x, pv=1000.0, sector=sectors[i % 5]))
    return trades


def _hedge_trades(share_pv, n_rounds=4):
    """A continuously-held IWM short sized so it takes ~share of total position-days."""
    months = [("2020-01-02", "2020-01-30"), ("2020-02-03", "2020-02-27"),
              ("2020-03-02", "2020-03-30"), ("2020-04-01", "2020-04-29")][:n_rounds]
    return [_trade("IWM", e, x, pv=share_pv, sector="ETF") for e, x in months]


def test_declared_hedge_exempt_from_single_name_share():
    """THE near-miss scenario: clean 20-name alpha book + IWM hedge at ~47% of
    position-days, hedge declared with cap 0.50 -> must PASS."""
    trades = _alpha_book() + _hedge_trades(share_pv=18_000.0)  # 18k vs 20k alpha -> ~47%
    out = deployment_sanity(trades, strategy_meta={"max_positions": 20},
                            hedge_tickers=["IWM"], hedge_cap=0.50)
    assert out["passed"], out["forced_fail_reasons"]
    assert out["hedge_share"] > 0.40          # would have tripped the old gate
    assert out["single_name_share"] <= 0.40   # alpha book judged alone


def test_undeclared_hedge_still_fails():
    """Same book, NO declaration -> old behavior byte-for-byte: forced fail."""
    trades = _alpha_book() + _hedge_trades(share_pv=18_000.0)
    out = deployment_sanity(trades, strategy_meta={"max_positions": 20})
    assert not out["passed"]
    assert any("single_name_share" in r for r in out["forced_fail_reasons"])


def test_hedge_cap_breach_fails():
    """Hedge at ~64% of position-days with declared cap 0.35 -> forced fail on hedge_share."""
    trades = _alpha_book() + _hedge_trades(share_pv=36_000.0)
    out = deployment_sanity(trades, strategy_meta={"max_positions": 20},
                            hedge_tickers=["IWM"], hedge_cap=0.35)
    assert not out["passed"]
    assert any("hedge_share" in r for r in out["forced_fail_reasons"])


def test_non_whitelisted_hedge_declaration_fails():
    """Declaring a single STOCK as 'hedge' to dodge concentration -> forced fail."""
    trades = _alpha_book(n_names=3) + [_trade("AAPL", "2020-01-02", "2020-04-29",
                                              pv=50_000.0)]
    out = deployment_sanity(trades, strategy_meta={"max_positions": 5},
                            hedge_tickers=["AAPL"], hedge_cap=0.80)
    assert not out["passed"]
    assert any("whitelist" in r for r in out["forced_fail_reasons"])


def test_alpha_book_must_stand_alone():
    """2-name alpha book + legal hedge: excluding the hedge must EXPOSE the thin book
    (peak/trade-count floors bite ex-hedge), not mask it."""
    trades = _alpha_book(n_names=2) + _hedge_trades(share_pv=2_000.0)
    out = deployment_sanity(trades, strategy_meta={"max_positions": 20},
                            hedge_tickers=["IWM"], hedge_cap=0.50)
    assert not out["passed"]   # 8 alpha trades < MIN_TRADES, peak 2 < floor


def test_no_hedge_unchanged():
    """No hedge involved, no declaration -> identical to the legacy gate."""
    trades = _alpha_book()
    legacy = deployment_sanity(trades, strategy_meta={"max_positions": 20})
    new = deployment_sanity(trades, strategy_meta={"max_positions": 20},
                            hedge_tickers=None, hedge_cap=None)
    assert legacy == new
    assert legacy["passed"]
