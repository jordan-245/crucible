#!/usr/bin/env python3
"""Forward-paper evidence accumulator (LOOPS_FRAMEWORK_PLAN 2.3).

Weekly loop: score every deployed forward book against the board's pre-registered
go-live gate (memo 2026-06-09-forge-go-live-policy), write the verdict to the wiki,
and queue a trajectory notice for the morning report. Telegram-critical ONLY if the
gate flips to PASS (capital decision needs the human) — everything else is report
material.

Gate criteria (pre-registered, frozen here — do not tune to make the book pass):
  G1  fills        >= 40 executed orders
  G2  days         >= 20 trading days of recorded returns
  G3  expectancy   mean daily net return > 0
  G4  regimes      >= 2 market regimes covered, each with >= 5 trading days.
                   Regime def (pre-registered 2026-06-11): sign of IWM trailing
                   21-day return on each forward date (up / down market).
  G5  reconciliation  book.json positions == broker positions (within 0.5%)
                   — checked daily by the shadow loop; here we assert no
                   'blocked' run rows (a mismatch halts and records blocked).
  G6  slippage     median fill-vs-decision slippage <= 2x modeled cost (8bps for
                   val_mom => bar 16bps). Day-1 book-build excluded (one-off
                   position establishment at open after overnight gap — not the
                   steady-state rebalance cost the gate regulates; recorded anyway).
  G7  broker-error rate of ok=False order placements < 1%

State: wiki forward/<book>.md (trajectory table appended weekly).
Usage: python3 forward/evidence.py [--book val_mom_trend_smallcap]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crucible_paths import WIKI  # noqa: E402

LIVE = Path("/root/atlas/data/live")
MIN_FILLS = 40
MIN_DAYS = 20
MIN_REGIME_DAYS = 5


def _jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _iwm_regimes(dates: list[str]) -> dict[str, str]:
    """Regime per forward date: sign of IWM trailing 21d return (pre-registered def)."""
    import pandas as pd
    # IWM is an ETF — absent from Sharadar SEP (stocks only). yfinance is fine here:
    # the survivorship anti-pattern is about cross-sectional stock selection, not a
    # single index ETF used as a regime flag.
    from sdk.adapters import yf_panel
    panel = yf_panel(["IWM"], start="2025-06-01")
    px = panel["IWM"].dropna().sort_index()
    trail = px.pct_change(21)
    out = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts in trail.index and pd.notna(trail.loc[ts]):
            out[d] = "up" if trail.loc[ts] > 0 else "down"
        else:  # date beyond cached data (e.g. today) — use last known trailing value
            prior = trail.loc[:ts].dropna()
            out[d] = ("up" if prior.iloc[-1] > 0 else "down") if len(prior) else "?"
    return out


MODELED_COST_BPS = {"val_mom_trend_smallcap": 8.0}  # from each frozen design's cost spec
SLIPPAGE_MULT = 2.0
MAX_BROKER_ERR = 0.01


def evaluate(book: str) -> dict:
    d = LIVE / book
    returns = _jsonl(d / "returns.jsonl")
    runs = _jsonl(d / "runs.jsonl")
    fills = _jsonl(d / "fills.jsonl")

    n_days = len(returns)
    n_fills = sum(len(r.get("orders") or []) for r in runs if not r.get("dry_run"))
    rets = [float(r["ret"]) for r in returns if r.get("ret") is not None]
    expectancy = sum(rets) / len(rets) if rets else None
    blocked = [r for r in runs if r.get("blocked")]

    dates = [r["date"] for r in returns]
    try:
        regs = _iwm_regimes(dates)
        counts: dict[str, int] = {}
        for v in regs.values():
            counts[v] = counts.get(v, 0) + 1
        regimes_covered = sum(1 for k, c in counts.items() if k != "?" and c >= MIN_REGIME_DAYS)
        regime_detail = counts
    except Exception as e:
        regimes_covered, regime_detail = None, {"error": str(e)[:80]}

    gates = {
        "G1_fills": {"value": n_fills, "need": MIN_FILLS, "pass": n_fills >= MIN_FILLS},
        "G2_days": {"value": n_days, "need": MIN_DAYS, "pass": n_days >= MIN_DAYS},
        "G3_expectancy": {"value": round(expectancy, 6) if expectancy is not None else None,
                          "need": "> 0", "pass": bool(expectancy and expectancy > 0)},
        "G4_regimes": {"value": regimes_covered, "detail": regime_detail, "need": 2,
                       "pass": bool(regimes_covered and regimes_covered >= 2)},
        "G5_reconciliation": {"value": len(blocked), "need": "0 blocked runs",
                              "pass": not blocked},
        "G6_slippage": _g6(book, returns, fills),
        "G7_broker_errors": _g7(runs),
    }
    scoreable = [g for g in gates.values() if g["pass"] is not None]
    # PASS requires ALL SEVEN gates scoreable and green — a gate without data is not a pass
    verdict = "PASS" if len(scoreable) == len(gates) and all(g["pass"] for g in scoreable) \
        else "ACCUMULATING"
    return {"book": book, "asof": datetime.now().strftime("%Y-%m-%d"),
            "verdict": verdict, "gates": gates,
            "equity": (returns[-1].get("equity") if returns else None)}


def _g6(book: str, returns: list, fills: list) -> dict:
    import statistics
    # exclude the day-1 book build: one-off establishment cost, not steady-state rebalance.
    # Build day = the EARLIEST fill date (returns.jsonl starts a day later — first return
    # needs a prior close — so keying on returns[0] would exclude the wrong day).
    build_day = min((f["date"] for f in fills if f.get("date")), default=None)
    sl = [f["slippage_bps"] for f in fills
          if f.get("slippage_bps") is not None and f.get("date") != build_day]
    modeled = MODELED_COST_BPS.get(book)
    if not sl or modeled is None:
        return {"value": None, "need": f"median <= {SLIPPAGE_MULT}x modeled", "pass": None,
                "note": f"{len(sl)} steady-state fills — accumulating"}
    med = statistics.median(sl)
    bar = SLIPPAGE_MULT * modeled
    return {"value": round(med, 1), "need": f"<= {bar:.0f}bps (2x {modeled:.0f}bps modeled)",
            "pass": med <= bar, "n_fills": len(sl)}


def _g7(runs: list) -> dict:
    placed = [o for r in runs if not r.get("dry_run") and not r.get("blocked")
              for o in r.get("orders", []) if o.get("ok") is not None]
    if not placed:
        return {"value": None, "need": f"< {MAX_BROKER_ERR:.0%}", "pass": None,
                "note": "no ok-flagged orders yet (field added 2026-06-11)"}
    err = sum(1 for o in placed if not o["ok"]) / len(placed)
    return {"value": round(err, 4), "need": f"< {MAX_BROKER_ERR:.0%}", "pass": err < MAX_BROKER_ERR,
            "n_orders": len(placed)}


def write_wiki(ev: dict) -> Path:
    page = WIKI / "forward" / f"{ev['book']}.md"
    page.parent.mkdir(exist_ok=True)
    if not page.exists():
        page.write_text(
            f"# Forward-paper evidence — {ev['book']}\n\n"
            "Board go-live gate ([[2026-06-09 forge-go-live-policy]]): >=40 fills, >=20 days, "
            "+ve net expectancy, >=2 regimes (IWM trailing-21d sign, pre-registered "
            "2026-06-11), clean reconciliation; slippage/broker-error gates pending fill "
            "data. Real capital additionally needs the AUM floor + human approval.\n\n"
            "| asof | verdict | fills | days | expectancy | regimes | recon | slip | err | equity |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    g = ev["gates"]

    def _m(key):
        v = g[key]
        mark = "?" if v["pass"] is None else ("✅" if v["pass"] else "✗")
        return f"{v['value']} {mark}"

    exp = g["G3_expectancy"]["value"]
    row = (f"| {ev['asof']} | {ev['verdict']} | {_m('G1_fills')} | {_m('G2_days')} | "
           f"{f'{exp * 1e4:+.1f}bps' if exp is not None else '?'} "
           f"{'✅' if g['G3_expectancy']['pass'] else '✗'} | "
           f"{_m('G4_regimes')} | {_m('G5_reconciliation')} | "
           f"{_m('G6_slippage')} | {_m('G7_broker_errors')} | "
           f"${ev['equity']:,.0f} |\n" if ev.get("equity") else "n/a |\n")
    with page.open("a") as f:
        f.write(row)
    return page


def main() -> int:
    books = [p.name for p in LIVE.iterdir()
             if (p / "returns.jsonl").exists()] if LIVE.exists() else []
    if not books:
        print("[evidence] no forward books found")
        return 0
    for book in books:
        ev = evaluate(book)
        page = write_wiki(ev)
        g = ev["gates"]
        traj = (f"📋 forward-evidence {book}: {ev['verdict']} — "
                f"fills {g['G1_fills']['value']}/{MIN_FILLS}, days {g['G2_days']['value']}/{MIN_DAYS}, "
                f"expectancy {f'{g['G3_expectancy']['value'] * 1e4:+.1f}bps' if g['G3_expectancy']['value'] is not None else '?'}, "
                f"regimes {g['G4_regimes']['value']}/2, recon {'clean' if g['G5_reconciliation']['pass'] else 'BLOCKED RUNS'}")
        print("[evidence]", traj)
        try:
            if ev["verdict"] == "PASS":
                from sdk.notify import telegram_critical
                telegram_critical(f"🟢 <b>Go-live gate PASS</b> — {book} cleared all scoreable "
                                  f"forward-paper criteria.\n{traj}\nCapital decision requires "
                                  f"human + AUM floor check (memo 2026-06-09). "
                                  f"See wiki/forward/{book}.md")
            else:
                from sdk.notify import notice
                notice(traj, source="forward-evidence")
        except Exception as e:
            print(f"[evidence] notify failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
