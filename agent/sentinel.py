"""Data-integrity sentinel — daily READ-ONLY checks across the research/trading estate.

The safest possible loop (alert-only, zero writes) and the TEMPLATE for new loops
(cf. tasks/LOOPS_FRAMEWORK_PLAN.md 2.2). Every check is a pure assertion against an
objective source of truth; failures are collected and sent as ONE Telegram message.
Silence = healthy. Exit code 1 if any check fails (systemd visibility).

Checks:
  S1  Sharadar freshness        SEP.zip mtime within N days (data pipeline alive)
  S2  SEP cache schema          cached parquet columns ⊇ source columns we depend on
                                (the 2026-06-10 closeunadj gap, generalized)
  S3  SEP cache staleness       cache at least as new as SEP.zip (rebuild trigger works)
  S4  Forward-paper liveness    returns.jsonl last row within N business days + finite
  S5  Forward-paper equity      book equity within sane band (fat-finger / runaway)
  S6  Wiki off-box backup       last nightly auto-push commit is fresh (off-box memory)
  S7  Queue health              research queue not empty-and-stale (forge starvation)
  S8  Run-log heartbeat         forge produced rows on the last scheduled night
  S9  Holdout ledger integrity  jsonl parses, no duplicate config_hash entries
  S10 Loop registry sync        every live crucible/atlas timer has a row in wiki loops.md
                                (comprehension-debt rule: unregistered loop = stray)
  S11 Forward-paper log scan    yesterday's cycle log carries no 'FAILED' step markers
                                (the steps are '|| echo FAILED' guarded — exit 0 lies)

Usage: python3 -m agent.sentinel
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from crucible_paths import DATA, WIKI, QUEUE, RUN_LOG, REGISTRY

SHARADAR = DATA / "sharadar"
CACHE = DATA / "cache" / "sep_long_v2.parquet"
FWD = Path("/root/atlas/data/live/val_mom_trend_smallcap")
REQUIRED_FIELDS = ("close", "closeadj", "closeunadj", "volume")

MAX_SHARADAR_AGE_D = 7        # weekly refresh cadence + slack
MAX_RETURNS_AGE_BD = 3        # forward-paper runs Mon-Fri 23:45
MAX_WIKI_PUSH_AGE_H = 36      # nightly 06:30 backup + slack
MAX_RUNLOG_GAP_D = 2          # forge runs nightly 03:30
EQUITY_BAND = (10_000, 25_000)  # val_mom book: $14.5K capital; outside this = investigate


def _age_days(p: Path) -> float | None:
    return (time.time() - p.stat().st_mtime) / 86400 if p.exists() else None


def _bdays_since(date_str: str) -> int:
    import pandas as pd
    return len(pd.bdate_range(date_str, datetime.now().strftime("%Y-%m-%d"))) - 1


def check_sharadar(fail):
    sep = SHARADAR / "SEP.zip"
    age = _age_days(sep)
    if age is None:
        fail(f"S1 Sharadar: {sep} MISSING")
    elif age > MAX_SHARADAR_AGE_D:
        fail(f"S1 Sharadar: SEP.zip is {age:.1f}d old (> {MAX_SHARADAR_AGE_D}d) — refresh pipeline dead?")


def check_sep_cache(fail):
    if not CACHE.exists():
        fail(f"S2 SEP cache: {CACHE.name} missing (first strategy run will pay the rebuild)")
        return
    try:
        import pyarrow.parquet as pq
        cols = set(pq.ParquetFile(CACHE).schema_arrow.names)
        missing = [f for f in REQUIRED_FIELDS if f not in cols]
        if missing:
            fail(f"S2 SEP cache: missing fields {missing} — strategies needing them will runtime_error")
    except Exception as e:
        fail(f"S2 SEP cache: unreadable ({type(e).__name__}: {e})")
    src = SHARADAR / "SEP.zip"
    if src.exists() and CACHE.exists() and src.stat().st_mtime > CACHE.stat().st_mtime:
        fail("S3 SEP cache: OLDER than SEP.zip — rebuild trigger failed (stale prices feeding rails)")


def check_forward_paper(fail):
    rj = FWD / "returns.jsonl"
    if not rj.exists():
        fail(f"S4 forward-paper: {rj} missing — recorder never ran?")
        return
    rows = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]
    if not rows:
        fail("S4 forward-paper: returns.jsonl EMPTY")
        return
    last = rows[-1]
    gap = _bdays_since(last["date"])
    if gap > MAX_RETURNS_AGE_BD:
        fail(f"S4 forward-paper: last return {last['date']} ({gap} bdays ago > {MAX_RETURNS_AGE_BD}) — daily cycle dead?")
    r = last.get("ret")
    if r is None or not (-0.5 < float(r) < 0.5):
        fail(f"S4 forward-paper: insane daily return {r} on {last['date']}")
    eq = last.get("equity")
    if eq is not None and not (EQUITY_BAND[0] <= float(eq) <= EQUITY_BAND[1]):
        fail(f"S5 forward-paper: equity ${eq:,.0f} outside band ${EQUITY_BAND[0]:,}-${EQUITY_BAND[1]:,}")


def check_wiki_pushed(fail):
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%ct %s"], cwd=str(WIKI),
                           capture_output=True, text=True, timeout=30)
        ts, _, msg = r.stdout.strip().partition(" ")
        age_h = (time.time() - int(ts)) / 3600
        if age_h > MAX_WIKI_PUSH_AGE_H:
            fail(f"S6 wiki backup: last commit {age_h:.0f}h ago (> {MAX_WIKI_PUSH_AGE_H}h) — nightly snapshot dead? ('{msg[:40]}')")
        # verify it actually landed off-box, not just committed locally
        r2 = subprocess.run(["git", "status", "-sb"], cwd=str(WIKI), capture_output=True, text=True, timeout=30)
        if "ahead" in r2.stdout:
            fail(f"S6 wiki backup: local commits NOT pushed ({r2.stdout.strip().splitlines()[0]})")
    except Exception as e:
        fail(f"S6 wiki backup: git check failed ({type(e).__name__}: {e})")


def check_queue(fail):
    if not QUEUE.exists():
        fail(f"S7 queue: {QUEUE} missing")
        return
    rows = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    open_items = [r for r in rows if r.get("status") in ("queued", "claimed")]
    if not open_items and _age_days(QUEUE) and _age_days(QUEUE) > 2:
        fail("S7 queue: no open items and file stale >2d — director top-up failing (forge will idle)")


def check_run_log(fail):
    if not RUN_LOG.exists():
        fail(f"S8 run-log: {RUN_LOG} missing")
        return
    age = _age_days(RUN_LOG)
    if age is not None and age > MAX_RUNLOG_GAP_D:
        fail(f"S8 run-log: no forge activity for {age:.1f}d (> {MAX_RUNLOG_GAP_D}d) — nightly forge dead?")


def check_holdout_ledger(fail):
    led = REGISTRY / "holdout_ledger.jsonl"
    if not led.exists():
        return  # legitimate pre-first-run state
    seen, dupes = set(), []
    for i, l in enumerate(led.read_text().splitlines()):
        if not l.strip():
            continue
        try:
            h = json.loads(l).get("config_hash")
        except json.JSONDecodeError:
            fail(f"S9 holdout ledger: line {i + 1} unparseable — write-once enforcement compromised")
            return
        if h in seen:
            dupes.append(h)
        seen.add(h)
    if dupes:
        fail(f"S9 holdout ledger: DUPLICATE config_hash entries {dupes[:3]} — a second OOS look was recorded; investigate")


def check_loop_registry(fail):
    """S10: every live crucible/atlas timer must be registered in wiki loops.md."""
    reg = WIKI / "loops.md"
    if not reg.exists():
        fail("S10 loop registry: research-wiki/loops.md missing")
        return
    txt = reg.read_text()
    try:
        r = subprocess.run(["systemctl", "list-timers", "--all", "--no-legend", "--plain"],
                           capture_output=True, text=True, timeout=10)
        timers = [l.split()[-2] for l in r.stdout.splitlines()
                  if l.strip() and ("crucible-" in l or "atlas-" in l)]
    except Exception as e:
        fail(f"S10 loop registry: systemctl unavailable ({e})")
        return
    strays = [t for t in timers if t.removesuffix(".timer") not in txt]
    if strays:
        fail(f"S10 loop registry: live timers NOT in loops.md: {strays} — "
             f"register or disable (unregistered loop = stray)")


def check_forward_paper_log(fail):
    """S11: ops/forward-paper.sh guards each step with '|| echo ... FAILED' so the unit
    exits 0 even when a step dies — scan the last cycle's log block for the markers."""
    log = Path("/root/atlas/data/live/forward_paper.log")
    if not log.exists():
        return  # S4 already covers total absence via returns.jsonl
    lines = log.read_text().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("=== forward-paper cycle")]
    if not starts:
        return
    last_block = lines[starts[-1]:]
    failed = [l for l in last_block if "FAILED" in l]
    if failed:
        fail(f"S11 forward-paper log: last cycle had failed steps: {failed} "
             f"(unit exited 0 — this is the only place the failure shows)")


CHECKS = [check_sharadar, check_sep_cache, check_forward_paper,
          check_wiki_pushed, check_queue, check_run_log, check_holdout_ledger,
          check_loop_registry, check_forward_paper_log]


def main() -> int:
    failures: list[str] = []
    fail = failures.append
    for c in CHECKS:
        try:
            c(fail)
        except Exception as e:   # a crashed check is itself a failure (never silent)
            fail(f"{c.__name__}: check crashed ({type(e).__name__}: {str(e)[:120]})")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if failures:
        for f in failures:
            print(f"[sentinel] FAIL {f}")
        from sdk.notify import telegram_msg
        telegram_msg("🛰 <b>Data-integrity sentinel</b> — " + stamp + "\n"
                     + "\n".join("❌ " + f for f in failures))
        return 1
    print(f"[sentinel] {stamp}: all {len(CHECKS)} checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
