"""Director: keep the shared work queue topped with deduped, prioritized, UNTESTED hypotheses.

Runs under the 'director' lock so only one top-up happens at a time even if several workers
invoke it when the queue runs dry. Dedup is against recorded experiments + everything in-flight
(queued/claimed), so N workers never get the same hypothesis.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path("/root/hephaestus")
sys.path.insert(0, str(ROOT))
WIKI = Path("/root/research-wiki")

from agent.propose import propose, mutate as propose_mutate
from agent.scout import scout
from agent import elite
from sdk import queue
from sdk.locks import FileLock, LockTimeout

TARGET = 4  # keep at least this many items queued


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40]


def _theme(prop: dict) -> str:
    """Coarse FAMILY key (robust to LLM rewording) so the 2/theme queue cap catches sibling mutations
    (e.g. reworded value×mom variants) instead of treating each as a novel premium."""
    from agent.families import family_bucket
    return family_bucket(str(prop.get("title", "")) or str(prop.get("premium", "")))


def _tested_titles() -> set[str]:
    return {_norm(p.stem) for p in (WIKI / "experiments").glob("*.md")}


def top_up(target: int = TARGET, max_new: int = 4) -> dict:
    """Enqueue up to max_new fresh, deduped hypotheses to reach `target` queued items.
    Returns fast (added=0, skipped) if a peer already holds the director lock — the caller
    should then just retry claim_next; the peer is filling the queue."""
    try:
        lock = FileLock("director", ttl=900, wait=8).acquire()
    except LockTimeout:
        return {"added": 0, "skipped": "director busy", **queue.stats()}
    try:
        if queue.stats().get("queued", 0) >= target:
            return {"added": 0, **queue.stats()}
        if random.random() < 0.4:
            try:
                scout()  # occasionally pull fresh external ideas into candidates.md first
            except Exception:
                pass
        tested, inflight = _tested_titles(), queue.inflight_titles()
        # seed per-theme counts from what's already in-flight, so we cap clustering (max 2/theme)
        themes: dict[str, int] = {}
        for it in queue._read_all():
            if it["status"] in ("queued", "claimed"):
                themes[_theme(it["proposal"])] = themes.get(_theme(it["proposal"]), 0) + 1
        added = 0
        need = target - queue.stats().get("queued", 0)
        for _ in range(min(need, max_new) * 3):  # extra tries to find DIVERSE ideas
            if added >= min(need, max_new):
                break
            e = elite.sample(random.Random()) if (random.random() < 0.4 and elite.top()) else None
            prop = propose_mutate(e) if e else propose()  # 40% EXPLOIT (evolve an elite) / 60% EXPLORE (fresh)
            if "error" in prop:
                continue
            key, th = _norm(prop.get("title", "")), _theme(prop)
            if th in elite._closed_families():
                continue  # HARD gate: family closed by decision (cf. decisions/CLOSED.md) — never enqueue
            if not key or key in tested or key in inflight or themes.get(th, 0) >= 2:
                continue  # dedup vs recorded experiments + in-flight + theme cluster cap
            queue.enqueue(prop)
            inflight.add(key)
            themes[th] = themes.get(th, 0) + 1
            added += 1
        return {"added": added, **queue.stats()}
    finally:
        lock.release()


if __name__ == "__main__":
    print(top_up())
