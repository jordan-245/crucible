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

from agent.propose import propose
from agent.scout import scout
from sdk import queue
from sdk.locks import FileLock, LockTimeout

TARGET = 4  # keep at least this many items queued


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40]


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
        added = 0
        need = target - queue.stats().get("queued", 0)
        for _ in range(min(need, max_new)):
            prop = propose()
            if "error" in prop:
                continue
            key = _norm(prop.get("title", ""))
            if not key or key in tested or key in inflight:
                continue  # dedup vs recorded experiments + in-flight
            queue.enqueue(prop)
            inflight.add(key)
            added += 1
        return {"added": added, **queue.stats()}
    finally:
        lock.release()


if __name__ == "__main__":
    print(top_up())
