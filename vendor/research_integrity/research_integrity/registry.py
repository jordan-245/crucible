"""Rail 2 — hypothesis registry (cross-family multiple-testing context).

The effective-N DSR (search_history.py) corrects for the config search WITHIN a strategy. It does
NOT correct for the thousands of DISTINCT ideas an industrialized loop throws at the gate. This
registry logs every battery run's FAMILY (coarse hypothesis class) so the PROMOTION bar can escalate
with the cumulative count of distinct ideas tested (see adapter.promote_dsr). Counting FAMILIES, not
configs, avoids both under- and over-counting (the grid is already DSR-deflated within a family).

Append-only JSONL, pure/deterministic except file I/O. Spec: research/INTEGRITY_RAILS_SPEC.md (Rail 2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import os
from .holdout import _LazyPath
REGISTRY = _LazyPath("hypothesis_registry.jsonl")  # resolved at use time (see holdout._state_dir)


def family_of(strategy: str) -> str:
    """Coarse hypothesis class. Default = strategy name (one idea, many grid configs)."""
    return str(strategy)


def _read_families(path: Optional[Path] = None) -> set:
    p = path or REGISTRY
    fams: set = set()
    if not p.exists():
        return fams
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            fams.add(json.loads(line).get("family"))
        except Exception:
            continue
    fams.discard(None)
    return fams


def distinct_families(extra: Optional[str] = None, path: Optional[Path] = None) -> int:
    """Number of distinct families ever tested, optionally including a not-yet-logged one.

    Always >= 1 (a single idea still carries its own bar at the base level).
    """
    fams = _read_families(path)
    if extra:
        fams.add(extra)
    return max(1, len(fams))


def append_run(record: dict, path: Optional[Path] = None) -> None:
    p = path or REGISTRY
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


__all__ = ["family_of", "distinct_families", "append_run", "REGISTRY"]
