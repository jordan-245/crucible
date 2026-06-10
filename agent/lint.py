"""Wiki lint — memory hygiene (selective forgetting: the thing the memory research says everyone fails at).
Succinct + cheap (structural, no LLM): cap the append-only log, prune already-tested candidates, flag orphan
pages, report health. Run weekly (hephaestus-lint.timer) or on demand: python3 -m agent.lint"""
import re
from datetime import datetime
from pathlib import Path

WIKI = Path("/root/research-wiki")
LOG_KEEP = 60  # keep the most recent N log entries; archive the rest


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40]


def cap_log():
    log = WIKI / "log.md"
    if not log.exists():
        return 0
    lines = log.read_text().splitlines()
    entries = [i for i, l in enumerate(lines) if l.startswith("## [")]
    if len(entries) <= LOG_KEEP:
        return 0
    cut = entries[-LOG_KEEP]
    with (WIKI / "log-archive.md").open("a") as f:
        f.write("\n".join(lines[:cut]) + "\n")
    head = lines[0] if lines and lines[0].startswith("#") else "# Log"
    log.write_text(head + "\n" + "\n".join(lines[cut:]) + "\n")
    return len(entries) - LOG_KEEP


def prune_candidates():
    """Drop candidates whose idea is already an experiment (tested) — keep the queue fresh."""
    cand = WIKI / "candidates.md"
    if not cand.exists():
        return 0
    tested = {_norm(p.stem) for p in (WIKI / "experiments").glob("*.md")}
    kept, dropped = [], 0
    for l in cand.read_text().splitlines():
        m = re.match(r"^- \*\*(.+?)\*\*", l.strip())
        if m and _norm(m.group(1)) in tested:
            dropped += 1
            continue
        kept.append(l)
    cand.write_text("\n".join(kept) + "\n")
    return dropped


def archive_strategies(days: int = 30):
    """Move strategy modules older than N days into strategies/archive/ (the run record lives in
    run_log.jsonl + the wiki; old generated modules are only needed for forensics)."""
    import time
    sdir = Path("/root/hephaestus/strategies")
    adir = sdir / "archive"
    cutoff = time.time() - days * 86400
    moved = 0
    for p in sdir.glob("*.py"):
        if p.name == "__init__.py":
            continue
        if p.stat().st_mtime < cutoff:
            adir.mkdir(exist_ok=True)
            p.rename(adir / p.name)
            moved += 1
    return moved


def prune_forge_logs(days: int = 30):
    """Delete dated forge logs older than N days (logs/forge-YYYY-MM-DD.log)."""
    import time
    ldir = Path("/root/hephaestus/logs")
    if not ldir.exists():
        return 0
    cutoff = time.time() - days * 86400
    n = 0
    for p in ldir.glob("forge-*.log"):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            n += 1
    return n


def orphans():
    """Pages with no inbound [[wikilink]] (excluding spine pages)."""
    pages = {p.stem for p in WIKI.rglob("*.md")}
    text = " ".join(p.read_text(errors="replace") for p in WIKI.rglob("*.md"))
    linked = set(re.findall(r"\[\[([^\]|/]+)", text))
    spine = {"index", "log", "overview", "AGENTS", "log-archive", "DATA_CATALOG"}
    return sorted(p for p in pages if p not in linked and p not in spine)[:15]


def lint():
    a, d = cap_log(), prune_candidates()
    n_arch, n_logs = archive_strategies(), prune_forge_logs()
    nexp = len(list((WIKI / "experiments").glob("*.md")))
    reg = WIKI / ".registry" / "hypothesis_registry.jsonl"
    nreg = len(reg.read_text().splitlines()) if reg.exists() else 0
    elite = WIKI / ".elite" / "pool.jsonl"
    nelite = len(elite.read_text().splitlines()) if elite.exists() else 0
    npages = len(list(WIKI.rglob("*.md")))
    orph = orphans()
    print(f"[lint] {datetime.now():%Y-%m-%d}: archived {a} log entries | pruned {d} tested candidates | "
          f"archived {n_arch} old strategy modules | pruned {n_logs} old forge logs")
    print(f"[lint] health: {nexp} experiments | {nreg} registry rows | {npages} pages | {nelite} elite | orphans {orph}")
    with (WIKI / "log.md").open("a") as f:
        f.write(f"\n## [{datetime.now():%Y-%m-%d}] lint | archived {a} log, pruned {d} candidates | "
                f"{nexp} exp, {nreg} reg, {nelite} elite, {len(orph)} orphans")
    return {"archived": a, "pruned": d, "experiments": nexp, "registry": nreg, "elite": nelite, "orphans": orph}


if __name__ == "__main__":
    lint()
