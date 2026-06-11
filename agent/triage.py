"""Runtime-error self-triage loop (LOOPS_FRAMEWORK_PLAN 2.1).

Nightly after the forge: collect fail_reason=runtime_error/worker_exception rows,
classify each traceback (SDK/infrastructure vs strategy-code), and for SDK-implicated
failures ask a debugger LLM for a root-cause diagnosis + minimal unified diff.

RAILS (non-negotiable):
  - NEVER touches master. Patches apply in a throwaway git worktree on branch
    triage/<id>; the branch survives only if the FULL test suite passes there.
  - NEVER auto-merges. Output = diagnosis + branch name in the morning report
    (notice, not a phone buzz). Human merges; SDK feeds frozen designs, so the
    byte-exactness rules of any SDK change apply at merge review.
  - Budget: max 3 triages/night, one LLM call each (+1 if the diff fails to apply).
  - State: logs/triage_state.json records processed (ts, id) — a row is triaged once.

Re-enqueue after merge:  python3 -m agent.triage --requeue <queue_id>
Usage (loop):            python3 -m agent.triage
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crucible_paths import ROOT, WIKI  # noqa: E402

RUN_LOG = ROOT / "agent" / "run_log.jsonl"
STATE = ROOT / "logs" / "triage_state.json"
TRIAGE_REASONS = ("runtime_error", "worker_exception")
MAX_PER_NIGHT = 3
MAX_SRC_CHARS = 12_000  # per file fed to the LLM

PROMPT = """You are a senior quant-infrastructure debugger. A nightly research worker \
crashed after exhausting its fix-retries. Diagnose the ROOT CAUSE and, only if the bug \
is in the SHARED SDK (not the auto-generated strategy), produce a minimal fix.

## Failure
hypothesis: {title}
fail_reason: {fail_reason}

## Traceback tail
{log_tail}

## Auto-generated strategy module ({module_name}) — context, often NOT the bug
{module_src}

## SDK source implicated by the traceback
{sdk_src}

## Rules
- The SDK feeds FROZEN experiment designs: a fix must not change behavior for any \
existing caller — additive/fallback fixes only, no signature changes, no renames.
- If the root cause is in the STRATEGY code (the generator's own bug), say so — \
diagnosis only, no diff. The strategy is disposable; the SDK is not.
- If the root cause is missing DATA or environment (not code), say so — no diff.

## Output (exactly this JSON, nothing else)
{{"root_cause": "<2-3 sentences>",
 "location": "sdk" | "strategy" | "data" | "environment",
 "fix_summary": "<1-2 sentences, or null>",
 "diff": "<unified diff against the given SDK file(s), repo-root-relative paths, or null>"}}"""


def _state() -> set:
    if STATE.exists():
        try:
            return {tuple(x) for x in json.loads(STATE.read_text())}
        except (ValueError, TypeError):
            pass
    return set()


def _save_state(done: set) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(sorted(done)))


def _sdk_files(log_tail: str) -> list[Path]:
    """SDK/agent files named in the traceback (the repo's own code, not strategies/)."""
    out = []
    for m in re.finditer(r'File "(/root/crucible/(?:sdk|agent)/[^"]+\.py)"', log_tail or ""):
        p = Path(m.group(1))
        if p.exists() and p not in out:
            out.append(p)
    return out


def _module_file(rec: dict) -> Path | None:
    rid = rec.get("id", "")
    cand = sorted(ROOT.glob(f"strategies/*{rid.split('-')[-1]}*.py"))
    if cand:
        return cand[0]
    m = re.search(r'File "(/root/crucible/strategies/[^"]+\.py)"', rec.get("log_tail") or "")
    return Path(m.group(1)) if m and Path(m.group(1)).exists() else None


def _src(p: Path | None) -> str:
    if not p or not p.exists():
        return "(unavailable)"
    t = p.read_text(errors="replace")
    return t[:MAX_SRC_CHARS] + ("\n…(truncated)" if len(t) > MAX_SRC_CHARS else "")


def _test_in_worktree(branch: str, diff: str) -> tuple[bool, str]:
    """Apply diff on a fresh worktree branch; run the full suite there. Master untouched.
    Structural cleanup (not caller courtesy — cf. canary isolation lesson): on ANY
    failure path the branch is deleted here; only a green branch survives."""
    wt = Path(f"/tmp/triage-wt-{branch.replace('/', '-')}")
    ok = False
    try:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=ROOT, capture_output=True)
        subprocess.run(["git", "branch", "-D", branch], cwd=ROOT, capture_output=True)
        r = subprocess.run(["git", "worktree", "add", "-b", branch, str(wt)],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False, f"worktree add failed: {r.stderr[:150]}"
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=diff,
                           cwd=wt, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False, f"diff does not apply: {r.stderr[:150]}"
        r = subprocess.run(["python3", "-m", "pytest", "tests/", "-q", "-x"],
                           cwd=wt, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            tail = (r.stdout or r.stderr).splitlines()[-3:]
            return False, "tests FAILED on patch: " + " | ".join(tail)
        subprocess.run(["git", "add", "-A"], cwd=wt, capture_output=True)
        subprocess.run(["git", "-c", "user.name=triage", "-c", "user.email=triage@local",
                        "commit", "-q", "-m", f"triage candidate fix ({branch})"],
                       cwd=wt, capture_output=True, timeout=60)
        ok = True
        return True, "tests green on branch"
    except subprocess.TimeoutExpired:
        return False, "worktree/test timeout"
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=ROOT, capture_output=True)
        if not ok:
            subprocess.run(["git", "branch", "-D", branch], cwd=ROOT, capture_output=True)


def triage_one(rec: dict) -> dict:
    sdk_files = _sdk_files(rec.get("log_tail") or "")
    sdk_src = "\n\n".join(f"### {p.relative_to(ROOT)}\n{_src(p)}" for p in sdk_files) \
              or "(no SDK file in traceback — likely strategy/data issue)"
    mod = _module_file(rec)
    prompt = PROMPT.format(title=rec.get("title", "?"), fail_reason=rec.get("fail_reason"),
                           log_tail=rec.get("log_tail") or "(none)",
                           module_name=(mod.name if mod else "?"), module_src=_src(mod),
                           sdk_src=sdk_src)
    from agent.llm import call, extract_json
    ans = extract_json(call(prompt, timeout=600)) or {}
    result = {"id": rec.get("id"), "queue_id": rec.get("queue_id"),
              "title": rec.get("title"), "ts": rec.get("ts"),
              "root_cause": ans.get("root_cause", "(no diagnosis returned)"),
              "location": ans.get("location", "?"),
              "fix_summary": ans.get("fix_summary"), "branch": None, "branch_status": None}
    diff = ans.get("diff")
    if result["location"] == "sdk" and diff:
        branch = f"triage/{(rec.get('id') or 'unknown')[:40]}"
        ok, note = _test_in_worktree(branch, diff)
        result["branch"], result["branch_status"] = (branch if ok else None), note
    return result


def _wiki_note(results: list) -> None:
    page = WIKI / "triage.md"
    if not page.exists():
        page.write_text("# Nightly runtime-error triage\n\nDrafted fixes live on triage/* "
                        "branches — NEVER auto-merged. Merge review must apply the SDK "
                        "byte-exactness rules; after merge, re-enqueue the casualty with "
                        "`python3 -m agent.triage --requeue <queue_id>`.\n")
    with page.open("a") as f:
        for r in results:
            f.write(f"\n## [{datetime.now():%Y-%m-%d}] {r['title'][:70]}\n"
                    f"- cause ({r['location']}): {r['root_cause']}\n"
                    + (f"- fix: {r['fix_summary']}\n" if r.get("fix_summary") else "")
                    + (f"- branch: `{r['branch']}` ({r['branch_status']})\n" if r.get("branch")
                       else (f"- patch attempt: {r['branch_status']}\n" if r.get("branch_status") else ""))
                    + f"- requeue after merge: `python3 -m agent.triage --requeue {r['queue_id']}`\n")


def requeue(queue_id: str) -> int:
    rows = [json.loads(l) for l in RUN_LOG.read_text().splitlines() if l.strip()]
    match = [r for r in rows if r.get("queue_id") == queue_id]
    if not match:
        print(f"[triage] no run-log row with queue_id {queue_id}")
        return 1
    from sdk import queue
    new_id = queue.enqueue(match[-1]["proposal"])
    print(f"[triage] re-enqueued '{match[-1].get('title', '?')[:60]}' as {new_id}")
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--requeue":
        return requeue(sys.argv[2])
    rows = [json.loads(l) for l in RUN_LOG.read_text().splitlines() if l.strip()] \
        if RUN_LOG.exists() else []
    done = _state()
    todo = [r for r in rows if r.get("fail_reason") in TRIAGE_REASONS
            and (r.get("ts"), r.get("id")) not in done][-MAX_PER_NIGHT:]
    if not todo:
        print("[triage] no new runtime failures — nothing to do")
        return 0
    results = []
    for rec in todo:
        print(f"[triage] diagnosing: {rec.get('title', '?')[:70]}")
        try:
            results.append(triage_one(rec))
        except Exception as e:
            results.append({"id": rec.get("id"), "queue_id": rec.get("queue_id"),
                            "title": rec.get("title"), "ts": rec.get("ts"),
                            "root_cause": f"triage crashed: {type(e).__name__}: {str(e)[:120]}",
                            "location": "?", "fix_summary": None, "branch": None,
                            "branch_status": None})
        done.add((rec.get("ts"), rec.get("id")))
    _save_state(done)
    _wiki_note(results)
    try:
        from sdk.notify import notice
        for r in results:
            line = (f"🩺 triage [{r['location']}] {str(r['title'])[:60]}: {r['root_cause'][:140]}"
                    + (f" → branch {r['branch']} (tests green, awaiting your merge)" if r["branch"] else ""))
            notice(line, source="triage")
            print("[triage]", line)
    except Exception as e:
        print(f"[triage] notice failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
