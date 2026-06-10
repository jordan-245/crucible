"""Scout: the 'ingest external sources' step. Searches the web for NEW edges/strategies based on
wiki gaps, distills findings into the wiki, and queues testable candidates for the propose step.
This makes generation OPEN (discovers what others run) instead of only recombining what we know.
Pattern: karpathy LLM-Wiki 'ingest' + a research scout. LLM via the pi CLI + brave-search."""
import json
from datetime import date
from pathlib import Path
from agent.llm import call as _llm_call, extract_json

from crucible_paths import WIKI  # central config


def _read(p):
    f = WIKI / p
    return f.read_text() if f.exists() else ""


def _pi(prompt):
    return _llm_call(prompt, timeout=480)


def _brave(query, n=5):
    try:
        from agent.brave import rich_search_text
        return rich_search_text(query)
    except Exception as e:
        return f"(search failed: {e})"


def _json(text):
    for op, cl in (("{", "}"), ("[", "]")):
        obj = extract_json(text, op, cl)
        if obj is not None:
            return obj
    return None


def scout(n_queries=4):
    ctx = ("=== OVERVIEW ===\n" + _read("overview.md") +
           "\n\n=== ANTI-PATTERNS (avoid) ===\n" + _read("patterns/META-LESSONS.md")[:2500] +
           "\n\n=== DATA WE OWN (prefer ideas buildable on these) ===\n" + _read("DATA_CATALOG.md") +
           "\n\n=== ALREADY TESTED / SURFACED ===\n" + _read("index.md"))
    # 1. generate targeted search queries aimed at wiki gaps / untested promising directions
    q_raw = _pi(f"{ctx}\n\nYou are a quant research scout. Based on the wiki's UNTESTED promising "
                f"directions and gaps, produce {n_queries} web-search queries that would surface NEW, "
                f"specific, backtestable strategies/edges that real practitioners or recent research "
                f"actually use (risk premia, structural edges, combinations) — NOT generic. "
                f"DIVERSIFY HARD: each query a DISTINCT premium AND market — span e.g. one equity "
                f"factor/event, one rates/credit, one volatility, one cross-asset/commodity/FX. Do NOT "
                f"cluster (the wiki is ALREADY heavy on crypto funding-carry and PEAD — deliberately look "
                f"ELSEWHERE). Prefer edges buildable on the OWNED data above. Avoid anything tested/closed. "
                f"Return ONLY a JSON array of query strings.")
    queries = _json(q_raw) or ["systematic risk premia retail backtest 2025 carry trend vol",
                               "crypto delta neutral funding basis strategy 2025"]
    queries = queries[:n_queries]
    # 2. search
    results = "\n\n".join(f"### QUERY: {q}\n{_brave(q)}" for q in queries)
    # 3. distill into structured findings + testable candidates (with sources), flag contradictions
    d_raw = _pi(f"{ctx}\n\n=== WEB SEARCH RESULTS ===\n{results}\n\n"
                f"Distill these into NEW knowledge for the wiki. Return ONLY JSON:\n"
                f'{{"summary": "2-3 sentences on what is new/relevant", '
                f'"candidates": [{{"title":"...","premium":"...","market":"...","why_promising":"...",'
                f'"data_feasible":"free/owned?","not_already_tested":"...","source":"..."}}], '
                f'"premia_updates": ["short factual updates to add to premia/market pages, with source"], '
                f'"contradictions": ["any finding that contradicts a wiki claim"]}}\n'
                f'Surface DIVERSE candidates across DIFFERENT premia/markets (NOT multiple variants of one '
                f'theme); PREFER ideas buildable on the OWNED data above (mark data_feasible accordingly).')
    findings = _json(d_raw) or {"summary": d_raw[:600], "candidates": [], "premia_updates": [], "contradictions": []}
    _ingest(queries, findings)
    return findings


def _ingest(queries, findings):
    """Write the distilled web findings into the wiki (sources page + candidates queue + log)."""
    today = date.today()
    src = WIKI / "sources" / f"{today}-scout.md"
    src.parent.mkdir(exist_ok=True)
    cand_lines = "\n".join(
        f"- **{c.get('title')}** ({c.get('premium')}, {c.get('market')}) — {c.get('why_promising')} "
        f"[data: {c.get('data_feasible')}] src: {c.get('source')}" for c in findings.get("candidates", []))
    src.write_text(f"""---
type: scout-ingest
date: {today}
queries: {queries}
---
# Web Scout — {today}

## Summary
{findings.get('summary','')}

## New testable candidates (queued for generation)
{cand_lines or '(none surfaced)'}

## Premia/market updates
{chr(10).join('- ' + u for u in findings.get('premia_updates', [])) or '(none)'}

## Contradictions with wiki claims (review)
{chr(10).join('- ⚠️ ' + c for c in findings.get('contradictions', [])) or '(none)'}
""")
    # candidates queue that propose reads
    cq = WIKI / "candidates.md"
    with open(cq, "a") as f:
        f.write(f"\n## [{today}] web-scout candidates\n{cand_lines}\n")
    with open(WIKI / "log.md", "a") as f:
        f.write(f"\n## [{today}] ingest | web-scout: {len(findings.get('candidates',[]))} candidates, "
                f"{len(findings.get('contradictions',[]))} contradictions -> sources/{today}-scout.md")
    print(f"[scout] ingested {len(findings.get('candidates',[]))} candidates, "
          f"{len(findings.get('contradictions',[]))} contradictions -> wiki")


if __name__ == "__main__":
    f = scout()
    print(json.dumps(f, indent=2)[:1800])
