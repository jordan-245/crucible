"""Generation step: read the shared wiki, propose ONE new untested hypothesis (Claude Max OAuth, $0).
Grounded in accumulated knowledge so it never re-tests closed sets or duplicates experiments."""
import json, subprocess
from pathlib import Path
from agent.config import pi_cmd

WIKI = Path("/root/research-wiki")
SYS = "You are Claude Code, Anthropic's official CLI for Claude."


def _read(p):
    f = WIKI / p
    return f.read_text() if f.exists() else ""


def propose() -> dict:
    context = (
        "=== OVERVIEW ===\n" + _read("overview.md") +
        "\n\n=== PATTERNS & ANTI-PATTERNS (obey these) ===\n" + _read("patterns/META-LESSONS.md") +
        "\n\n=== CLOSED DECISIONS (never re-open) ===\n" + _read("decisions/CLOSED.md") +
        "\n\n=== EXISTING EXPERIMENTS (do not duplicate) ===\n" + _read("index.md") +
        "\n\n=== DATA WE OWN / CAN USE (build ONLY on these; anything else is DATA-GATED -> Gate-0 FAIL) ===\n" + _read("DATA_CATALOG.md") +
        "\n\n=== WEB-SCOUTED CANDIDATES (fresh external ideas — prefer a strong one of these) ===\n" + _read("candidates.md")
    )
    prompt = f"""{context}

You are a quant research agent. Propose EXACTLY ONE new, untested strategy hypothesis to test next.
HARD CONSTRAINTS (from the wiki above):
- Must be a RISK PREMIUM or a COMBINATION of complementary premia — NOT a standalone prediction edge in a liquid market.
- Must NOT duplicate any existing experiment and must NOT violate any anti-pattern or closed decision.
- Must be data-FEASIBLE on the OWNED/FREE data in the DATA CATALOG above (for US equities PREFER survivorship-clean Sharadar SEP/SF1 via sep_panel/us_universe/sf1, NOT yfinance). If it would need a DATA-GATED source, set prior=low and state exactly what's missing in gate0_data_check.
- Prefer: combinations of validated legs, complementary premia (opposite tails), or less-efficient corners.
- DIVERSIFY — do NOT propose yet another variant of a premium that already appears 2+ times in the experiments/queue above. If one theme (e.g. PEAD/SUE) is already well-represented, pick a DIFFERENT premium or market entirely.
- CROWDING/DECAY — PREFER novel inefficiencies in less-arbitraged corners over heavily-published factors. Famous factors (betting-against-beta/low-vol, value, momentum, size) are CROWDED -> decayed + regime-fragile (cf. BAB: passed every single-universe gate but FAILED cross-market and is heavily published). If you propose a known premium, it must be a genuinely under-exploited IMPLEMENTATION or corner.
- DEPLOYABILITY (board 2026-06-09 — avoid STRANDED ALPHA) — a pass is only valuable if it can be TRADED at retail scale (~$5K, IB/Alpaca, no special borrow). STRONGLY PREFER: long-only or long-tilt equity books in liquid-enough names, futures-implementable premia (IB micro-futures: equity index, rates, FX, metals, energy), ETF-implementable cross-asset books, crypto perps (small size). AVOID proposing strategies whose construction REQUIRES: shorting illiquid/hard-to-borrow names (micro-cap short legs are NOT borrowable at retail), gross leverage >2x, intraday execution, or options/OTC structures we cannot route. A long/short CROSS-SECTIONAL design is acceptable ONLY if the short leg is liquid large/mid-caps or an index hedge (e.g. short SPY/sector ETF vs the long book). If the cleanest test of the premium is long-short but a deployable long-only/index-hedged variant exists, PROPOSE THE DEPLOYABLE VARIANT.
Return ONLY a JSON object:
{{"title": "...", "premium": "...", "market": "...", "data_source": "...", "free_or_owned": "...",
"signal_approach": "one-paragraph frozen construction", "why_not_duplicate": "...", "prior": "low|medium|high",
"pairs_with": "...", "gate0_data_check": "what to verify before building",
"crowding_risk": "low|medium|high — how heavily-published/arbitraged is this edge? (favour low)",
"retail_tradable_5k": "yes|no — can THIS construction be executed at ~$5K via IB/Alpaca (instruments routable, short leg borrowable or index-hedged, no >2x gross leverage)? If no, this proposal will be DOWN-RANKED — prefer redesigning it to a deployable variant",
"scope": "broad|local — broad if a UNIVERSAL mechanism (theory says it appears across markets; a pass must later GENERALISE) or local if defensibly universe-specific (then forward-validation confirms it)",
"generalization_plan": "if broad: the untouched universes to confirm the mechanism in (e.g. other cap-tiers/sectors/asset-classes); if local: the economic reason it lives ONLY in this universe + the forward-validation plan"}}"""
    r = subprocess.run(pi_cmd(), input=prompt, capture_output=True, text=True, timeout=300)
    text = _assistant_text(r.stdout)
    try:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1])
    except Exception:
        return {"raw": text[:1500] or r.stdout[:800], "error": "parse_failed", "stderr": r.stderr[:300]}


def mutate(elite: dict) -> dict:
    """EVOLVE an elite near-miss: ONE targeted change to make it more robust / generalise better — keep what
    worked, fix what's fragile. NOT a fresh idea. Same JSON proposal format. (Evolutionary exploit step.)"""
    ctx = ("=== ANTI-PATTERNS (obey) ===\n" + _read("patterns/META-LESSONS.md")[:2000] +
           "\n\n=== DATA WE OWN ===\n" + _read("DATA_CATALOG.md")[:1500])
    prompt = f"""{ctx}

ELITE STRATEGY to EVOLVE (fitness/DSR {elite.get('fitness')}):
{json.dumps(elite.get('proposal'), indent=2)}

Propose ONE targeted MUTATION to make it MORE ROBUST or GENERALISE better — e.g. a different/broader universe,
a complementary leg, a cleaner/lower-turnover construction, a regime filter, a cost-hardening. Keep what worked;
fix the fragile part. DEPLOYABILITY: the mutated construction must stay retail-tradable at ~$5K (IB/Alpaca;
no illiquid short legs, no >2x gross leverage) — mutating TOWARD deployability (e.g. long-only/index-hedged
variant) is itself a high-value mutation. This is an EVOLUTION of THIS strategy, NOT a new idea, and must still
obey the anti-patterns and use owned/free data. Return ONLY the SAME JSON proposal object (title, premium, market, data_source,
free_or_owned, signal_approach, why_not_duplicate, prior, pairs_with, gate0_data_check, crowding_risk, scope,
generalization_plan) with the mutation applied (title should note it's a variant)."""
    r = subprocess.run(pi_cmd(), input=prompt, capture_output=True, text=True, timeout=300)
    text = _assistant_text(r.stdout)
    try:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1])
    except Exception:
        return {"error": "mutate_parse_failed"}


def _assistant_text(stream: str) -> str:
    """Return the FULL assistant message. pi streams cumulative snapshots, so return the
    longest single assistant-text candidate (the final complete message), not a concat."""
    parts = []
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        d = ev.get("delta") or {}
        if d.get("text"):
            parts.append(d["text"])
        msg = ev.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                    parts.append(c["text"])
        for k in ("text", "content"):
            if ev.get("type") in ("agent_message", "assistant_message") and isinstance(ev.get(k), str):
                parts.append(ev[k])
    return max(parts, key=len) if parts else ""


if __name__ == "__main__":
    p = propose()
    print(json.dumps(p, indent=2)[:2000])
