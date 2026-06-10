"""Generation step: read the shared wiki, propose ONE new untested hypothesis (LLM via the pi CLI).
Grounded in accumulated knowledge so it never re-tests closed sets or duplicates experiments."""
import json  # noqa: F401 (used by prompt formatting)
from pathlib import Path
from agent.llm import call as _llm_call, assistant_text as _assistant_text, extract_json  # noqa: F401 (re-export for legacy importers)

from crucible_paths import WIKI  # central config
SYS = "You are Claude Code, Anthropic's official CLI for Claude."


def _read(p):
    f = WIKI / p
    return f.read_text() if f.exists() else ""


def _read_tail(p, max_chars: int):
    """E5: growing files (index, candidates) are read NEWEST-LAST and capped — prompt size must
    not rise monotonically with project history (cost, latency, 420s-timeout risk; the director
    makes up to 12 propose calls per top-up). Curated files (overview, lessons, closed, catalog)
    stay full: they are pruned by the weekly lint, not by truncation — dropping an anti-pattern
    or a closed decision from the prompt would un-learn it."""
    t = _read(p)
    if len(t) <= max_chars:
        return t
    cut = t[-max_chars:]
    nl = cut.find("\n")  # start at a whole line
    return f"[... older entries omitted ({len(t) - max_chars} chars) ...]\n" + (cut[nl + 1:] if nl >= 0 else cut)


def propose() -> dict:
    context = (
        "=== OVERVIEW ===\n" + _read("overview.md") +
        "\n\n=== PATTERNS & ANTI-PATTERNS (obey these) ===\n" + _read("patterns/META-LESSONS.md") +
        "\n\n=== CLOSED DECISIONS (never re-open) ===\n" + _read("decisions/CLOSED.md") +
        "\n\n=== EXISTING EXPERIMENTS (do not duplicate; newest last, older omitted) ===\n" + _read_tail("index.md", 12_000) +
        "\n\n=== DATA WE OWN / CAN USE (build ONLY on these; anything else is DATA-GATED -> Gate-0 FAIL) ===\n" + _read("DATA_CATALOG.md") +
        "\n\n=== WEB-SCOUTED CANDIDATES (fresh external ideas — prefer a strong one of these) ===\n" + _read_tail("candidates.md", 8_000)
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
    # 420s: Fable-5 measured 285s on a real propose — 300s left only ~15s headroom (2026-06-10 smoke test)
    text = _llm_call(prompt)
    obj = extract_json(text)
    return obj if obj is not None else {"raw": text[:1500], "error": "parse_failed"}


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
    obj = extract_json(_llm_call(prompt))
    return obj if obj is not None else {"error": "mutate_parse_failed"}


