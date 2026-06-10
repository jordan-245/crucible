"""Bootstrap a minimal research wiki so Crucible runs on a fresh machine.

Usage:
    python3 scripts/bootstrap_wiki.py [path]      # default: $CRUCIBLE_WIKI or ~/research-wiki

Creates the directory skeleton + seed files the agents read (overview, DATA_CATALOG,
META-LESSONS, queue, FDR registry). Idempotent: never overwrites existing files.
Make it a git repo afterwards if you want history (recommended):  cd <wiki> && git init && git add -A && git commit -m init
"""
import json
import os
import sys
from pathlib import Path

SEED = {
    "overview.md": """# Research Wiki — overview

The shared memory of the autonomous research system. Every experiment, verdict, pattern,
and closed decision lives here. Agents READ this before generating (so they never re-test
closed ideas) and WRITE every outcome back (so knowledge compounds).

## Current state
Fresh wiki — no experiments yet.
""",
    "DATA_CATALOG.md": """# Data catalog — what we own / can use

Hypotheses must be buildable on the sources listed here (Gate-0). Anything else is
DATA-GATED and fails feasibility. Edit this file to match YOUR data reality.

| Source | Access | Coverage | Adapter |
|---|---|---|---|
| yfinance | free, no key | futures/ETFs/indices daily OHLCV | `sdk.adapters.yf_panel` |
| FRED | free API key | rates/yields/credit spreads/macro | `sdk.adapters.fred_series` |
| Sharadar SEP/SF1/TICKERS (optional) | paid (Nasdaq Data Link) | survivorship-clean US equities + point-in-time fundamentals | `sdk.adapters.sep_panel/us_universe/sf1` — needs `CRUCIBLE_DATA/sharadar/{SEP.zip,SF1.zip,SHARADAR_TICKERS_*.csv}` |
""",
    "patterns/META-LESSONS.md": """# Meta-lessons — confirmed patterns + anti-patterns

Agents MUST read this before generating. Append a lesson every time a gate kills
something for a new reason.

## Anti-patterns (seed set)
1. **Long-only "edges" are usually universe beta.** Benchmark long books against the
   equal-weight universe, not zero (the beta-confound gate enforces this).
2. **Breadth replication does not validate a construction.** Construction artifacts
   (vol-targeted noise sorts, bid-ask-bounce harvesting) replicate on EVERY universe;
   only a permutation test (MCPT) catches them. Perm mean >= real Sharpe is the
   strongest red flag there is.
3. **Never tune to rescue a failed pre-registration.** The frozen design IS the
   experiment; the grid exists only to make the DSR search-burden honest.
""",
    "index.md": "# Index\n\n- [[overview]]\n- [[DATA_CATALOG]]\n- [[patterns/META-LESSONS]]\n",
    "log.md": "# Run log\n",
    "candidates.md": "# Candidates awaiting confirmation\n",
    "decisions/CLOSED.md": "# Closed decisions — do NOT re-test on the same data\n",
    "decisions/closed-families.txt": "",
}

DIRS = ["experiments", "patterns", "decisions", "methodology",
        ".queue", ".locks", ".registry", ".elite"]


def main():
    wiki = Path(sys.argv[1] if len(sys.argv) > 1
                else os.environ.get("CRUCIBLE_WIKI", os.path.expanduser("~/research-wiki")))
    wiki.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        (wiki / d).mkdir(parents=True, exist_ok=True)
    for rel, content in SEED.items():
        f = wiki / rel
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
            print(f"  created {rel}")
    for rel in [".queue/queue.jsonl", ".registry/hypothesis_registry.jsonl", ".elite/pool.jsonl"]:
        f = wiki / rel
        if not f.exists():
            f.write_text("")
            print(f"  created {rel}")
    print(f"\nWiki ready at {wiki}")
    print(f"Set:  export CRUCIBLE_WIKI={wiki}")


if __name__ == "__main__":
    main()
