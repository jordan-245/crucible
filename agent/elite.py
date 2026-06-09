"""Elite pool — the forge's EVOLUTIONARY memory. Keeps the top-K strategies by fitness (DSR); the director
MUTATES these (exploit) alongside fresh proposals (explore), so good constructions compound instead of being
re-discovered each night. Succinct by design: one jsonl, fitness=DSR, fitness-weighted sampling."""
import json
from pathlib import Path

POOL = Path("/root/research-wiki/.elite/pool.jsonl")
CLOSED_FAMILIES = Path("/root/research-wiki/decisions/closed-families.txt")
K = 12          # pool size
MIN_FIT = 0.5   # only genuinely promising runs enter (DSR > 0.5)


def _closed_families() -> set:
    """Family buckets CLOSED by decision (cf. decisions/CLOSED.md). Elites in these families are
    never sampled for mutation and never (re-)recorded — the exploit loop must not keep evolving
    a falsified premium (the value×mom-hammering failure mode, closed 2026-06-10)."""
    if not CLOSED_FAMILIES.exists():
        return set()
    return {l.strip() for l in CLOSED_FAMILIES.read_text().splitlines()
            if l.strip() and not l.startswith("#")}


def _family(item: dict) -> str:
    from agent.families import family_bucket
    return family_bucket((item.get("title") or "") or (item.get("proposal") or {}).get("premium", ""))


def _fitness(v: dict) -> float:
    if not v:
        return 0.0
    if v.get("beta_confound"):
        return 0.0   # long-only-beta confound -> never seed the evolutionary exploit loop with it
    if v.get("dsr") is not None:
        return float(v["dsr"])                 # the deflated, multiple-testing-aware Sharpe = the natural fitness
    s, h = v.get("search_sharpe"), v.get("holdout_sharpe")
    return float(min(s, h)) if (s and h and s > 0 and h > 0) else 0.0  # fallback: search/holdout consistency


def _load() -> list:
    return [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()] if POOL.exists() else []


def record(outcome: dict) -> None:
    fit = _fitness(outcome.get("verdict"))
    if fit <= MIN_FIT:
        return
    if _family(outcome) in _closed_families():
        return  # falsified family — do not seed the evolutionary loop with it
    items = _load()
    items.append({"id": outcome.get("id"), "fitness": round(fit, 4), "title": outcome.get("title"),
                  "proposal": outcome.get("proposal"), "ts": outcome.get("ts")})
    items.sort(key=lambda x: x["fitness"], reverse=True)
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text("".join(json.dumps(i) + "\n" for i in items[:K]))


def sample(rng) -> dict | None:
    """Fitness-weighted pick of an elite to evolve, DOWN-WEIGHTED by family representation so the exploit
    branch can't over-concentrate on the highest-fitness family (the value×mom-hammering failure mode)."""
    closed = _closed_families()
    items = [i for i in _load() if _family(i) not in closed]
    if not items:
        return None
    from collections import Counter
    fams = [_family(i) for i in items]
    fc = Counter(fams)
    w = [max(i["fitness"], 0.01) / fc[f] for i, f in zip(items, fams)]  # diversity-adjusted: /count of its family
    r = rng.random() * sum(w)
    c = 0.0
    for it, wi in zip(items, w):
        c += wi
        if r <= c:
            return it
    return items[0]


def top(k: int = K) -> list:
    return _load()[:k]
