"""Lifecycle states + decay/retirement rule (SYNTHESIS_PLAN Stage 3).

Implements the FROZEN pre-registration (research-wiki/methodology/prereg-retirement-rule.md,
2026-06-12) exactly as written. Owned by the weekly evidence loop (forward/evidence.py calls
evaluate_lifecycle per book). 'retired' is human-only via:

    python3 -m forward.lifecycle retire <book>     # human-confirmed exit (never auto)
    python3 -m forward.lifecycle status            # show lifecycle of every deployed book

The rule (frozen — do not tune):
  D1: rolling-60-trading-day realized mean < 0.25 x modeled daily_mean on 2 CONSECUTIVE weekly
      evaluations (each needing >=60 obs; modeled mean must be > 0 else not_evaluable).
  D2: one-sided CUSUM over the full history: z_t=(modeled_mean - r_t)/modeled_std,
      S_t=max(0, S_{t-1} + z_t - K_ALLOWANCE); fires when S exceeds H_THRESHOLD.
  decaying iff D1 AND D2 -> Telegram critical, human decides. D1 XOR D2 -> watch annotation only.
No auto-liquidation in any path. Stage-D allocator contract: allocate only over
evidence/real_capital_candidate books.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# frozen pre-reg parameters (2026-06-12) — changing any of these requires a new pre-registration
ROLL_DAYS = 60
LEVEL_FRAC = 0.25
CONSECUTIVE_NEEDED = 2
K_ALLOWANCE = 0.25
H_THRESHOLD = 5.0
EVIDENCE_MIN_DAYS = 20   # shadow -> evidence (matches the G2 go-live-gate floor)

LIVE = Path("/root/atlas/data/live")
ATLAS = Path("/root/atlas")
# D1 consecutive-evaluation memory (crucible-side state; registry holds only the resulting lifecycle)
STATE_FILE = Path(__file__).resolve().parent / "lifecycle_state.json"


def _registry():
    sys.path.insert(0, str(ATLAS))
    from atlas.execution import registry
    return registry


def _returns(book: str) -> list[float]:
    f = LIVE / book / "returns.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [float(r["ret"]) for r in rows if r.get("ret") is not None]


def decay_check(rets: list[float], expectation: dict) -> dict:
    """Pure rule evaluation (one snapshot — D1's consecutive-week memory lives in the caller).
    Returns {d1, d2, cusum_peak, roll_mean, modeled_mean, evaluable, note}."""
    out = {"d1": None, "d2": None, "cusum_peak": None, "roll_mean": None,
           "modeled_mean": expectation.get("daily_mean"), "evaluable": False, "note": None}
    mu, sd = expectation.get("daily_mean"), expectation.get("daily_std")
    if not mu or not sd or mu <= 0 or sd <= 0:
        out["note"] = "not_evaluable (modeled mean/std missing or <= 0)"
        return out
    if len(rets) < ROLL_DAYS:
        out["note"] = f"not_evaluable ({len(rets)} obs < {ROLL_DAYS})"
        return out
    out["evaluable"] = True
    roll = sum(rets[-ROLL_DAYS:]) / ROLL_DAYS
    out["roll_mean"] = roll
    out["d1"] = bool(roll < LEVEL_FRAC * mu)
    s = peak = 0.0
    for r in rets:                      # full history, deterministic (no carried state)
        s = max(0.0, s + (mu - r) / sd - K_ALLOWANCE)
        peak = max(peak, s)
    out["cusum_peak"] = round(peak, 2)
    out["d2"] = bool(s > H_THRESHOLD)   # current S, not the peak: a recovered book un-fires
    return out


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def evaluate_lifecycle(book: str, gates_all_pass: bool, n_days: int) -> dict:
    """Weekly transition evaluation for one book. Returns {lifecycle, decay, changed, watch}."""
    reg = _registry()
    strat = next((s for s in reg.load() if s.name == book), None)
    if strat is None:
        return {"lifecycle": None, "decay": None, "changed": False,
                "watch": f"{book} not in registry"}
    cur = getattr(strat, "lifecycle", "shadow") or "shadow"
    if cur == "retired":                # terminal, human-only — never touched here
        return {"lifecycle": "retired", "decay": None, "changed": False, "watch": None}

    decay = decay_check(_returns(book), strat.expectation or {})
    state = _load_state()
    hist = state.get(book, {"d1_streak": 0})
    if decay["evaluable"]:
        hist["d1_streak"] = hist.get("d1_streak", 0) + 1 if decay["d1"] else 0
    state[book] = {**hist, "last_eval": datetime.now().isoformat(timespec="seconds"),
                   "last_decay": {k: decay[k] for k in ("d1", "d2", "cusum_peak", "roll_mean")}}
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    d1_confirmed = hist["d1_streak"] >= CONSECUTIVE_NEEDED
    watch = None
    if d1_confirmed and decay["d2"]:
        new = "decaying"
    elif decay["evaluable"] and (d1_confirmed != bool(decay["d2"])) and (d1_confirmed or decay["d2"]):
        # pre-committed: D1 XOR D2 = watch annotation only, no state change, no escalation
        watch = (f"decay-watch: D1_streak={hist['d1_streak']}/{CONSECUTIVE_NEEDED} "
                 f"D2={decay['d2']} (cusum {decay['cusum_peak']}) — rule needs BOTH")
        new = "real_capital_candidate" if gates_all_pass else ("evidence" if n_days >= EVIDENCE_MIN_DAYS else "shadow")
    else:
        new = "real_capital_candidate" if gates_all_pass else ("evidence" if n_days >= EVIDENCE_MIN_DAYS else "shadow")

    changed = new != cur
    if changed:
        reg.update(book, lifecycle=new)
    _write_verdict(book, new, decay, gates_all_pass, n_days, watch)
    return {"lifecycle": new, "decay": decay, "changed": changed, "watch": watch}


def _write_verdict(book: str, lifecycle: str, decay: Optional[dict], gates_all_pass: bool,
                   n_days: int, watch: Optional[str]) -> None:
    """#34 one-direction contract: data/live/<book>/lifecycle_verdict.json (schema v1).
    During the soak the direct reg.update above still runs; atlas's intake consumer
    verifies in shadow mode that applying the artifact would land the same state.
    After a clean soak the direct mutation dies and the artifact is the only channel."""
    try:
        d = LIVE / book
        d.mkdir(parents=True, exist_ok=True)
        v = {"schema_version": 1, "book": book,
             "asof": datetime.now().strftime("%Y-%m-%d"),
             "lifecycle": lifecycle, "gates_all_pass": gates_all_pass, "n_days": n_days,
             "decay": {k: decay[k] for k in ("d1", "d2", "cusum_peak", "roll_mean")} if decay else None,
             "watch": watch}
        tmp = d / "lifecycle_verdict.json.tmp"
        tmp.write_text(json.dumps(v, indent=2, default=str), encoding="utf-8")
        tmp.replace(d / "lifecycle_verdict.json")
    except OSError as e:
        print(f"[lifecycle] WARNING: verdict artifact write failed for {book}: {e}")


def retire(book: str) -> bool:
    """HUMAN-ONLY terminal transition. Stops the daily loop for this book; positions are closed
    manually at the broker (no auto-liquidation, board policy)."""
    reg = _registry()
    ok = reg.update(book, lifecycle="retired")
    if ok:
        _write_verdict(book, "retired", None, False, 0, "human-confirmed retirement")
        print(f"[lifecycle] {book} -> RETIRED (human-confirmed). Daily loop will skip it; "
              f"close any open positions manually at the broker.")
    else:
        print(f"[lifecycle] {book} not found in registry")
    return ok


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["retire"] and len(args) == 2:
        return 0 if retire(args[1]) else 1
    reg = _registry()
    for s in reg.load():
        print(f"  {s.name:28s} lifecycle={getattr(s, 'lifecycle', 'shadow'):24s} "
              f"exec_state={s.state} cap=${s.capital:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
