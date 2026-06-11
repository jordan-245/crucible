"""Unified morning report — ONE Telegram message covering the whole research operation.

Sections:
  1. Forge night: every cycle since the last forge-timer trigger (not just last 5),
     verdict mix, stage timings, codegen quality, FDR bar trajectory.
  2. Forward-paper: latest val_mom_trend_smallcap run + realized-return track state.
  3. BAB forward validation: ledger delta + days to verdict.
  4. Ops: service failures, queue state, killswitch.

Replaces digest.py as the human-facing daily picture (digest.py retained for ad-hoc use).
Scheduled by crucible-morning-report.timer at 07:00 AEST (after the 03:30 forge night).
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sdk.notify import telegram_msg

from crucible_paths import ROOT, WIKI, DEPLOY_TARGET  # central config
RUNLOG = ROOT / "agent" / "run_log.jsonl"
ATLAS_LIVE = (DEPLOY_TARGET / "data" / "live") if DEPLOY_TARGET else None
REGISTRY = (DEPLOY_TARGET / "config" / "live_strategies.json") if DEPLOY_TARGET else None
BAB_LEDGER = ROOT / "forward" / "bab_ledger.jsonl"


def _jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _fmt_s(sec) -> str:
    return f"{sec/60:.0f}m" if isinstance(sec, (int, float)) else "?"


def forge_section() -> list:
    cutoff = (datetime.now() - timedelta(hours=18)).isoformat()
    runs = [r for r in _jsonl(RUNLOG) if r.get("ts", "") > cutoff]
    lines = [f"🔨 <b>Forge night</b> — {len(runs)} cycles"]
    if not runs:
        lines.append("  (no cycles — check crucible-forge.timer)")
        return lines
    n_pass = sum(1 for r in runs if r.get("passed_all"))
    tiers = {}
    for r in runs:
        v = r.get("verdict") or {}
        t = v.get("tier") if isinstance(v, dict) else "CRASH"
        tiers[t] = tiers.get(t, 0) + 1
    lines.append("  " + " · ".join(f"{k or 'none'}:{v}" for k, v in sorted(tiers.items(), key=lambda x: -x[1])))
    if n_pass:
        lines.append(f"  🟢 <b>{n_pass} FULL PASS — human review required</b>")
    for r in runs:
        v = r.get("verdict") or {}
        tier = (v.get("tier") if isinstance(v, dict) else None) or "—"
        mark = ("🟢" if r.get("passed_all")
                else "✗" if (tier in ("FAIL", "SCREEN_FAIL") or not r.get("ran"))
                else "🟡")
        # O4/O3: a non-run shows WHY (schema-2 fail_reason), not a bare dash
        why = f" [{r['fail_reason']}]" if (not r.get("ran") and r.get("fail_reason")) else ""
        lines.append(f"  {mark} {tier[:7]:<7} {str(r.get('title', '?'))[:52]}{why}")
    # O4: near-misses deserve eyes — they're the director's mutation fuel
    nm = [r for r in runs if isinstance(r.get("verdict"), dict)
          and (r["verdict"].get("dsr") or 0) >= 0.85 and not r.get("passed_all")]
    for r in nm:
        v = r["verdict"]
        lines.append(f"  🔍 near-miss: {str(r.get('title','?'))[:40]} DSR {v.get('dsr')} "
                     f"holdout {v.get('holdout_sharpe')} (bar was {v.get('promote_bar')})")
    # stage health (instrumented runs only)
    st = [r["stages"] for r in runs if isinstance(r.get("stages"), dict)]
    if st:
        cg = [s["codegen_s"] for s in st if s.get("codegen_s")]
        bt = [s["backtest_s"] for s in st if s.get("backtest_s")]
        empty = sum(1 for s in st if (s.get("codegen_attempts") or 1) > 1)
        fixes = sum(1 for s in st if s.get("consistency_fix"))
        retries = sum(max(0, (s.get("run_attempts") or 1) - 1) for s in st)
        lines.append(f"  ⏱ codegen med {_fmt_s(sorted(cg)[len(cg)//2]) if cg else '?'}"
                     f" · backtest med {_fmt_s(sorted(bt)[len(bt)//2]) if bt else '?'}"
                     f" · empty-gen {empty}/{len(st)} · thesis-fix {fixes} · run-retries {retries}")
    # FDR bar trajectory
    reg = _jsonl(WIKI / ".registry" / "hypothesis_registry.jsonl")
    if reg:
        bars = [r.get("promote_dsr") for r in reg if r.get("promote_dsr")]
        fams = reg[-1].get("n_families")
        if bars:
            lines.append(f"  📈 FDR bar {bars[-1]:.3f} ({fams} families)"
                         + (f" — was {bars[-10]:.3f} 10 runs ago" if len(bars) >= 10 else ""))
    return lines


def forward_paper_section() -> list:
    """ALL deployed paper-book strategies (virtual sub-books) + portfolio rollup. Scales with N."""
    try:
        reg = json.loads(REGISTRY.read_text()) if (REGISTRY and REGISTRY.exists()) else []
    except Exception:
        reg = []
    if not reg:
        return ["📄 <b>Paper portfolio</b> — no strategies deployed"]
    lines = [f"📄 <b>Paper portfolio</b> — {len(reg)} strateg" + ("y" if len(reg) == 1 else "ies")]
    tot_eq, tot_base = 0.0, 0.0
    for s in reg:
        name = s.get("name", "?")
        d = ATLAS_LIVE / name
        runs = _jsonl(d / "runs.jsonl")
        rets = _jsonl(d / "returns.jsonl")
        book = {}
        try:
            book = json.loads((d / "book.json").read_text()) if (d / "book.json").exists() else {}
        except Exception:
            pass
        base = float(book.get("capital_base") or s.get("capital") or 0)
        eq = None
        try:
            eq = json.loads((d / "equity_state.json").read_text()).get("equity")
        except Exception:
            pass
        cum = 1.0
        for r in rets:
            cum *= 1 + (r.get("ret") or 0)
        last_run = runs[-1] if runs else {}
        bits = [f"  • {name} [{s.get('state','?')}]"]
        if eq is not None:
            bits.append(f"${eq:,.0f}")
            tot_eq += eq
            tot_base += base
        if rets:
            bits.append(f"cum {(cum-1)*100:+.2f}% ({len(rets)}d, last {rets[-1].get('ret',0)*100:+.2f}%)")
        else:
            bits.append("0 returns yet")
        if last_run:
            bits.append(f"orders {last_run.get('n_orders',0)} · track={last_run.get('track','?')}")
            if last_run.get("blocked"):
                bits.append(f"⚠ {last_run['blocked']}")
        lines.append(" · ".join(bits))
    if tot_base > 0:
        pnl = tot_eq - tot_base
        lines.append(f"  Σ ${tot_eq:,.0f} on ${tot_base:,.0f} · P&L {'+' if pnl >= 0 else ''}{pnl:,.0f} "
                     f"({(tot_eq/tot_base-1)*100:+.2f}%)")
    return lines


def bab_section() -> list:
    led = _jsonl(BAB_LEDGER)
    if not led:
        return []
    last = led[-1]
    days_left = (datetime.fromisoformat("2026-12-09") - datetime.now()).days
    fs = last.get("fwd_sharpe")
    return [f"🛡 <b>BAB forward</b> — {last.get('fwd_days', 0)}d tracked, "
            f"fwd Sharpe {fs if fs is not None else '—'}, "
            f"cum {last.get('fwd_cum_return', 0)*100:+.1f}% · verdict in {days_left}d"]


def loop_health_section() -> list:
    """Loop-health KPIs (LOOPS_FRAMEWORK_PLAN 1.2): cost-per-accepted-change + retry-rate
    trends from run_log schema-2 stages. Catches regressions like the Fable-5 'empty
    codegen' pattern QUANTITATIVELY (retry-rate spike) instead of by eyeball."""
    rows = [r for r in _jsonl(RUNLOG) if isinstance(r.get("stages"), dict)]
    if len(rows) < 3:
        return []
    lines = ["\U0001f501 <b>Loop health</b> (7-night window vs prior)"]
    cut_now = (datetime.now() - timedelta(days=7)).isoformat()
    cut_prev = (datetime.now() - timedelta(days=14)).isoformat()
    cur = [r for r in rows if r.get("ts", "") > cut_now]
    prev = [r for r in rows if cut_prev < r.get("ts", "") <= cut_now]

    def _kpis(rs):
        if not rs:
            return None
        n = len(rs)
        accepted = sum(1 for r in rs if (r.get("verdict") or {}).get("tier") in ("PROMOTE", "SCREEN")
                       or r.get("passed_all"))
        retries = sum(max(0, (r["stages"].get("run_attempts") or 1) - 1)
                      + max(0, (r["stages"].get("codegen_attempts") or 1) - 1) for r in rs)
        crashes = sum(1 for r in rs if r.get("fail_reason"))
        wall = sorted(r["stages"].get("total_s") or 0 for r in rs)[len(rs) // 2]
        return {"n": n, "accepted": accepted, "retry_rate": retries / n,
                "crash_rate": crashes / n, "med_wall_s": wall}

    k, p = _kpis(cur), _kpis(prev)
    if not k:
        return []
    cpac = f"{k['n']}/{k['accepted']}" if k["accepted"] else f"{k['n']}/0 (∞)"
    lines.append(f"  cost-per-accepted: {cpac} hypotheses/accept · retry-rate "
                 f"{k['retry_rate']:.2f}/run · crash {k['crash_rate']:.0%} · med wall {_fmt_s(k['med_wall_s'])}")
    if p:
        d_retry = k["retry_rate"] - p["retry_rate"]
        if abs(d_retry) >= 0.5:
            arrow = "⚠️ UP" if d_retry > 0 else "↓ down"
            lines.append(f"  {arrow} retry-rate {p['retry_rate']:.2f} → {k['retry_rate']:.2f} "
                         f"vs prior week" + (" — codegen/model regression? check fail_reasons" if d_retry > 0 else ""))
        if p["crash_rate"] == 0 and k["crash_rate"] > 0.2:
            lines.append(f"  ⚠️ crash-rate jumped 0% → {k['crash_rate']:.0%}")
    return lines


def ops_section() -> list:
    lines = []
    if (ROOT / "LOOP_DISABLED").exists():
        lines.append("⛔ LOOP_DISABLED is set — forge halted")
    try:
        failed = subprocess.run(["systemctl", "--failed", "--no-legend", "--plain"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        relevant = [l.split()[0] for l in failed.splitlines()
                    if any(k in l for k in ("crucible", "heph", "atlas", "forward"))]
        if relevant:
            lines.append("🔴 failed units: " + ", ".join(relevant))
    except Exception:
        pass  # non-systemd host: skip unit health
    try:
        from sdk import queue
        q = queue.stats()
        lines.append(f"⚙️ queue {q.get('queued', 0)} queued / {q.get('claimed', 0)} claimed")
    except Exception:
        pass
    return lines


def main() -> None:
    sections = (forge_section() + [""] + forward_paper_section() + [""]
                + bab_section() + loop_health_section() + ops_section())
    msg = "☀️ <b>Morning report</b> — " + datetime.now().strftime("%a %Y-%m-%d") + "\n\n" \
          + "\n".join(s for s in sections if s is not None)
    ok = all(telegram_msg(part) for part in _split_html(msg))
    print(f"[morning_report] sent={ok} chars={len(msg)}")
    if not ok:
        sys.exit(1)  # visible as a failed unit -> shows up in tomorrow's ops section


def _split_html(msg: str, limit: int = 4000) -> list:
    """O4 truncation fix: msg[:4000] could cut MID-<b> TAG -> Telegram rejects the whole
    message as malformed HTML -> busiest nights silently lose their report. Split on line
    boundaries instead and close any dangling <b> per part (the only tag we emit)."""
    if len(msg) <= limit:
        return [msg]
    parts, cur = [], ""
    for line in msg.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            parts.append(cur)
            cur = line
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        parts.append(cur)
    fixed = []
    for p in parts:
        if p.count("<b>") > p.count("</b>"):
            p += "</b>"
        fixed.append(p)
    return fixed


if __name__ == "__main__":
    main()
