"""Unified morning report — ONE Telegram message covering the whole research operation.

Sections:
  1. Forge night: every cycle since the last forge-timer trigger (not just last 5),
     verdict mix, stage timings, codegen quality, FDR bar trajectory.
  2. Forward-paper: latest val_mom_trend_smallcap run + realized-return track state.
  3. BAB forward validation: ledger delta + days to verdict.
  4. Ops: service failures, queue state, killswitch.

Replaces digest.py as the human-facing daily picture (digest.py retained for ad-hoc use).
Scheduled by hephaestus-morning-report.timer at 07:00 AEST (after the 03:30 forge night).
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "/root/hephaestus")
from sdk.notify import telegram_msg

ROOT = Path("/root/hephaestus")
WIKI = Path("/root/research-wiki")
RUNLOG = ROOT / "agent" / "run_log.jsonl"
LIVE = Path("/root/atlas/data/live/val_mom_trend_smallcap")
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
        lines.append("  (no cycles — check hephaestus-forge.timer)")
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
        mark = "🟢" if r.get("passed_all") else ("✗" if tier in ("FAIL", "SCREEN_FAIL") else "🟡")
        lines.append(f"  {mark} {tier[:7]:<7} {str(r.get('title', '?'))[:52]}")
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
    lines = ["📄 <b>Forward-paper</b> (val_mom_trend_smallcap)"]
    runs = _jsonl(LIVE / "runs.jsonl")
    rets = _jsonl(LIVE / "returns.jsonl")
    if not runs:
        lines.append("  no runs recorded — check atlas-forward-paper.timer")
        return lines
    last = runs[-1]
    lines.append(f"  {last.get('date')}: {last.get('n_orders', 0)} orders, "
                 f"turnover ${last.get('turnover', 0):,.0f}, track={last.get('track', '?')}"
                 + (f" ⚠ blocked: {last['blocked']}" if last.get("blocked") else ""))
    if rets:
        cum = 1.0
        for r in rets:
            cum *= 1 + (r.get("ret") or 0)
        lines.append(f"  {len(rets)} daily returns · cum {(cum-1)*100:+.2f}% · last {rets[-1].get('ret', 0)*100:+.2f}%")
    else:
        lines.append("  0 returns yet (first lands after the first full US session)")
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


def ops_section() -> list:
    lines = []
    if (ROOT / "LOOP_DISABLED").exists():
        lines.append("⛔ LOOP_DISABLED is set — forge halted")
    failed = subprocess.run(["systemctl", "--failed", "--no-legend", "--plain"],
                            capture_output=True, text=True, timeout=5).stdout.strip()
    relevant = [l.split()[0] for l in failed.splitlines()
                if any(k in l for k in ("heph", "atlas", "forward"))]
    if relevant:
        lines.append("🔴 failed units: " + ", ".join(relevant))
    try:
        from sdk import queue
        q = queue.stats()
        lines.append(f"⚙️ queue {q.get('queued', 0)} queued / {q.get('claimed', 0)} claimed")
    except Exception:
        pass
    return lines


def main() -> None:
    sections = (forge_section() + [""] + forward_paper_section() + [""]
                + bab_section() + ops_section())
    msg = "☀️ <b>Morning report</b> — " + datetime.now().strftime("%a %Y-%m-%d") + "\n\n" \
          + "\n".join(s for s in sections if s is not None)
    ok = telegram_msg(msg[:4000])  # Telegram hard limit 4096
    print(f"[morning_report] sent={ok} chars={len(msg)}")
    if not ok:
        sys.exit(1)  # visible as a failed unit -> shows up in tomorrow's ops section


if __name__ == "__main__":
    main()
