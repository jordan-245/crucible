"""Write an experiment verdict into the shared research wiki (the compounding memory)."""
from datetime import date
from pathlib import Path

from crucible_paths import WIKI  # central config


def write_experiment(spec, verdict: dict):
    status = ("VALIDATED" if verdict["PASSED_ALL_GATES"]
              else "REJECTED-MCPT" if verdict.get("stage1_pass") and verdict.get("mcpt_pass") is False
              else "CANDIDATE" if verdict.get("stage1_pass")   # cleared stage-1, awaiting confirmation
              else "NEAR-MISS" if (verdict.get("dsr") or 0) and verdict["dsr"] >= 0.85
              else "FAIL")
    page = WIKI / "experiments" / f"{spec.id}.md"
    # ID-COLLISION GUARD: two different proposals can pick the same spec.id (happened with
    # amihud_illiquidity_smallcap 2026-06-10: a cost-hardened VARIANT silently overwrote the
    # original PROMOTE-tier page). Never overwrite a page whose title differs — version it.
    if page.exists():
        try:
            old = page.read_text()
            old_title = next((l[2:].strip() for l in old.splitlines() if l.startswith("# ")), "")
            if old_title and old_title != spec.title:
                import hashlib
                suffix = hashlib.sha1(spec.title.encode()).hexdigest()[:6]
                page = WIKI / "experiments" / f"{spec.id}__{date.today().strftime('%Y%m%d')}-{suffix}.md"
                print(f"[wiki] id collision on '{spec.id}' (existing page is a different experiment) "
                      f"-> versioned page {page.name}")
        except Exception:
            pass
    page.write_text(f"""---
id: {spec.id}
status: {status}
project: {spec.project}
date: {date.today()}
family: {spec.family}
markets: {spec.markets}
data: {spec.data_desc}
generated_by: crucible-agent
---
# {spec.title}

## Pre-registration (FROZEN before running)
{spec.pre_registration}

## Verdict: {status}
- tier **{verdict['tier']}** (FDR bar {verdict['promote_bar']}, n_families {verdict['n_families']})
- DSR {verdict['dsr']} | CPCV {verdict['median_cpcv']} | PBO {verdict['pbo']}
- search Sharpe {verdict['search_sharpe']} -> holdout Sharpe {verdict['holdout_sharpe']} | **holdout_gate PASS={verdict['holdout_pass']}** {verdict['holdout_reasons']}
- deployment passed={verdict['deployment_passed']} peak={verdict['deploy_peak']} sectors={verdict['deploy_sectors']} {verdict['deploy_reasons']}
- full Sharpe {verdict['full_sharpe']} | maxDD {verdict['full_maxdd']} | trades {verdict['n_trades']}
- stage-1 pass: {verdict.get('stage1_pass')} | scope: {verdict.get('scope')}
- stage-2 MCPT: pass={verdict.get('mcpt_pass')} {verdict.get('mcpt') or ''}
- stage-2 generalization: {verdict.get('generalization')} — {verdict.get('generalization_note') or 'n/a'}
- needs_confirmation: {verdict.get('needs_confirmation') or 'none'}
- **PASSED ALL GATES: {verdict['PASSED_ALL_GATES']}**
""")
    # append to log + index (use the page stem — may be the versioned name on collision)
    stem = page.stem
    with open(WIKI / "log.md", "a") as f:
        f.write(f"\n## [{date.today()}] experiment | {stem} -> {status} (tier {verdict['tier']}, holdout {verdict['holdout_pass']})")
    with open(WIKI / "index.md", "a") as f:
        f.write(f"\n- [[experiments/{stem}]] — {status} ({spec.title})")
    return page
