# On-Demand Big-RAM Compute — LOCKED SPEC (Hetzner-x86)

> **FORM UPDATED 2026-06-09:** was AWS-ARM spot; board reopened on a new fact and flipped
> UNANIMOUS (5–0) to **Hetzner Cloud x86 on-demand**. (Filename kept for link stability.)
> Trigger to build: next OOM/concurrency need. Fleet stays deferred to first deployed revenue.
> Sources: ceo-board/memos/2026-06-09-compute-substrate-hostinger-ceiling/ (supersedes the form of
> 2026-06-09-aws-spot-backtesting).

## The two facts that set this form
1. **Hostinger is maxed** — always-on VPS is KVM 8 (8 vCPU / 32 GB), Hostinger's top plan. No bigger RAM tier. "Just resize" is impossible; the cheap vertical fallback is gone.
2. **Hetzner Cloud CCX = x86/AMD = same arch as local (x86_64)** → eliminates the ARM↔x86 reproducibility-drift gate AND all AMI-bake/ARM-wheel toil. Hourly billing, ~1/5 AWS cost.

## Build trigger (any one)
- Next heavy battery that OOMs on 32 GB, OR ≥2 concurrent batteries, OR first deployed-strategy revenue.
- Until built: **status quo (C)** — serial single batteries on the 32 GB box + swap (~50–100× slower; works, unattended). NOT the endgame, just the bridge until the reaper is proven.

## NON-NEGOTIABLE GATES — no remote run until every item is green

### Gate 1 — Integrity (single authority; token/SSH-scoped, not policy)
- **Remote = dumb calculator.** Receives code + **non-holdout** data, runs the signal, emits **raw returns/trades** per hypothesis to a scoped results location. It does NOT run CPCV/DSR/PBO, touch `/root/research-wiki/.registry`, see holdout, or write any verdict.
- **Local x86 VPS = sole rails + FDR-registry + write-once-holdout authority.** Local pulls artifacts, runs `research_integrity`, appends the registry under the existing single-writer FileLock. One ledger appender, always.
- **Holdout data NEVER leaves the local box.**
- **Scoping:** the remote box / its token gets write-only to its own results prefix; zero registry/holdout read or write. Prefer: local pushes inputs to the box over SSH, box writes results back to a dedicated dir local pulls — no shared secret store on the box.

### Gate 2 — Reproducibility: **DELETED** (arch-match)
- Hetzner-x86 ≡ local x86 → no float divergence. One-time sanity check only: run **one reference hypothesis** on Hetzner, confirm it matches local within tolerance. No containers / bit-tolerance proofs / per-PASS re-validation.

### Gate 3 — Money / orphan control: **STRENGTHENED (self-built — Hetzner has NO native billing kill-switch)**
- **External, unconditional reaper built BY US:** a LOCAL cron (on the always-on box) that calls the Hetzner API and **destroys any server tagged `purpose=backtest` older than a ~2h TTL, regardless of server health.** This is the primary cleanup — `shutdown -h now` self-terminate is best-effort only (fails on every exception path before the final call).
- **Spend monitor:** local cron checks Hetzner billing / running-server count; alerts (Telegram) on any server alive >TTL or projected month >$10.
- **Prove the reaper by deliberately orphaning a test server** (leave one running, confirm the cron kills it) BEFORE any real run.
- Per-run: pick the smallest box that fits in RAM (no swap), destroy on completion.

### Gate 4 — No LLM on remote
- Codegen/propose stay **local** (Claude Max OAuth, $0). Remote runs **frozen, hashed pure-compute Python** only.

## Checkpointing (interruption safety)
- Write each hypothesis result to the results dir as it completes (atomic), pulled incrementally — so an interrupted/destroyed box costs one in-flight run, not the battery. (Hetzner on-demand isn't pre-emptible like spot, but TTL-kill + crashes still warrant it.)

## DO NOT BUILD YET (deferred behind the revenue trigger)
- The **N-agent hypothesis-queue/locks fan-out** (the "fleet"). Build the **boundary + reaper** first; turn the dial to N only once a strategy is deployed and revenue justifies throughput. *Throughput is motion, not money.*

## Sizing & cost
- CCX33 (8 vCPU/32 GB) ≈ €0.10/hr; **CCX43 (16/64 GB) ≈ €0.20/hr — default for single legs**; CCX53 (32/128 GB) ≈ €0.40/hr for concurrent. 20–40 min battery ≈ a few cents; realistic **<$10/mo**.

## What the operator must supply to build
- A **Hetzner Cloud project + scoped API token** (and an SSH key for the box). Everything else (provision/run/pull/destroy script, reaper cron, boundary) is local build work.

## Migrate-the-whole-stack (Option A) — REJECTED 5–0
Moving Atlas/Cronus/Hermes/Hephaestus/wiki off Hostinger = un-staged 5-service live cutover (new IP, DNS, redeploy), days of toil + live-system risk, to solve a burst problem. Don't move the home to fix a burst. Revisit only if always-on resource needs (not burst) outgrow KVM 8 structurally.
