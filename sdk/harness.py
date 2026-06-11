"""Hephaestus Strategy SDK — the fixed harness.

The agent writes ONLY a StrategySpec (a signal fn + data loader + pre-registration + grid).
This harness owns everything else and the rails are NON-BYPASSABLE: data split, the
research_integrity gates (CPCV/DSR/PBO + FDR bar + write-once holdout + deployment-sanity),
the FDR registry append, the verdict, the wiki write, and the Telegram-on-pass.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

# rails (per-project FDR/holdout state lives in the research-wiki so it's SHARED across agents)
from crucible_paths import WIKI, REGISTRY
from sdk.locks import FileLock
os.environ.setdefault("RESEARCH_INTEGRITY_DIR", str(REGISTRY))
import research_integrity as ri

SCREEN_FLOOR = 0.3   # tier-0: |search Sharpe| below this -> no in-sample edge -> skip grid+CPCV, keep holdout
BETA_HI = 0.6        # long-only beta-to-universe above this + weak selection-alpha -> beta-confound
SEL_FLOOR = 0.4      # selection-alpha Sharpe a high-beta strategy must clear to be a real edge (not just beta)


@dataclass
class StrategySpec:
    """What the agent fills in. Everything else is the fixed harness."""
    id: str                                   # unique, kebab-case -> experiments/<id>.md
    family: str                               # FDR family (one idea); the bar rises per distinct family
    title: str
    markets: list                             # ["futures"], ["crypto"], ...
    data_desc: str                            # human description (free? owned? source)
    pre_registration: str                     # the FROZEN design text (written BEFORE running)
    load_data: Callable[[], pd.DataFrame]     # () -> panel the signal consumes
    signal: Callable[..., tuple]              # (panel, **params) -> (daily_returns: pd.Series, trades: list[dict])
    default_params: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)  # label -> params override (DSR effective-N; honest search burden)
    holdout_start: str = "2022-01-01"         # write-once quarantine
    deploy_max_positions: int = 10
    # PRE-REGISTERED hedge sleeve (2026-06-12): broad-index-ETF tickers + position-day share cap.
    # Declared here = part of the FROZEN design (hashed into the write-once config). The
    # deployment gate then judges the ALPHA book alone (stricter) and gates the sleeve on
    # whitelist + cap — see research_integrity.deployment.HEDGE_ETF_WHITELIST.
    hedge_tickers: list = None                # e.g. ["IWM"]; None = no sleeve (legacy behavior)
    hedge_cap: float = None                   # e.g. 0.35; required iff hedge_tickers given
    # PROMOTION POLICY (2026-06-09): a stage-1 gate pass is only a CANDIDATE. Confirmation needs
    # either cross-market generalization (broad mechanism) or forward-validation (defensibly local).
    scope: str = "broad"                       # "broad" (universal mechanism -> MUST generalize) | "local" (defensible universe-specific -> forward-validate)
    generalization_universes: list = field(default_factory=list)  # broad scope: untouched universes to confirm the mechanism in
    load_gen_data: Callable = None             # broad scope: label -> panel for each generalization universe (UNTOUCHED; same shape as load_data())
    project: str = "crucible"


from sdk.stats import sharpe as _sharpe, maxdd as _maxdd_canon  # canonical (sdk/stats.py)


def _provenance(spec) -> dict:
    """O1: WHICH code produced this verdict. Module SHA pins the generated strategy file;
    repo SHA pins the harness/gate stack it ran through. Never raises."""
    module_file, module_sha = None, None
    try:
        import hashlib
        import inspect
        src = inspect.getsourcefile(spec.signal)
        if src and Path(src).exists():
            module_file = str(Path(src))
            module_sha = hashlib.sha1(Path(src).read_bytes()).hexdigest()[:12]
    except Exception:
        pass
    return {"module_file": module_file, "module_sha": module_sha, "repo_sha": _repo_sha(),
            "default_params": dict(spec.default_params or {}), "holdout_start": spec.holdout_start}


def _repo_sha() -> str | None:
    """Crucible repo HEAD at run time (O1 provenance). Read from .git directly — no subprocess
    (works inside restricted contexts), never raises."""
    try:
        git = Path(__file__).resolve().parents[1] / ".git"
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            return (git / head[5:]).read_text().strip()[:12]
        return head[:12]
    except Exception:
        return None


def _sharpe_doc_anchor():
    """_sharpe is sdk.stats.sharpe — ONE definition repo-wide (was 8 divergent copies)."""


_maxdd = _maxdd_canon

def _price_matrix(panel):
    """Best-effort (dates x assets) price matrix for the long-only benchmark; None if not a price panel."""
    try:
        if isinstance(panel, pd.DataFrame) and isinstance(panel.columns, pd.MultiIndex):
            lvl0 = set(panel.columns.get_level_values(0))
            for key in ("px", "close", "closeadj", "price", "prices", "adj_close"):
                if key in lvl0:
                    return panel[key]
            return None
        if isinstance(panel, pd.DataFrame) and panel.shape[1] >= 5:
            return panel
    except Exception:
        return None
    return None


def _beta_decomp(search_ret, price_mx):
    """Decompose SEARCH returns into universe-beta + selection-alpha (residual after removing the equal-weight
    universe return). Cheap stage-1 version of MCPT's finding: catches the long-only-beta confound."""
    if price_mx is None:
        return None
    try:
        ew = price_mx.pct_change().mean(axis=1)
        df = pd.concat([pd.Series(search_ret), ew], axis=1).dropna()
        df.columns = ["r", "m"]
        if len(df) < 60 or df["m"].var() == 0:
            return None
        beta = df["r"].cov(df["m"]) / df["m"].var()
        return {"beta_to_universe": round(float(beta), 2),
                "selection_alpha_sharpe": _sharpe(df["r"] - beta * df["m"])}
    except Exception:
        return None


def _permute_panel_prices(panel, rng):
    """Permute each asset's daily PRICE returns (destroys serial + cross-sectional structure, keeps the
    marginal distribution and the listing/delisting NaN shape); rebuild prices; keep every non-price
    block (volume, bvps, ...) as-is. Returns the permuted panel, or None if no price matrix found."""
    px = _price_matrix(panel)
    if px is None:
        return None
    rets = px.pct_change()
    out = {}
    for c in rets.columns:
        s = rets[c]
        mask = s.notna()
        perm = s.copy()
        perm[mask] = rng.permutation(s[mask].values)
        out[c] = perm
    sh = pd.DataFrame(out, index=rets.index)
    px_p = (1 + sh.fillna(0)).cumprod()
    px_p = px_p.div(px_p.bfill().iloc[0]).mul(px.bfill().iloc[0])
    px_p = px_p.where(px.notna())
    if isinstance(panel.columns, pd.MultiIndex):
        lvl0 = list(dict.fromkeys(panel.columns.get_level_values(0)))
        price_key = next(k for k in ("px", "close", "closeadj", "price", "prices", "adj_close") if k in lvl0)
        blocks = {k: (px_p if k == price_key else panel[k]) for k in lvl0}
        new = pd.concat(blocks, axis=1)
    else:
        new = px_p
    new.attrs = dict(panel.attrs)
    return new


def _mcpt_stat(spec: StrategySpec, panel, benchmark_relative: bool):
    """MCPT statistic on ONE panel. benchmark_relative (long-biased books): Sharpe MINUS the
    equal-weight-universe Sharpe on the SAME panel — time-shuffling destroys cross-sectional
    correlation, which mechanically inflates ANY diversified long book's Sharpe to ~3+ (fake
    perfect diversification that cannot exist in real markets); the EW benchmark enjoys the
    identical inflation, so the difference isolates SELECTION skill. Market-neutral books
    (legs cancel the common factor): absolute Sharpe is a fair null."""
    sh = _sharpe(pd.Series(spec.signal(panel, **spec.default_params)[0]).dropna())
    if not benchmark_relative:
        return sh
    px = _price_matrix(panel)
    bench = _sharpe(px.pct_change().mean(axis=1)) if px is not None else 0.0
    return sh - bench


# --- parallel plumbing (fork-only: workers inherit spec/panel/loaders by COW, no pickling of
#     generated-strategy closures needed). Serial fallback on non-POSIX / 1 worker. ---
_MCPT_CTX: dict = {}
_FORK_CTX: dict = {}


def _fork_call(item):
    """Worker: apply the COW-inherited fn to one item -> (item, result | ('err', msg))."""
    try:
        return item, _FORK_CTX["fn"](item)
    except Exception as e:
        return item, ("err", f"{type(e).__name__}: {str(e)[:120]}")


def _fork_map(fn, items, workers: int):
    """Map fn over items with a fork pool (closures OK — inherited, not pickled).
    Yields (item, result) where result may be ('err', msg). Serial fallback off-posix."""
    items = list(items)
    if os.name != "posix" or workers <= 1 or len(items) <= 1:
        for it in items:
            try:
                yield it, fn(it)
            except Exception as e:
                yield it, ("err", f"{type(e).__name__}: {str(e)[:120]}")
        return
    import multiprocessing as mp
    _FORK_CTX["fn"] = fn
    try:
        with mp.get_context("fork").Pool(min(workers, len(items))) as pool:
            yield from pool.imap(_fork_call, items)
    finally:
        _FORK_CTX.clear()


def _mcpt_one_perm(seed: int):
    """Worker: one permutation -> statistic (or ('err', msg)). Reads _MCPT_CTX set pre-fork."""
    rng = np.random.default_rng(seed)
    p = _permute_panel_prices(_MCPT_CTX["panel"], rng)
    if p is None:
        return None
    try:
        return float(_mcpt_stat(_MCPT_CTX["spec"], p, _MCPT_CTX["bench_rel"]))
    except Exception as e:  # surfaced + counted by the caller
        return ("err", f"{type(e).__name__}: {str(e)[:100]}")


def _mcpt_workers() -> int:
    """Worker count: CRUCIBLE_MCPT_WORKERS env > default min(cores-2, 6). Capped because each
    worker materializes a permuted panel copy (large-cap panels have OOM'd at 12G serial)."""
    try:
        env = int(os.environ.get("CRUCIBLE_MCPT_WORKERS", 0))
    except ValueError:
        env = 0
    if env > 0:
        return env
    return max(1, min((os.cpu_count() or 2) - 2, 6))


def _stage2_mcpt(spec: StrategySpec, panel, real_sharpe: float, n: int = 50,
                 beta_to_universe: float | None = None) -> tuple:
    """STAGE-2 MCPT (Monte Carlo Permutation Test) — runs BEFORE breadth. Permutes the price panel
    (no structure left to exploit), re-runs the FROZEN signal, p = P(perm stat >= real stat). A
    construction artifact (vol-targeted noise-sorting, bid-ask bounce harvesting) reproduces on
    permuted data -> high p. Breadth CANNOT catch these (they replicate on every universe — confirmed
    twice: value×mom p=0.97 after 2/2 tiers, amihud p=0.94 after 3/3 universes). PASS bar p<=0.05.
    LONG-BIASED books (|beta| evidence of carrying the universe factor, beta > 0.3) use the
    benchmark-RELATIVE statistic (see _mcpt_stat) — absolute Sharpe under time-shuffle mechanically
    fails every diversified long book (decorrelation inflation, perm means 3-5 observed).
    Early-stops once enough exceedances make p>0.05 mathematically certain. Returns (result, passed)."""
    bench_rel = beta_to_universe is not None and beta_to_universe > 0.3
    try:
        real_stat = _mcpt_stat(spec, panel, bench_rel)
    except Exception as e:
        return {"note": f"real-stat computation failed: {type(e).__name__}: {str(e)[:100]}"}, False
    if _permute_panel_prices(panel, np.random.default_rng(0)) is None:
        return {"note": "no price matrix — MCPT not applicable to this panel shape"}, True
    max_exceed = int(0.05 * (n + 1)) - 1            # max exceedances that still allow p<=0.05
    seeds = list(range(1, n + 1))                   # deterministic, order-independent
    n_workers = _mcpt_workers()
    perms, exceed = [], 0

    def _ingest(sh) -> bool:
        """Record one perm result; True -> failure is mathematically certain, stop."""
        nonlocal exceed
        if sh is None or (isinstance(sh, tuple) and sh[0] == "err"):
            if isinstance(sh, tuple):
                print(f"[mcpt] perm failed: {sh[1]}")
            return False
        perms.append(sh)
        if sh >= real_stat:
            exceed += 1
        return exceed > max_exceed

    stopped = False
    if os.name == "posix" and n_workers > 1:
        # fork pool: COW-shared panel/spec; batch-wise early-stop keeps the compute bound
        import multiprocessing as mp
        _MCPT_CTX.update({"spec": spec, "panel": panel, "bench_rel": bench_rel})
        try:
            with mp.get_context("fork").Pool(n_workers) as pool:
                for batch_start in range(0, n, n_workers):
                    batch = seeds[batch_start:batch_start + n_workers]
                    for sh in pool.map(_mcpt_one_perm, batch):
                        if _ingest(sh):
                            stopped = True
                            break
                    if stopped:
                        break
        finally:
            _MCPT_CTX.clear()
    else:
        for s in seeds:
            rng = np.random.default_rng(s)
            p = _permute_panel_prices(panel, rng)
            try:
                sh = float(_mcpt_stat(spec, p, bench_rel))
            except Exception as e:
                sh = ("err", f"{type(e).__name__}: {str(e)[:100]}")
            if _ingest(sh):
                stopped = True
                break

    if stopped:                                      # fail certain — compute stopped early
        pval = (exceed + 1) / (len(perms) + 1)
        res = {"n_ran": len(perms), "early_stop": True, "benchmark_relative": bench_rel,
               "real_stat": round(real_stat, 3), "perm_mean": round(float(np.mean(perms)), 3),
               "perm_max": round(float(np.max(perms)), 3), "p_value_lb": round(pval, 4), "pass": False}
        print(f"[mcpt] EARLY FAIL after {len(perms)} perms ({exceed} >= real stat {real_stat:.2f}, "
              f"bench_rel={bench_rel}, workers={n_workers})")
        return res, False
    if not perms:
        return {"note": "all perms errored — inconclusive, treating as FAIL"}, False
    arr = np.array(perms)
    pval = float((np.sum(arr >= real_stat) + 1) / (len(arr) + 1))
    res = {"n_ran": len(perms), "early_stop": False, "benchmark_relative": bench_rel,
           "real_stat": round(real_stat, 3), "perm_mean": round(float(arr.mean()), 3),
           "perm_p95": round(float(np.percentile(arr, 95)), 3), "perm_max": round(float(arr.max()), 3),
           "p_value": round(pval, 4), "pass": pval <= 0.05}
    print(f"[mcpt] {len(perms)} perms ({n_workers} workers) | real stat {real_stat:.2f} "
          f"(bench_rel={bench_rel}) | perm mean {arr.mean():.2f} | p={pval:.4f} -> "
          f"{'PASS' if res['pass'] else 'FAIL'}")
    return res, bool(res["pass"])


def _stage2_generalize(spec: StrategySpec) -> tuple:
    """STAGE-2 fluke-confirmation for BROAD-scope stage-1 passes: run the SAME frozen signal (default
    params, no re-search) on each PRE-DECLARED untouched universe, score ONLY the holdout window, and
    apply the breadth verdict (>=60% of >=3 universes positive OOS). This is the automated version of
    forward/generalize.py — legitimate to run same-night because the universes were frozen in the
    pre-registration and only their untouched holdouts are read (BAB lesson: breadth, never one slice).
    Returns (results: dict|None, confirmed: bool, note: str)."""
    loader = getattr(spec, "load_gen_data", None)
    unis = list(getattr(spec, "generalization_universes", []) or [])
    if not loader or len(unis) < 3:
        return None, False, (f"stage-2 NOT runnable (load_gen_data={'set' if loader else 'missing'}, "
                             f"{len(unis)} universes declared, need >=3) -> manual battery required "
                             f"(forward/generalize.py)")
    def _one_universe(u):
        r_u = pd.Series(spec.signal(loader(u), **spec.default_params)[0]).dropna()
        h_u = r_u[r_u.index >= spec.holdout_start]
        return round(_sharpe(h_u), 2) if len(h_u) > 20 else None

    results = {}
    # universes are independent (own loader call each) -> parallel; capped at 3 (panel RAM)
    for u, r in _fork_map(_one_universe, unis, workers=min(_mcpt_workers(), 3)):
        if isinstance(r, tuple) and r and r[0] == "err":
            results[u] = None
            print(f"[stage2] universe {u} failed: {r[1]}")
        else:
            results[u] = r
    vals = [v for v in results.values() if v is not None]
    pos = sum(1 for v in vals if v > 0)
    if len(vals) < 3:
        return results, False, f"stage-2 INCONCLUSIVE: only {len(vals)}/{len(unis)} universes ran (need >=3)"
    frac = pos / len(vals)
    if frac >= 0.60:
        return results, True, f"stage-2 CONFIRMED: {pos}/{len(vals)} untouched universes positive OOS ({frac:.0%}) -> generalises"
    return results, False, f"stage-2 REJECTED: {pos}/{len(vals)} positive OOS ({frac:.0%} < 60%) -> overfit outlier (cf. BAB)"


def run_experiment(spec: StrategySpec, write_wiki=True, alert=True) -> dict:
    """Run one pre-registered hypothesis through ALL rails. Returns the verdict dict."""
    panel = spec.load_data()
    full_ret, trades = spec.signal(panel, **spec.default_params)
    full_ret = pd.Series(full_ret).dropna()
    search = full_ret[full_ret.index < spec.holdout_start]
    s_sh = _sharpe(search)

    # TIER-0 SCREEN (efficiency + holdout discipline): no in-sample edge -> SCREEN_FAIL. Skip the expensive
    # grid + CPCV AND do NOT burn the write-once holdout -- only an earned in-sample edge gets the OOS look.
    if abs(s_sh) < SCREEN_FLOOR:
        with FileLock("fdr-registry", ttl=120):
            n_fam = ri.distinct_families(extra=spec.family)
            bar = ri.promote_dsr(n_fam)
            registry_recorded = True
            try:
                ri.registry.append_run({"strategy": spec.id, "family": spec.family, "tier": "SCREEN_FAIL",
                    "dsr": None, "promote_dsr": bar, "n_families": n_fam, "holdout_touched": False, "passed_all": False})
            except Exception as e:  # FDR-bar integrity: NEVER silent (a frozen bar = multi-agent false-discovery risk)
                registry_recorded = False
                print(f"[harness] REGISTRY APPEND FAILED ({spec.id}): {e} -- FDR bar is NOT rising; investigate now")
        verdict = {
            "id": spec.id, "family": spec.family, "title": spec.title, "markets": spec.markets,
            "tier": "SCREEN_FAIL", "promote_bar": round(bar, 3), "n_families": n_fam,
            "dsr": None, "median_cpcv": None, "pbo": None, "deployment_passed": None,
            "deploy_peak": None, "deploy_sectors": None, "deploy_reasons": None,
            "search_sharpe": round(s_sh, 3), "holdout_sharpe": None, "holdout_pass": False,
            "holdout_reasons": [f"tier-0 screen: |search Sharpe| {abs(s_sh):.2f} < {SCREEN_FLOOR} -- no in-sample edge; full rails + holdout skipped"],
            "full_sharpe": round(_sharpe(full_ret), 3), "full_maxdd": round(_maxdd(full_ret), 3), "n_trades": len(trades),
            "stage1_pass": False, "confirmed": False, "scope": getattr(spec, "scope", "broad"),
            "needs_confirmation": None, "generalization": None, "generalization_note": None,
            "PASSED_ALL_GATES": False, "registry_recorded": registry_recorded,
            **_provenance(spec),  # O1: even a SCREEN_FAIL page must say WHICH code failed
        }
        if write_wiki:
            from sdk.wiki import write_experiment
            write_experiment(spec, verdict)
        return verdict

    # earned the full rails + the OOS look.
    # WRITE-ONCE HOLDOUT ENFORCEMENT (invariant #4): one OOS look per frozen config, EVER.
    # The ledger is the refusal mechanism, not just a record -- a second look at the same
    # quarantined slice converts "out-of-sample" into "in-sample with extra steps".
    cfg = dict(spec.default_params)
    if spec.hedge_tickers:   # the hedge sleeve is part of the frozen design -> part of the hash
        cfg["__hedge"] = {"tickers": sorted(spec.hedge_tickers), "cap": spec.hedge_cap}
    cfg_hash = ri.config_hash(spec.id, cfg, str(spec.markets))
    prior = ri.ledger_lookup(cfg_hash)
    holdout_burned = prior is not None
    if holdout_burned:
        print(f"[harness] HOLDOUT ALREADY BURNED for {spec.id} (hash {cfg_hash}, "
              f"first look {prior.get('ts', '?')}) -- refusing a second OOS read; "
              f"holdout gate forced FAIL")
    holdout = full_ret[full_ret.index >= spec.holdout_start]
    grid = {}
    grid_items = list((spec.grid or {"default": {}}).items())
    nondefault = [(label, kw) for label, kw in grid_items if kw]

    def _one_variant(item):
        label, kw = item
        r = pd.Series(spec.signal(panel, **{**spec.default_params, **kw})[0]).dropna()
        return r[r.index < spec.holdout_start]   # keep the Series (index) so PBO/DSR align variants by DATE

    for label, kw in grid_items:
        if not kw:  # the mandated "default": {} variant == the line-1 run; reuse it (one full backtest saved per run)
            grid[label] = search
    # independent re-runs of the frozen signal -> parallel (capped: each holds a returns series only,
    # but the signal itself may allocate panel-sized temporaries)
    for (label, _), r in _fork_map(_one_variant, nondefault, workers=min(_mcpt_workers(), 4)):
        if isinstance(r, tuple) and len(r) and r[0] == "err":
            raise RuntimeError(f"grid variant '{label}' failed: {r[1]}")  # grid is part of the frozen design
        grid[label] = r

    # --- the gates (non-bypassable) ---
    # I4: stage-1 gates consume SEARCH-WINDOW trades only. The full ledger includes
    # holdout-period trades, so concentration/regime/LOO gates were silently computed on
    # quarantined data BEFORE the single OOS look was earned — a leak from the holdout into
    # stage-1, and double-dipping the slice the write-once ledger protects.
    # Membership by ENTRY date: exit-date filtering zeroes the ledger for low-turnover books
    # whose holds straddle the boundary (verified live — empty ledger -> deployment gate fails
    # mechanically, not on merit). Boundary-straddling trades carry some post-boundary PnL;
    # accepted: the protected resource is the holdout RETURN series (gated by the write-once
    # ledger), and entry-time membership uses no holdout information to SELECT trades.
    search_trades = [t for t in trades if str(t.get("entry_date", "")) < spec.holdout_start]
    result = ri.assemble_bundle(search.values, search_trades, grid_returns=grid)
    # assemble_bundle returns {"bundle": <gate inputs>, "diagnostics": ...}. evaluate_tiers AND the
    # verdict need the INNER bundle — passing the wrapper made EVERY gate 'missing' -> tier ALWAYS FAIL
    # (every forge run failed on this, not on merit). This is THE gate-wiring fix.
    b = (result.get("bundle") or {}) if isinstance(result, dict) else {}
    # O1: keep the diagnostics assemble_bundle already computed (previously discarded) — they're
    # the reviewer's evidence: PBO detail, effective DSR trials, concentration, regime, LOO.
    diag = (result.get("diagnostics") or {}) if isinstance(result, dict) else {}
    # deployment sanity on search-window trades too: a strategy that only reaches >=50 trades /
    # sector spread by counting holdout-period activity hasn't demonstrated it in-sample.
    dep = ri.deployment_sanity(search_trades, strategy_meta={"max_positions": spec.deploy_max_positions},
                               hedge_tickers=spec.hedge_tickers, hedge_cap=spec.hedge_cap)

    h_sh = _sharpe(holdout)  # s_sh computed above (survived the tier-0 screen)
    deg = (h_sh - s_sh) / abs(s_sh) * 100 if abs(s_sh) > 0.1 else None
    h_pass, h_reasons = ri.holdout_gate(h_sh, deg, dep["passed"])
    if holdout_burned:
        h_pass = False
        h_reasons = list(h_reasons) + [
            f"WRITE-ONCE VIOLATION: holdout already burned for config {cfg_hash} "
            f"(first look {prior.get('ts', '?')}); this re-read cannot count as out-of-sample"]
    else:
        try:
            ri.ledger_append({"config_hash": cfg_hash, "strategy": spec.id, "family": spec.family,
                              "ts": pd.Timestamp.now().isoformat(),
                              "holdout_start": spec.holdout_start, "holdout_sharpe": round(h_sh, 4)})
        except Exception as e:
            print(f"[harness] WARNING: holdout ledger append failed ({e}) -- "
                  f"write-once enforcement degraded for {spec.id}")
    # SEARCH-SANITY: a holdout "pass" is meaningless if there was no in-sample edge to confirm.
    # Guards the degenerate case search_sharpe~=0 -> degradation blows up -> spurious holdout pass
    # (this falsely flagged a credit-carry book that the trend over-blend had sunk to ~0 as a near-miss).
    if abs(s_sh) < 0.3:
        h_pass = False
        h_reasons = list(h_reasons) + [f"search Sharpe {s_sh:.2f} < 0.3 — no in-sample edge to confirm"]

    # --- FDR accounting + registry append: ATOMIC across all agents (multi-agent safe).
    #     The heavy CPCV (assemble_bundle) ran above OUTSIDE the lock; here we serialize only
    #     count -> bar -> evaluate -> append (milliseconds). This is what keeps the shared FDR
    #     bar correct when N agents test in parallel -- the non-negotiable Phase-3 invariant. ---
    # long-only BETA-CONFOUND check: decompose search returns into universe-beta + selection-alpha.
    # MCPT's lesson made a cheap stage-1 gate: an apparent edge that is really long-only universe beta
    # (high beta, weak selection-alpha) must NOT pass -- it'd just be holding the universe (cf. value×mom 0.994).
    decomp = _beta_decomp(search, _price_matrix(panel)) or {}
    beta_confound = bool(decomp.get("beta_to_universe", 0) > BETA_HI
                         and decomp.get("selection_alpha_sharpe") is not None
                         and decomp["selection_alpha_sharpe"] < SEL_FLOOR)

    with FileLock("fdr-registry", ttl=120):
        n_fam = ri.distinct_families(extra=spec.family)
        bar = ri.promote_dsr(n_fam)
        tiers = ri.evaluate_tiers(b, promote_dsr=bar)
        tier = str(tiers.get("tier"))
        stage1_pass = tier.upper() == "PROMOTE" and h_pass and dep["passed"]
        if stage1_pass and beta_confound:   # demote: edge is long-only universe beta, not the signal
            stage1_pass = False
            h_reasons = list(h_reasons) + [
                f"BETA-CONFOUND: beta_to_universe {decomp['beta_to_universe']} + selection-alpha Sharpe "
                f"{decomp['selection_alpha_sharpe']} < {SEL_FLOOR} -> edge is long-only universe beta, demoted"]
        registry_recorded = True
        try:
            ri.registry.append_run({"strategy": spec.id, "family": spec.family, "tier": tier,
                                    "dsr": b.get("dsr"), "promote_dsr": bar, "n_families": n_fam,
                                    "holdout_touched": True, "passed_all": stage1_pass})
        except Exception as e:  # FDR-bar integrity: NEVER silent
            registry_recorded = False
            print(f"[harness] REGISTRY APPEND FAILED ({spec.id}): {e} -- FDR bar is NOT rising; investigate now")

    # POLICY: a stage-1 pass is a CANDIDATE. PASSED_ALL_GATES requires INDEPENDENT fluke-confirmation.
    # BROAD scope: the cross-universe battery runs HERE, same-night, on the PRE-DECLARED untouched
    # universes (frozen in the spec before any data was touched -> still pre-registered, not mining).
    # LOCAL scope: confirmation is forward-validation (calendar time) -> stays candidate; the worker
    # auto-deploys it to shadow-paper so the forward track starts immediately.
    gen_results, gen_confirmed, gen_note = None, False, None
    mcpt_res, mcpt_pass = None, None
    if stage1_pass:
        # MCPT runs FIRST for every stage-1 pass (broad AND local): construction artifacts replicate
        # across universes (breadth blind) and would auto-deploy as local candidates. Cheap vs the cost
        # of a false PASS; early-stops on certain failure.
        print("[stage2] stage-1 PASS -> running MCPT (permutation test) first...")
        mcpt_res, mcpt_pass = _stage2_mcpt(spec, panel, _sharpe(full_ret),
                                           beta_to_universe=(decomp or {}).get("beta_to_universe"))
        if getattr(spec, "scope", "broad") == "broad":
            if mcpt_pass:
                print("[stage2] MCPT pass -> running cross-universe generalization battery...")
                gen_results, gen_confirmed, gen_note = _stage2_generalize(spec)
                print(f"[stage2] {gen_note}")
            else:
                gen_note = "stage-2 REJECTED at MCPT: edge reproduces on permuted (structureless) data -> construction artifact, breadth skipped"
                print(f"[stage2] {gen_note}")
        else:
            gen_note = ("local scope -> confirmation = forward-validation (auto-deploying to shadow paper)"
                        if mcpt_pass else
                        "local scope but MCPT FAIL -> construction artifact, NOT deploying to paper")
    passed_all = bool(stage1_pass and mcpt_pass and gen_confirmed)

    grid_sharpes = {label: round(_sharpe(r), 3) for label, r in grid.items()}

    verdict = {
        "id": spec.id, "family": spec.family, "title": spec.title, "markets": spec.markets,
        "tier": tier, "promote_bar": round(bar, 3), "n_families": n_fam,
        "dsr": b.get("dsr"), "median_cpcv": b.get("median_cpcv_sharpe"), "pbo": b.get("pbo"),
        "deployment_passed": dep["passed"], "deploy_peak": dep.get("peak_concurrent"),
        "deploy_sectors": dep.get("sector_spread"), "deploy_reasons": dep.get("forced_fail_reasons"),
        "hedge_share": dep.get("hedge_share"), "hedge_tickers": dep.get("hedge_tickers"),
        "search_sharpe": round(s_sh, 3), "holdout_sharpe": round(h_sh, 3),
        "holdout_pass": h_pass, "holdout_reasons": h_reasons,
        "full_sharpe": round(_sharpe(full_ret), 3), "full_maxdd": round(_maxdd(full_ret), 3),
        "n_trades": len(trades),
        "beta_to_universe": decomp.get("beta_to_universe"),
        "selection_alpha_sharpe": decomp.get("selection_alpha_sharpe"), "beta_confound": beta_confound,
        "stage1_pass": stage1_pass, "confirmed": gen_confirmed, "scope": getattr(spec, "scope", "broad"),
        "mcpt": mcpt_res, "mcpt_pass": mcpt_pass,
        "needs_confirmation": (None if not stage1_pass or passed_all else
            ("cross-market-generalization" if getattr(spec, "scope", "broad") == "broad" else "forward-validation")),
        "generalization": gen_results, "generalization_note": gen_note,
        "PASSED_ALL_GATES": passed_all, "registry_recorded": registry_recorded,
        "holdout_burned": holdout_burned, "config_hash": cfg_hash,
        # O1 — reproducibility + gate diagnostics
        **_provenance(spec),
        "grid_sharpes": grid_sharpes,
        "diagnostics": {k: diag.get(k) for k in (
            "pbo", "dsr_grid", "dsr_source", "dsr_n_trials_raw", "dsr_n_trials_effective",
            "grid_participation_ratio", "search_burden", "ticker_concentration",
            "regime", "n_obs", "cpcv") if k in diag},
    }

    if write_wiki:
        from sdk.wiki import write_experiment
        write_experiment(spec, verdict)
    if alert and passed_all:
        from sdk.notify import telegram_pass
        telegram_pass(spec, verdict)
    elif alert and stage1_pass:
        from sdk.notify import telegram_candidate
        telegram_candidate(spec, verdict)
    return verdict
