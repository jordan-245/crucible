"""Signal codegen: the agent writes a complete StrategySpec module from its proposal,
with a bounded fix-retry loop (reads the traceback, repairs the code). LLM via the pi CLI."""
import json, re
from agent.llm import call as _llm_call, extract_json

CONTRACT = '''
You are writing a Python strategy module for the Crucible research harness. Output ONLY one
```python code block: a complete module with NO external side effects (no file writes, no capital,
no config). It MUST define exactly:

  def load_data() -> pd.DataFrame:        # the panel signal() consumes (use the adapters below)
  def signal(panel, **params) -> (pd.Series daily_returns, list trades):
  def load_gen_data(label) -> pd.DataFrame:  # REQUIRED for scope='broad': the panel for ONE
      # generalization universe (same shape as load_data(); label is one of generalization_universes)
  SPEC = StrategySpec(id=..., family=..., title=..., markets=[...], data_desc=..., pre_registration=...,
                      load_data=load_data, signal=signal, default_params={...}, grid={label:params,...},
                      scope='broad'|'local', generalization_universes=[...], load_gen_data=load_gen_data,
                      holdout_start="2022-01-01", deploy_max_positions=N)

CONTRACT:
- daily_returns: a pandas Series of daily net-of-cost portfolio returns, DatetimeIndex, name set.
- trades: list of dicts, each {"ticker","sector","entry_date"(YYYY-MM-DD str),"exit_date","hold_days"(int),
  "position_value"(float),"pnl"(float),"entry_regime"(str, set by trades_from_weights)} — used for
  deployment-sanity + cross-regime gates (needs >=50 trades, spread across
  sectors, no single name >40% of position-days). For a factor book, emit one trade per held position run.
- grid: a few pre-declared param variants for the DSR effective-N (honest search burden); "default"={} is primary.
- scope: 'broad' if the edge is a UNIVERSAL mechanism (a factor/premium theory says appears across markets ->
  a stage-1 pass MUST GENERALISE to other untouched universes, or it's an overfit outlier like BAB);
  'local' if it's a defensibly UNIVERSE-SPECIFIC edge (then forward-validation confirms it).
- For 'broad': declare >=3 generalization_universes (DISJOINT from the search universe — different cap tier,
  different sectors, or sub-slices that share NO tickers) AND implement load_gen_data(label) returning each
  one's panel. The harness runs the STAGE-2 battery automatically on a stage-1 pass: same frozen signal +
  default params on each universe's HOLDOUT only; >=60% must be OOS-positive or the candidate is rejected.
  Keep each gen universe SMALL (~150-400 names) — it runs same-night, N+1 extra signal() calls.
- Apply realistic costs (~8bps on turnover). Inverse-vol size. Weekly rebalance. NO look-ahead (lag signals 1 day).

USE ONLY these tested imports (do NOT download raw / reinvent). Full data inventory: research-wiki/DATA_CATALOG.md.
  from sdk.harness import StrategySpec
  from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, inv_vol_position
  from sdk.universe import sector_universe
  from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
  import numpy as np, pandas as pd

MANDATORY KIT (do NOT re-implement these — every hand-rolled copy is a fresh chance for a
lookahead bug the harness can't see; the ONLY novel code in your module is the signal itself):
- sector_universe(marketcap, top_n_per_sector) -> (tickers, sector_map): sector-spread universe +
  the {ticker: sector} map the trade ledger needs. USE THIS instead of looping us_universe per sector.
- xs_zscore(df, winsor=(0.05,0.95)) -> cross-sectional per-date z-score, winsorized, NaN-preserving.
- net_of_cost(W, rets, cost_bps=8.0, name=...) -> daily net returns from a LAGGED weight matrix
  (pass W.shift(1) if you built same-day weights — the lag is YOUR responsibility, state it in the code).
- trades_from_weights(W, rets, sector_map) -> the CONTRACT trade ledger (run-length per held name),
  auto-stamping each trade's entry_regime (bull/bear x calm/vol, trailing-data-only) — the cross-regime
  robustness gates depend on it. Never write entry_regime yourself; the kit's labeller is the standard.
- market_regime(rets) -> the per-date regime label Series, if you need it for a regime FILTER in the
  signal itself (it is shift(1)-lagged — safe to act on same-day).
- pit_panel(sf1_df, field, dates, tickers) -> point-in-time fundamental panel (datekey-based, ffilled;
  never use calendardate — that's lookahead).
A typical module body is therefore: build universe -> load panels -> compute YOUR signal -> weights
-> W=...shift(1) -> net_of_cost + trades_from_weights -> return. Keep it SHORT.
- sep_panel(tickers, start, field='closeadj') -> SURVIVORSHIP-CLEAN US equity daily panel from OWNED Sharadar SEP
  (delisted incl, split+div adjusted). **PREFER over yf_panel for US stocks** (yfinance has survivorship bias).
- us_universe(sector=, category='Domestic Common Stock', marketcap=, include_delisted=True, top_n=) -> US ticker
  list from OWNED Sharadar TICKERS (delisted INCLUDED -> survivorship-clean). Pass its output to sep_panel.
  **CRITICAL for cross-sectional EQUITY: you MUST bound the universe — pass top_n (e.g. us_universe(..., top_n=1000)
  = 1000 most-liquid names) and/or marketcap='Large'/'Mid'. NEVER run the full ~16k common-stock universe through
  the rails — the CPCV is pathologically slow + OOM'd at 14.5GB. Target ~few-hundred to ~1500 liquid names.** Cross-sectional ANOMALIES (issuance, value, accruals, low-vol…) live in SMALL/ILLIQUID names — test
  them in us_universe(marketcap='Small' or 'Mid', top_n=~1500), NOT the largest liquid names (there they
  are arbitraged away -> false nulls).
- sf1(tickers, fields, dimension='ARQ') -> OWNED Sharadar fundamentals. Use 'datekey' (filing date) as the as-of
  date to avoid look-ahead (NEVER calendardate). Fields e.g. eps, revenue, bvps, marketcap, pe, de, roe.
- yf_panel(tickers, start) -> Close panel (FREE; futures/ETFs/intl indices — NOT US single stocks).
  fred_series({fred_id:col}, start) -> daily rates/yields/credit-spreads (FREE; e.g. BAMLH0A0HYM2=HY OAS).
- trend_returns(**p) -> (returns, trades) the validated 21-market CTA trend hedge leg.
- inv_vol_position(signal_df, rets, target_vol, vol_lb, max_pos, rebalance) -> weekly-held lagged positions.
- For a COMBINATION: build each leg's daily returns, align (pd.concat axis=1 dropna), vol-match, blend.
- BUT test the premium STANDALONE first. Only ADD a hedge (e.g. trend) if it CUTS THE TAIL without
  diluting the standalone Sharpe — size it to MINIMISE drag, NOT a reflexive 50/50. A ~0-Sharpe hedge
  blended 50/50 HALVES the edge (it sank a real +0.27-Sharpe credit-carry premium to ~0). Prefer the
  standalone leg or a SMALL tail-overlay; do not pair with trend just because the wiki pattern says so.
- HEDGE SLEEVE (broad-index-ETF beta trim, e.g. a residual IWM short): declare it on the spec —
  StrategySpec(hedge_tickers=["IWM"], hedge_cap=0.35) — so the deployment gate judges your ALPHA
  book alone and gates the sleeve on whitelist+cap. An UNDECLARED ETF hedge held continuously will
  force-fail single_name_share (it killed two otherwise-clean Amihud near-misses on 2026-06-11).
  Whitelist: SPY/IVV/VOO/IWM/MDY/IJH/IJR/QQQ/VTI/ITOT/EFA/EEM/ACWI/TLT/IEF/SHY/GLD/USO/DBC.
  The cap is position-day share; >0.60 is never allowed. The sleeve is part of the FROZEN design.
Be economical and correct. OWNED/FREE data only (see DATA_CATALOG.md). The harness runs ALL the rails; you only produce returns+trades.
'''


_pi = _llm_call  # plumbing consolidated in agent.llm


def _extract_code(text: str) -> str:
    # take the LARGEST fenced block (Opus sometimes emits a skeleton block first, then the real one)
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL) or re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    return (max(blocks, key=len) if blocks else (text or "")).strip()


def _looks_complete(code: str) -> bool:
    return "def signal" in code and len(code) > 300


# Observability: stats of the LAST generate() call, read by run_worker for the run record.
LAST_GEN = {"attempts": 0, "empty_retries": 0}


def generate(proposal: dict) -> str:
    prompt = (f"{CONTRACT}\n\n=== PROPOSAL TO IMPLEMENT ===\n{json.dumps(proposal, indent=2)}\n\n"
              f"Write the COMPLETE module now as ONE ```python code block with the full implementation "
              f"(imports + def signal + any helpers). Do NOT emit a skeleton, outline, or partial block.")
    code = ""
    LAST_GEN["attempts"] = 0
    for _ in range(3):  # retry-on-empty at the SOURCE -> kills the wasted consistency/fix call per run
        LAST_GEN["attempts"] += 1
        code = _extract_code(_pi(prompt))
        if _looks_complete(code):
            break
    LAST_GEN["empty_retries"] = LAST_GEN["attempts"] - 1
    return code  # if still incomplete after 3, the run-loop fix() repairs


def fix(code: str, traceback: str) -> str:
    prompt = (f"{CONTRACT}\n\nThe following module FAILED. Fix it; output ONLY the corrected ```python module.\n\n"
              f"=== CODE ===\n{code}\n\n=== ERROR ===\n{traceback[-2500:]}")
    return _extract_code(_pi(prompt))


def consistency_check(proposal: dict, code: str) -> tuple:
    """Verify the generated code FAITHFULLY implements the proposal's economic thesis (catches code that
    claims one mechanism but computes another — e.g. 'split-consistent' but uses raw shares). Fail-OPEN
    on a parse error (best-effort guard, not a hard gate). Returns (ok: bool, issues: str)."""
    claim = {k: proposal.get(k) for k in ("premium", "market", "signal_approach", "why_not_duplicate")}
    prompt = (f"PROPOSAL (the economic thesis the code MUST implement):\n{json.dumps(claim, indent=2)}\n\n"
              f"GENERATED CODE:\n```python\n{code[:30000]}\n```\n\n"  # full module (Fable writes 15-20K; a truncated view causes false 'code is truncated' verdicts -> wasted fix() calls)
              f"Does the code FAITHFULLY implement that thesis + frozen signal construction? Check specifically: "
              f"the actual computation matches the claimed mechanism/direction; point-in-time data (datekey, no "
              f"look-ahead); correct adjustments (splits, dividends, costs); the right universe; the signal sign. "
              f'Return ONLY JSON: {{"consistent": true|false, "issues": "specific claim-vs-code mismatches, or empty"}}')
    d = extract_json(_pi(prompt))
    if d is None:
        return True, ""  # fail-OPEN: best-effort guard, not a hard gate
    return bool(d.get("consistent", True)), str(d.get("issues", ""))[:500]
