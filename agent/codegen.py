"""Signal codegen: the agent writes a complete StrategySpec module from its proposal,
with a bounded fix-retry loop (reads the traceback, repairs the code). Claude Max OAuth ($0)."""
import json, re, subprocess
from agent.propose import _assistant_text
from agent.config import pi_cmd

SYS = "You are Claude Code, Anthropic's official CLI for Claude."

CONTRACT = '''
You are writing a Python strategy module for the Hephaestus research harness. Output ONLY one
```python code block: a complete module with NO external side effects (no file writes, no capital,
no config). It MUST define exactly:

  def load_data() -> pd.DataFrame:        # the panel signal() consumes (use the adapters below)
  def signal(panel, **params) -> (pd.Series daily_returns, list trades):
  SPEC = StrategySpec(id=..., family=..., title=..., markets=[...], data_desc=..., pre_registration=...,
                      load_data=load_data, signal=signal, default_params={...}, grid={label:params,...},
                      scope='broad'|'local', generalization_universes=[...],
                      holdout_start="2022-01-01", deploy_max_positions=N)

CONTRACT:
- daily_returns: a pandas Series of daily net-of-cost portfolio returns, DatetimeIndex, name set.
- trades: list of dicts, each {"ticker","sector","entry_date"(YYYY-MM-DD str),"exit_date","hold_days"(int),
  "position_value"(float),"pnl"(float)} — used for deployment-sanity (needs >=50 trades, spread across
  sectors, no single name >40% of position-days). For a factor book, emit one trade per held position run.
- grid: a few pre-declared param variants for the DSR effective-N (honest search burden); "default"={} is primary.
- scope: 'broad' if the edge is a UNIVERSAL mechanism (a factor/premium theory says appears across markets ->
  a stage-1 pass MUST later GENERALISE to other untouched universes, or it's an overfit outlier like BAB);
  'local' if it's a defensibly UNIVERSE-SPECIFIC edge (then forward-validation confirms it). For 'broad' equity
  factors set generalization_universes to untouched slices to confirm in, e.g. ['large','small','sectors'].
- Apply realistic costs (~8bps on turnover). Inverse-vol size. Weekly rebalance. NO look-ahead (lag signals 1 day).

USE ONLY these tested imports (do NOT download raw / reinvent). Full data inventory: research-wiki/DATA_CATALOG.md.
  from sdk.harness import StrategySpec
  from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, carry_returns, inv_vol_position
  import numpy as np, pandas as pd
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
- carry_returns() -> the crypto funding-carry near-miss leg (daily returns Series).
- inv_vol_position(signal_df, rets, target_vol, vol_lb, max_pos, rebalance) -> weekly-held lagged positions.
- For a COMBINATION: build each leg's daily returns, align (pd.concat axis=1 dropna), vol-match, blend.
- BUT test the premium STANDALONE first. Only ADD a hedge (e.g. trend) if it CUTS THE TAIL without
  diluting the standalone Sharpe — size it to MINIMISE drag, NOT a reflexive 50/50. A ~0-Sharpe hedge
  blended 50/50 HALVES the edge (it sank a real +0.27-Sharpe credit-carry premium to ~0). Prefer the
  standalone leg or a SMALL tail-overlay; do not pair with trend just because the wiki pattern says so.
Be economical and correct. OWNED/FREE data only (see DATA_CATALOG.md). The harness runs ALL the rails; you only produce returns+trades.
'''


def _pi(prompt: str) -> str:
    try:
        r = subprocess.run(pi_cmd(), input=prompt, capture_output=True, text=True, timeout=420)
        return _assistant_text(r.stdout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        return _assistant_text(out)


def _extract_code(text: str) -> str:
    # take the LARGEST fenced block (Opus sometimes emits a skeleton block first, then the real one)
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL) or re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    return (max(blocks, key=len) if blocks else (text or "")).strip()


def _looks_complete(code: str) -> bool:
    return "def signal" in code and len(code) > 300


def generate(proposal: dict) -> str:
    prompt = (f"{CONTRACT}\n\n=== PROPOSAL TO IMPLEMENT ===\n{json.dumps(proposal, indent=2)}\n\n"
              f"Write the COMPLETE module now as ONE ```python code block with the full implementation "
              f"(imports + def signal + any helpers). Do NOT emit a skeleton, outline, or partial block.")
    code = ""
    for _ in range(3):  # retry-on-empty at the SOURCE -> kills the wasted consistency/fix call per run
        code = _extract_code(_pi(prompt))
        if _looks_complete(code):
            return code
    return code  # last attempt; the run-loop fix() repairs if still incomplete


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
              f"GENERATED CODE:\n```python\n{code[:6500]}\n```\n\n"
              f"Does the code FAITHFULLY implement that thesis + frozen signal construction? Check specifically: "
              f"the actual computation matches the claimed mechanism/direction; point-in-time data (datekey, no "
              f"look-ahead); correct adjustments (splits, dividends, costs); the right universe; the signal sign. "
              f'Return ONLY JSON: {{"consistent": true|false, "issues": "specific claim-vs-code mismatches, or empty"}}')
    text = _pi(prompt)
    try:
        s, e = text.find("{"), text.rfind("}")
        d = json.loads(text[s:e + 1])
        return bool(d.get("consistent", True)), str(d.get("issues", ""))[:500]
    except Exception:
        return True, ""
