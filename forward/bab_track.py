"""Forward-track for the validated BAB defensive premium — accrue post-discovery OOS evidence
BEFORE any capital (the search+holdout are historical; only the forward is incorruptible).

Freeze 2026-06-09; verdict ~2026-12-09. Recomputes the FROZEN BAB strategy on current SEP, records the
forward (post-freeze) equity + Sharpe to a ledger. Refreshes SEP if the cache is stale (>14d)."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/crucible")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "forward"))

FREEZE = "2026-06-09"
VERDICT_DATE = "2026-12-09"
LEDGER = ROOT / "forward" / "bab_ledger.jsonl"
from crucible_paths import DATA as _DATA
SEP_CACHE = _DATA / "cache" / "sep_long.parquet"


def _refresh_sep_if_stale(max_age_days=14):
    """Re-download Sharadar SEP + rebuild the cache if it's older than max_age_days (the forward
    period needs fresh data). Best-effort — skips on any failure (the run still records on old data)."""
    try:
        if SEP_CACHE.exists() and (datetime.now().timestamp() - SEP_CACHE.stat().st_mtime) < max_age_days * 86400:
            return
        subprocess.run([sys.executable, "scripts/sharadar_download.py", "SEP"],
                       cwd=str(_DATA.parent), timeout=2400, check=True)
        SEP_CACHE.unlink(missing_ok=True)  # force the adapter to rebuild from the fresh zip
        from sdk.adapters import _sep_cache
        _sep_cache()
    except Exception as e:
        print(f"[bab-forward] SEP refresh skipped: {e}")


def track():
    _refresh_sep_if_stale()
    import importlib
    import numpy as np
    import pandas as pd
    m = importlib.import_module("defensive_bab_frozen")
    ret, _ = m.signal(m.load_data(), **m.SPEC.default_params)
    ret = pd.Series(ret).dropna()
    fwd = ret[ret.index >= FREEZE]

    def sh(r):
        r = pd.Series(r).dropna()
        return round(float(r.mean() / r.std() * np.sqrt(252)), 3) if len(r) > 5 and r.std() > 0 else None

    rec = {"ts": datetime.now().isoformat(), "freeze": FREEZE, "verdict_date": VERDICT_DATE,
           "last_data": str(ret.index[-1].date()) if len(ret) else None, "fwd_days": int(len(fwd)),
           "fwd_sharpe": sh(fwd), "fwd_cum_return": round(float((1 + fwd).prod() - 1), 4) if len(fwd) else 0.0,
           "backtest_search_sharpe": 1.22, "holdout_sharpe": 0.80}
    LEDGER.parent.mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[bab-forward] {rec['fwd_days']}d forward | Sharpe {rec['fwd_sharpe']} | "
          f"cum {rec['fwd_cum_return']:.2%} | last data {rec['last_data']} | verdict {VERDICT_DATE}")
    # Telegram nudge once we have a month of forward data
    if rec["fwd_days"] >= 21 and rec["fwd_sharpe"] is not None:
        try:
            from sdk.notify import telegram_msg
            telegram_msg(f"📈 BAB forward-track: {rec['fwd_days']}d, Sharpe {rec['fwd_sharpe']}, "
                         f"cum {rec['fwd_cum_return']:.1%} (backtest 1.22 / holdout 0.80). Verdict {VERDICT_DATE}.")
        except Exception:
            pass
    return rec


if __name__ == "__main__":
    track()
