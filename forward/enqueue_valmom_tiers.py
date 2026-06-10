"""Queue the per-tier re-tune+re-validate Value×Momentum books for a TRUE multi-tier portfolio.

2026-06-09 CPCV-hardened generalization proved: the V+M MECHANISM is real in every cap tier
(OOS-positive, CPCV-median-positive, 100% positive paths) BUT a config tuned on one tier does NOT
transfer (PBO 0.71-0.97 out-of-tier; low ONLY on its own discovery tier). To SCALE the edge we need
a tier-NATIVE book per cap tier — grid searched ON that tier so the default ranks robustly there.
Small-cap is already clean (trend_overlay, PBO 0.078). Queue the LARGE and MID native books here.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from sdk import queue

# Frozen construction = the VALIDATED small-cap trend_overlay (smith2_96154), parameterised by cap tier.
CONSTRUCTION = (
    "FROZEN construction (reproduce the validated small-cap trend-overlay book at this cap tier): "
    "Universe = US Domestic Common Stock at marketcap='{TIER}', survivorship-clean (include_delisted=True), "
    "built per-sector across the 11 GICS-style sectors via us_universe(sector=, marketcap='{TIER}', top_n~130) "
    "then bounded (~1.0-1.3k liquid names). Prices = sep_panel closeadj. Value = point-in-time book-to-market "
    "(sf1 ARQ 'bvps' forward-filled from its 'datekey' filing date / split-adj close — NO look-ahead). "
    "Momentum = 12-1 (px.shift(21)/px.shift(252)-1). Winsorise both at 5/95 pct then daily cross-sectional "
    "z-score. Composite = 50/50 blend (w_value=0.5, w_mom=0.5). Long-only TOP TERCILE (score >= 2/3 quantile), "
    "inverse-vol sized (vol_lb~63d). Weekly rebalance (last trading day of each ISO week) with a no-trade "
    "HYSTERESIS band (keep held names whose score >= 2/3 - band). Broad-market TREND overlay: equal-weight "
    "universe index, risk-OFF to cash when index < its trailing MA (trend_ma~200d), overlay lagged 1 day. "
    "All signals lagged 1 day; ~8bps turnover cost. "
    "CRITICAL — the grid (value_only, mom_only, no_trend, tight_band, and a trend_ma length variant) is searched "
    "ON THIS TIER so PBO/DSR are measured on the tier's OWN data; the goal is a default config that is ROBUST "
    "(low PBO) on its native tier, NOT a config lifted from another tier."
)

WHY = (
    "2026-06-09 CPCV-hardened generalization (forward/valmom_generalization_cpcv.jsonl) proved the V+M MECHANISM "
    "generalises across ALL cap tiers (OOS-positive, CPCV-median-positive, 100% positive paths) but a config "
    "tuned on Small/Mid does NOT transfer to {TIER} (PBO 0.71-0.97 out-of-tier; low only on own discovery tier). "
    "This is the {TIER}-cap-NATIVE book — grid searched on {TIER} so the default ranks robustly on its own tier — "
    "the missing leg of a true multi-tier V+M portfolio (Small trend_overlay PBO 0.078 already validated)."
)

def proposal(tier: str, others: list[str]) -> dict:
    return {
        "title": f"Value × Momentum + trend-overlay — {tier.upper()}-cap TIER-NATIVE re-tune (multi-tier scale-out leg)",
        "premium": "Value (point-in-time book-to-market) + Momentum (12-1) complementary equity factor premia, 50/50 composite with a broad-market trend overlay",
        "market": f"US {tier}-cap equities (Sharadar SEP/SF1, survivorship-clean incl. delisted)",
        "data_source": f"Sharadar SEP closeadj + SF1 ARQ bvps (datekey-lagged); us_universe(marketcap='{tier}')",
        "free_or_owned": "owned",
        "signal_approach": CONSTRUCTION.replace("{TIER}", tier),
        "why_not_duplicate": WHY.replace("{TIER}", tier),
        "prior": "medium",
        "pairs_with": "the validated Small-cap trend_overlay book + the other tier-native books -> 3-tier V+M portfolio",
        "gate0_data_check": f"us_universe(marketcap='{tier}', include_delisted=True) returns ~1k names; sf1 'bvps' coverage adequate on this tier",
        "crowding_risk": "high — V+M is published/crowded (esp. large-cap); the real test is whether a TIER-NATIVE construction still clears CPCV/PBO/DSR on this tier's OWN OOS",
        "scope": "broad",
        "generalization_plan": f"broad: the multi-tier portfolio itself is the generalization evidence — confirm the mechanism holds with low own-tier PBO here, then combine with the {', '.join(others)} books",
    }

if __name__ == "__main__":
    items = [
        proposal("Large", ["Mid", "Small"]),
        proposal("Mid", ["Large", "Small"]),
    ]
    for p in items:
        qid = queue.enqueue(p)
        print(f"enqueued {qid}: {p['title']}")
    print("queue stats:", queue.stats())
