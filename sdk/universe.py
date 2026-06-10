"""sdk/universe.py — sector-spread universe construction (the ~10-line loop that was
copy-pasted across 32 strategy/forward files, each a fresh chance for divergence)."""
from __future__ import annotations

from sdk.adapters import us_universe

# Sharadar sector labels (the 11 GICS-ish sectors used across all experiments)
SECTORS = [
    "Basic Materials", "Communication Services", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
    "Industrials", "Real Estate", "Technology", "Utilities",
]


def sector_universe(marketcap: str, top_n_per_sector: int,
                    sectors: list | None = None) -> tuple[list, dict]:
    """Survivorship-clean universe with enforced sector spread.

    Returns (tickers, sector_map): the top_n_per_sector most-liquid names per sector at the
    given cap tier, plus {ticker: sector} for the trade ledger (deployment-sanity needs it).
    """
    tickers, sector_map = [], {}
    for sec in (sectors or SECTORS):
        names = us_universe(sector=sec, marketcap=marketcap, top_n=top_n_per_sector)
        for t in names:
            if t not in sector_map:          # first sector wins on the rare dual-listed name
                tickers.append(t)
                sector_map[t] = sec
    return tickers, sector_map
