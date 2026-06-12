"""Tested data adapters the agent's generated signal code composes (reliability > reinvention).
All no-incremental-cost / already-owned sources. The agent is told to use THESE, not raw downloads."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


def _day_cache(kind: str, key_parts):
    """Per-day parquet cache path for network adapters (yfinance/FRED): the same request twice in
    one day (3 smiths, batteries, MCPT setup) = one download. Keyed by request + date, so it's
    never stale by more than a day and never wrong. Returns None if the cache dir is unwritable."""
    import hashlib
    from crucible_paths import DATA
    h = hashlib.sha256(repr(sorted(map(str, key_parts))).encode()).hexdigest()[:16]
    d = os.path.join(str(DATA), "cache", "net")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return os.path.join(d, f"{kind}_{h}_{pd.Timestamp.today():%Y%m%d}.parquet")


def yf_panel(tickers, start="2005-01-01") -> pd.DataFrame:
    """Close panel for a list of yfinance tickers (business-day grid). Handles single/multi shapes.
    Day-cached on disk (E4): repeated calls across smiths/batteries hit Yahoo once per day."""
    import yfinance as yf
    cache = _day_cache("yf", [*tickers, start])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    raw = yf.download(list(tickers), start=start, progress=False, group_by="ticker", auto_adjust=False)
    cols = {}
    for t in tickers:
        try:
            s = raw[t]["Close"].dropna() if t in raw.columns.get_level_values(0) else None
        except Exception:
            s = None
        if s is None:
            try:
                c = raw["Close"]; s = (c[t] if t in getattr(c, "columns", []) else c).dropna()
            except Exception:
                s = None
        if s is not None and len(s) > 200:
            cols[t] = s
    panel = pd.DataFrame(cols).sort_index()
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(bidx).ffill(limit=3)
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def fred_series(series_ids, start="2005-01-01") -> pd.DataFrame:
    """Daily-ffilled FRED series panel. series_ids: dict {fred_id: column_name} or list.
    Day-cached on disk (E4)."""
    import urllib.request
    from crucible_paths import SECRETS
    cache = _day_cache("fred", [*(series_ids if isinstance(series_ids, (list, tuple))
                                  else sorted(series_ids.items())), start])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    key = os.environ.get("FRED_API_KEY") or json.load(open(SECRETS))["fred_api_key"]
    if isinstance(series_ids, (list, tuple)):
        series_ids = {s: s for s in series_ids}
    out = {}
    for sid, col in series_ids.items():
        u = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
             f"&api_key={key}&file_type=json&observation_start={start}")
        d = json.load(urllib.request.urlopen(u, timeout=40))
        out[col] = pd.Series({pd.Timestamp(o["date"]): float(o["value"])
                              for o in d["observations"] if o["value"] != "."}).sort_index()
    df = pd.DataFrame(out).sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="B")).ffill()
    if cache:
        try:
            tmp = cache + ".tmp"
            df.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return df


def trend_returns(**params):
    """The validated Boreas 21-market TSMOM stream (daily_returns, trades) — a ready hedge leg."""
    sys.path.insert(0, os.environ.get("BOREAS_RESEARCH", "/root/boreas/research"))
    from tsmom import run_tsmom
    return run_tsmom(**params)


# carry_returns() removed 2026-06-10: Midas project killed, its data pipeline is gone.
# The carry+trend STRUCTURE remains validated knowledge (see wiki) — re-add only with a
# new carry leg + fresh forward validation.

def inv_vol_position(signal_df, rets, target_vol=0.10, vol_lb=60, max_pos=2.0, rebalance="W-FRI"):
    """Standard inverse-vol sizing + weekly hold + 1d lag (no look-ahead). Reusable building block."""
    vol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252)
    raw = (signal_df * (target_vol / vol.replace(0, np.nan))).clip(-max_pos, max_pos)
    return raw.resample(rebalance).last().reindex(rets.index, method="ffill").shift(1).fillna(0.0)


# ── Owned Sharadar — SURVIVORSHIP-CLEAN US equities (PREFER over yf_panel for US stocks) ──
from crucible_paths import DATA
SHARADAR_DIR = str(DATA / "sharadar")
_CACHE_DIR = str(DATA / "cache")


def _stale(out: str, src: str) -> bool:
    """Cache invalidation (E4): a fresh source drop must rebuild the derived parquet."""
    return (not os.path.exists(out)) or (os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(out))


def _sep_cache() -> str:
    """Build/load a cached long parquet of owned Sharadar SEP (one-time ~1-2min build).
    Rebuilds automatically when SEP.zip is newer than the cache (fresh data drop)."""
    import zipfile
    # v2: + close/closeunadj (ex-div inference needs an unadjusted companion — runtime_error 2026-06-10).
    # Versioned filename forces a rebuild on schema change (_stale only checks mtime, not columns).
    out = os.path.join(_CACHE_DIR, "sep_long_v2.parquet")
    src = os.path.join(SHARADAR_DIR, "SEP.zip")
    if _stale(out, src):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        z = zipfile.ZipFile(src)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, usecols=["ticker", "date", "close", "closeadj", "closeunadj", "volume"],
                             parse_dates=["date"])
        tmp = out + ".tmp"
        df.sort_values("ticker").to_parquet(tmp, index=False, row_group_size=500_000)
        os.replace(tmp, out)
    return out


def sep_panel(tickers=None, start="2000-01-01", end=None, field="closeadj") -> pd.DataFrame:
    """SURVIVORSHIP-CLEAN US equity daily panel from OWNED Sharadar SEP (delisted incl, split+div adj).
    Wide DataFrame: business-day DatetimeIndex x ticker of `field`
    (closeadj=adjusted close | closeunadj | close | volume).
    PREFER THIS over yf_panel for US stocks (yfinance has survivorship bias — a wiki anti-pattern).
    tickers=None loads ALL (~16k, heavy) — pass a list (e.g. from us_universe())."""
    path = _sep_cache()
    # E3: push date bounds into the parquet scan (row-group pruning) instead of filtering in pandas
    filt = [("date", ">=", pd.Timestamp(start))]
    if end is not None:
        filt.append(("date", "<=", pd.Timestamp(end)))
    if tickers is not None:
        filt.append(("ticker", "in", list(tickers)))
    df = pd.read_parquet(path, columns=["ticker", "date", field], filters=[filt])
    panel = df.pivot_table(index="date", columns="ticker", values=field).sort_index()
    if panel.empty:
        return panel
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    return panel.reindex(bidx).ffill(limit=3)


def us_universe(sector=None, category="Domestic Common Stock", marketcap=None,
                include_delisted=True, top_n=None) -> list:
    """US-equity universe from OWNED Sharadar TICKERS (survivorship-clean — DELISTED incl by default).
    Filter by `sector` (e.g. 'Financial Services'), `category` (default common stock), `marketcap`
    scale substring (e.g. 'Large','Mid','Small'). **`top_n` bounds to the N MOST-LIQUID names**
    (recent median dollar volume) — USE THIS for cross-sectional equity: the full ~16k universe is
    too slow/memory-heavy for the CPCV rails (a run OOM'd at 14.5GB). Returns ticker list."""
    import glob
    p = glob.glob(os.path.join(SHARADAR_DIR, "SHARADAR_TICKERS_*.csv"))[0]
    tk = pd.read_csv(p, usecols=["ticker", "category", "sector", "isdelisted", "scalemarketcap"])
    tk = tk[tk["ticker"].notna()]
    if category:
        tk = tk[tk["category"].fillna("").str.contains(category, case=False, regex=False)]
    if sector:
        tk = tk[tk["sector"] == sector]
    if marketcap:
        tk = tk[tk["scalemarketcap"].fillna("").str.contains(marketcap, case=False, regex=False)]
    if not include_delisted:
        tk = tk[tk["isdelisted"] == "N"]
    names = sorted(tk["ticker"].dropna().unique().tolist())
    if top_n and len(names) > top_n:
        # drop the untradable nano/micro tail cheaply first so the price load stays sane,
        # then keep the top_n by recent median dollar volume (close*volume).
        if len(names) > 3000 and not marketcap:
            tk = tk[~tk["scalemarketcap"].fillna("").str.contains("Nano|Micro", case=False, regex=True)]
            names = sorted(tk["ticker"].dropna().unique().tolist())
        # E3: ONE parquet scan for both fields; then the EXACT sep_panel grid transform
        # (business-day reindex + ffill(limit=3)) so the liquidity ranking is byte-identical
        # to the historical two-call path (universe selection is part of frozen pre-regs).
        path = _sep_cache()
        df = pd.read_parquet(path, columns=["ticker", "date", "closeadj", "volume"],
                             filters=[[("date", ">=", pd.Timestamp("2015-01-01")),
                                       ("ticker", "in", names)]])

        def _grid(field):
            panel = df.pivot_table(index="date", columns="ticker", values=field).sort_index()
            bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
            return panel.reindex(bidx).ffill(limit=3)

        dollar = (_grid("closeadj") * _grid("volume")).tail(252).median().dropna()
        names = sorted(dollar.nlargest(min(top_n, len(dollar))).index.tolist())
    return names


def _sf1_cache() -> str:
    import zipfile
    out = os.path.join(_CACHE_DIR, "sf1_long.parquet")
    src = os.path.join(SHARADAR_DIR, "SF1.zip")
    if _stale(out, src):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        z = zipfile.ZipFile(src)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, parse_dates=["datekey", "calendardate"], low_memory=False)
        tmp = out + ".tmp"
        df.sort_values("ticker").to_parquet(tmp, index=False, row_group_size=200_000)
        os.replace(tmp, out)
    return out


def sf1(tickers, fields=None, dimension="ARQ") -> pd.DataFrame:
    """OWNED Sharadar SF1 fundamentals (point-in-time). Long DataFrame [ticker, datekey, calendardate, <fields>]
    for `dimension` (ARQ=as-reported quarterly, MRQ=most-recent quarterly, ART=trailing-twelve as-reported).
    IMPORTANT: use `datekey` (the FILING/available date) — NOT calendardate — as the as-of date to avoid look-ahead."""
    path = _sf1_cache()
    cols = (["ticker", "datekey", "calendardate", "dimension"] + list(fields)) if fields else None
    df = pd.read_parquet(path, columns=cols,
                         filters=[("ticker", "in", list(tickers)), ("dimension", "==", dimension)])
    return df.sort_values(["ticker", "datekey"]).reset_index(drop=True)


# ── Databento GLBX — individual futures CONTRACT MONTHS (basis-momentum substrate) ──
DATABENTO_DIR = str(DATA / "databento")
_MONTH_CODE = {m: i + 1 for i, m in enumerate("FGHJKMNQUVXZ")}


def fut_curve(root, n_contracts=2, min_volume=1) -> pd.DataFrame:
    """Daily futures CURVE panel from owned Databento GLBX daily bars (one-time pull 2026-06-12,
    17 commodity roots, all contract months, 2010+). For each business day, ranks the OUTRIGHT
    contracts that actually traded (volume >= min_volume) by expiry and returns the nearest
    `n_contracts`: columns close_1..n, volume_1..n, symbol_1..n, days_to_roll_1 (days until the
    front contract's last trade — for roll-aware execution).

    THE point of this dataset: basis-momentum (Boons-Prado 2019) and curve signals need the first
    AND second contract separately; stitched continuous series (yf_panel) cannot express them.

    Gotchas handled here so generated code doesn't have to:
    - spread/butterfly symbols (e.g. 'CLN6-CLQ6', 'CL:BF ...') are EXCLUDED (outrights only);
    - single-digit year codes wrap each decade AND one symbol can be TWO contracts (CLZ0 rows in
      2010 = Dec-2010; CLZ0 rows in 2012+ = Dec-2020, listed ~9y out) — disambiguated per
      Databento instrument_id using each instrument's LAST trade date (≈ expiry month);
    - returns are NOT roll-adjusted: compute returns WITHIN a contract (groupby symbol) or use
      rank-1/rank-2 series with roll-day awareness — never diff close_1 across a roll naively."""
    import re
    cache = os.path.join(_CACHE_DIR, f"futcurve_{root}_{n_contracts}_{min_volume}.parquet")
    src = os.path.join(DATABENTO_DIR, f"{root}_ohlcv1d.parquet")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} — root '{root}' not in the owned Databento pull "
                                f"(see wiki DATA_CATALOG; re-quote with metadata.get_cost before adding).")
    if not _stale(cache, src):
        return pd.read_parquet(cache)
    df = pd.read_parquet(src, columns=["symbol", "close", "volume", "instrument_id"])
    df["date"] = df.index.tz_localize(None).normalize()
    df = df.reset_index(drop=True)  # ts_event index has duplicate labels (one row per contract per day)
    pat = re.compile(rf"^{re.escape(root)}([FGHJKMNQUVXZ])(\d{{1,2}})$")
    m = df["symbol"].str.extract(pat)
    df = df[m[0].notna() & (df["volume"] >= min_volume)].copy()
    df["_mon"] = df["symbol"].str.extract(pat)[0].map(_MONTH_CODE)
    df["_yd"] = df["symbol"].str.extract(pat)[1].astype(int)
    # expiry per INSTRUMENT (not symbol — symbols recycle): last trade date ≈ expiry month, so the
    # expiry year is the year (last_trade_year-1 .. +1) whose last digit(s) match the year code.
    last_trade = df.groupby("instrument_id")["date"].transform("max")
    lty = last_trade.dt.year
    yd = df["_yd"]
    exp_year = np.where(yd >= 10, 2000 + yd,
                        lty + ((yd - lty % 10 + 5) % 10) - 5)  # nearest year (±5) with matching digit
    df["_exp"] = exp_year * 100 + df["_mon"]
    df["_dtr"] = (last_trade - df["date"]).dt.days
    df = df.sort_values(["date", "_exp", "_dtr"])
    df = df.drop_duplicates(["date", "_exp"])  # rare dual-listing duplicates: keep nearer-expiry instrument
    df["_rank"] = df.groupby("date").cumcount() + 1
    df = df[df["_rank"] <= n_contracts]
    out = df.pivot(index="date", columns="_rank", values=["close", "volume", "symbol"])
    out.columns = [f"{f}_{r}" for f, r in out.columns]
    dtr = df[df["_rank"] == 1].set_index("date")["_dtr"]
    out["days_to_roll_1"] = dtr
    out = out.sort_index()
    try:
        tmp = cache + ".tmp"
        out.to_parquet(tmp)
        os.replace(tmp, cache)
    except OSError:
        pass
    return out


# ── Free public sources unblocking queued families (COT / CBOE / funding / auctions) ──

def _http_get(url, timeout=60, retries=3):
    """GET with a browser-ish User-Agent (CFTC 403s python-urllib) + simple retry (Binance
    public API is transiently flaky). Returns bytes."""
    import time as _t, urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; crucible)"})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if attempt == retries - 1:
                raise
            _t.sleep(5 * (attempt + 1))

# CFTC contract market codes for our futures roots (CME/NYMEX/COMEX/CBOT contracts —
# verified 2026-06-12 against deacot2024; deliberately NOT the ICE lookalikes).
_COT_CODES = {
    "CL": "067651", "NG": "023651", "HO": "022651", "RB": "111659",
    "GC": "088691", "SI": "084691", "HG": "085692", "PL": "076651", "PA": "075651",
    "ZC": "002602", "ZS": "005602", "ZW": "001602", "ZL": "007601", "ZM": "026603",
    "LE": "057642", "HE": "054642", "GF": "061641",
}


def cot_positioning(roots=None, start_year=2010) -> pd.DataFrame:
    """CFTC Commitments of Traders (legacy futures-only) weekly positioning panel, free.
    Columns per root: {root}_comm_net (commercial net = hedgers), {root}_noncomm_net
    (speculators), {root}_oi (open interest). LOOK-AHEAD DISCIPLINE: indexed by the
    RELEASE date (as-of Tuesday + 3 days = Friday 15:30 ET publication) — a backtest may
    use a row from its index date onward, never from the as-of Tuesday.
    Hedging-pressure signal (Basu-Miffre): comm_net / oi (more short = more hedging pressure)."""
    import io, zipfile
    roots = list(roots or _COT_CODES)
    this_year = pd.Timestamp.today().year
    cache = _day_cache("cot", [*roots, start_year, this_year])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    frames = []
    for yr in range(start_year, this_year + 1):
        u = f"https://www.cftc.gov/files/dea/history/deacot{yr}.zip"
        try:
            raw = _http_get(u, timeout=60)
            z = zipfile.ZipFile(io.BytesIO(raw))
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue  # current year may not exist yet early in Jan
        df.columns = [c.strip() for c in df.columns]
        df = df[df["CFTC Contract Market Code"].astype(str).str.strip().isin(
            {_COT_CODES[r] for r in roots})]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    code2root = {_COT_CODES[r]: r for r in roots}
    df["_root"] = df["CFTC Contract Market Code"].astype(str).str.strip().map(code2root)
    df["_asof"] = pd.to_datetime(df["As of Date in Form YYYY-MM-DD"])
    df["_release"] = df["_asof"] + pd.Timedelta(days=3)  # Tue data -> Fri publication
    out = {}
    for root, g in df.groupby("_root"):
        g = g.set_index("_release").sort_index()
        comm = g["Commercial Positions-Long (All)"] - g["Commercial Positions-Short (All)"]
        nonc = g["Noncommercial Positions-Long (All)"] - g["Noncommercial Positions-Short (All)"]
        out[f"{root}_comm_net"], out[f"{root}_noncomm_net"] = comm, nonc
        out[f"{root}_oi"] = g["Open Interest (All)"]
    panel = pd.DataFrame(out).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def cboe_index(names=("VIX3M", "VVIX", "SKEW", "PUT")) -> pd.DataFrame:
    """CBOE published index history (free CDN CSVs), daily CLOSE panel on a business-day grid.
    Depth: VIX3M 2009+, VVIX 2006+, SKEW 1990+, PUT 1991+. Spot VIX itself: use FRED VIXCLS.
    Canonical contango regime signal: VIXCLS / VIX3M (backwardation when > ~1.0)."""
    import io
    names = list(names)
    cache = _day_cache("cboe", names)
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    out = {}
    for n in names:
        u = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{n}_History.csv"
        df = pd.read_csv(io.BytesIO(_http_get(u, timeout=60)))
        df.columns = [c.strip().upper() for c in df.columns]
        col = "CLOSE" if "CLOSE" in df.columns else n.upper()
        out[n] = pd.Series(df[col].values, index=pd.to_datetime(df["DATE"])).sort_index()
    panel = pd.DataFrame(out).sort_index()
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(bidx).ffill(limit=3)
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def funding_rates(symbols=("BTCUSDT", "ETHUSDT"), source="binance") -> pd.DataFrame:
    """Perp funding-rate history (free public APIs), DAILY sum of the 8h funding prints per symbol.
    Binance depth: 2019-09+. Sign convention: positive = longs PAY shorts (short-perp earns).
    Replaces the deleted Midas carry_returns(); the carry+trend STRUCTURE is validated wiki
    knowledge but any new leg needs fresh forward validation before deployment."""
    import time as _t
    cache = _day_cache("funding", [source, *symbols])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    out = {}
    for sym in symbols:
        rows, start = [], 1568102400000  # 2019-09-10, Binance perp launch era
        while True:
            u = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}"
                 f"&startTime={start}&limit=1000")
            batch = json.loads(_http_get(u, timeout=40))
            if not batch:
                break
            rows += batch
            if len(batch) < 1000:
                break
            start = batch[-1]["fundingTime"] + 1
            _t.sleep(0.3)  # public rate-limit politeness
        s = pd.Series({pd.Timestamp(r["fundingTime"], unit="ms"): float(r["fundingRate"])
                       for r in rows}).sort_index()
        out[sym] = s.resample("1D").sum()  # daily funding accrual
    panel = pd.DataFrame(out).sort_index()
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def treasury_auctions(types=("Note", "Bond"), start="2010-01-01") -> pd.DataFrame:
    """US Treasury auction calendar/history from the free TreasuryDirect API (depth: 1979+).
    Long DataFrame [auction_date, announcement_date, issue_date, sec_type, term, cusip,
    offering_amount] sorted by auction_date. POINT-IN-TIME: the auction is knowable from
    announcement_date (~1 week prior); supply-concession studies may condition on the
    announcement but measure returns around auction_date."""
    types = list(types)
    cache = _day_cache("ustauct", [*types, start])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    rows = []
    for t in types:
        u = f"https://www.treasurydirect.gov/TA_WS/securities/search?type={t}&format=json"
        for r in json.loads(_http_get(u, timeout=120)):
            if not r.get("auctionDate"):
                continue
            rows.append({"auction_date": pd.Timestamp(r["auctionDate"]),
                         "announcement_date": pd.Timestamp(r["announcementDate"]) if r.get("announcementDate") else pd.NaT,
                         "issue_date": pd.Timestamp(r["issueDate"]) if r.get("issueDate") else pd.NaT,
                         "sec_type": t, "term": r.get("securityTerm", ""),
                         "cusip": r.get("cusip", ""),
                         "offering_amount": float(r["offeringAmount"]) if r.get("offeringAmount") else np.nan})
    df = pd.DataFrame(rows)
    df = df[df["auction_date"] >= pd.Timestamp(start)]
    df = df.sort_values("auction_date").reset_index(drop=True)
    if cache:
        try:
            tmp = cache + ".tmp"
            df.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return df
