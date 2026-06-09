"""Scratch prototype — validate the SUE-PEAD x trend pipeline before freezing the module."""
import sys, time, json
sys.path.insert(0, "/root/hephaestus")
import numpy as np, pandas as pd
from sdk.adapters import sf1, sep_panel, us_universe, trend_returns

t0 = time.time()
U = json.load(open("/root/atlas/data/universes/sharadar_midsmall.json"))["tickers"]
print("universe", len(U))

# ---- EPS events (point-in-time via datekey) ----
f = sf1(U, fields=["eps"], dimension="ARQ")
f = f.dropna(subset=["eps", "datekey"]).sort_values(["ticker", "datekey"])
g = f.groupby("ticker")["eps"]
dEPS = f["eps"] - g.shift(4)                      # seasonal random walk (YoY)
sig = dEPS.groupby(f["ticker"]).transform(lambda s: s.rolling(8, min_periods=6).std())
f["sue"] = (dEPS / sig.replace(0, np.nan))
f = f.dropna(subset=["sue"])
f["sue"] = f["sue"].clip(-10, 10)
print("SUE events", len(f), "names", f.ticker.nunique(), "t=%.1fs" % (time.time() - t0))

# ---- prices + returns ----
px = sep_panel(U, start="2004-01-01", field="closeadj")
vol = sep_panel(U, start="2004-01-01", field="volume")
rets = px.pct_change()
advdol = (px * vol).rolling(20, min_periods=10).median()   # ADV proxy
nvol = rets.rolling(60, min_periods=30).std()              # per-name daily vol
print("px", px.shape, "t=%.1fs" % (time.time() - t0))

# ---- monthly cohort formation ----
HOLD_CAL = 90        # ~60 trading days drift window
PMIN, ADVMIN = 5.0, 500_000.0
QTOP = 0.20
form_dates = pd.date_range(px.index.min(), px.index.max(), freq="BME")
form_dates = [d for d in form_dates if d in px.index]

names = px.columns
W = pd.DataFrame(0.0, index=pd.DatetimeIndex(form_dates), columns=names)
# latest SUE event per ticker as-of each form date, restricted to trailing window
fe = f[["ticker", "datekey", "sue"]].copy()
for d in form_dates:
    win = fe[(fe.datekey <= d) & (fe.datekey > d - pd.Timedelta(days=HOLD_CAL))]
    if win.empty:
        continue
    last = win.sort_values("datekey").groupby("ticker").last()  # most recent in window
    cand = last.index
    cand = cand[cand.isin(names)]
    if len(cand) < 50:
        continue
    p = px.loc[d, cand]; a = advdol.loc[d, cand]; v = nvol.loc[d, cand]
    ok = (p >= PMIN) & (a >= ADVMIN) & v.notna() & (v > 0)
    cand = cand[ok.values]
    if len(cand) < 50:
        continue
    s = last.loc[cand, "sue"]
    ql, qh = s.quantile(QTOP), s.quantile(1 - QTOP)
    longs = s[s >= qh].index; shorts = s[s <= ql].index
    if len(longs) < 5 or len(shorts) < 5:
        continue
    iv = (1.0 / v[cand]).clip(upper=v[cand].replace(0, np.nan).median()*0+1e9)
    wl = iv[longs] / iv[longs].sum()
    ws = iv[shorts] / iv[shorts].sum()
    W.loc[d, longs] = wl.values
    W.loc[d, shorts] = -ws.values

# count avg names/side
nside = (W > 0).sum(axis=1)
print("avg longs/side", round(nside[nside > 0].mean(), 1), "median", int(nside[nside > 0].median()),
      "cohorts", int((nside > 0).sum()))

# ---- daily leg returns (weights lagged, monthly-held) ----
Wd = W.reindex(px.index, method="ffill").shift(1).fillna(0.0)
gross = (Wd * rets).sum(axis=1)
turn = (W.reindex(px.index).fillna(0.0)).diff().abs().sum(axis=1).fillna(0.0)
cost = turn * (30.0 / 1e4)   # 30 bps one-way small-cap incl borrow
pead = (gross - cost).rename("pead")
pead = pead[pead.index >= "2005-01-01"].dropna()

def sharpe(r): r = pd.Series(r).dropna(); return r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
def mdd(r): e=(1+pd.Series(r).dropna()).cumprod(); return float((e/e.cummax()-1).min())
print("PEAD full Sharpe %.2f  mdd %.1f%%  ann %.1f%%" %
      (sharpe(pead), mdd(pead)*100, pead.mean()*252*100))
print("PEAD search<2022 Sharpe %.2f   holdout>=2022 Sharpe %.2f" %
      (sharpe(pead[pead.index<'2022-01-01']), sharpe(pead[pead.index>='2022-01-01'])))

# ---- combination with trend ----
tr, trtrades = trend_returns()
tr = pd.Series(tr); tr.index = pd.to_datetime(tr.index)
try: tr.index = tr.index.tz_localize(None)
except Exception: pass
both = pd.concat([pead.rename("pead"), tr.rename("trend")], axis=1).dropna()
print("overlap", str(both.index.min().date()), str(both.index.max().date()), "corr",
      round(both["pead"].corr(both["trend"]), 3))
# vol-match 50/50
ann=np.sqrt(252); tv=0.10
sc=(tv/ann)/both.rolling(63,min_periods=30).std().shift(1).replace(0,np.nan)
sc=sc.clip(upper=3).fillna(0)
held=sc.mul(pd.Series({"pead":.5,"trend":.5}),axis=1).groupby(both.index.to_period("W")).transform("first")
comb=(held*both).sum(axis=1)-held.diff().abs().sum(axis=1).fillna(0)*8/1e4
comb=comb.iloc[64:].dropna()
peadO=both["pead"].loc[comb.index]
print("PEAD-only(overlap) Sharpe %.2f mdd %.1f%% | COMB Sharpe %.2f mdd %.1f%%" %
      (sharpe(peadO), mdd(peadO)*100, sharpe(comb), mdd(comb)*100))
print("total %.1fs" % (time.time()-t0))

# stash W for trade-book check
W.to_pickle("/tmp/_pead_W.pkl"); rets.to_pickle("/tmp/_pead_rets.pkl")
