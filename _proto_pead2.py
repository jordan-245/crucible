import sys, time, json, glob
sys.path.insert(0, "/root/hephaestus")
import numpy as np, pandas as pd
from sdk.adapters import sf1, sep_panel, trend_returns

t0 = time.time()
U = json.load(open("/root/atlas/data/universes/sharadar_midsmall.json"))["tickers"]
# sector map
tkc = pd.read_csv(glob.glob("/root/atlas/data/sharadar/SHARADAR_TICKERS_*.csv")[0],
                  usecols=["ticker", "sector"]).dropna()
SEC = dict(zip(tkc.ticker, tkc.sector))

# ---- SUE events ----
f = sf1(U, fields=["eps"], dimension="ARQ").dropna(subset=["eps", "datekey"]).sort_values(["ticker", "datekey"])
d4 = f["eps"] - f.groupby("ticker")["eps"].shift(4)
s8 = d4.groupby(f["ticker"]).transform(lambda s: s.rolling(8, min_periods=6).std())
f["sue"] = (d4 / s8.replace(0, np.nan)).clip(-10, 10)
ev = f.dropna(subset=["sue"])[["ticker", "datekey", "sue"]].sort_values("datekey").reset_index(drop=True)

px = sep_panel(U, start="2004-01-01", field="closeadj")
vol = sep_panel(U, start="2004-01-01", field="volume")
rets = px.pct_change()
adv = (px * vol).rolling(20, min_periods=10).median()
nvol = rets.rolling(60, min_periods=30).std()
print("loaded t=%.1fs" % (time.time() - t0))

HOLD, PMIN, ADVMIN, Q = 90, 5.0, 5e5, 0.20
form = [d for d in pd.date_range(px.index.min(), px.index.max(), freq="BME") if d in px.index]
formidx = pd.DatetimeIndex(form)

# active SUE as-of each form date via merge_asof (most recent datekey <= d), then age filter
left = pd.MultiIndex.from_product([formidx, sorted(ev.ticker.unique())], names=["d", "ticker"]).to_frame(index=False)
left = left.sort_values("d")
act = pd.merge_asof(left, ev.rename(columns={"datekey": "d_ann"}).sort_values("d_ann"),
                    left_on="d", right_on="d_ann", by="ticker", direction="backward")
act["age"] = (act["d"] - act["d_ann"]).dt.days
act = act[(act.age >= 0) & (act.age < HOLD)].dropna(subset=["sue"])
print("active rows", len(act), "t=%.1fs" % (time.time() - t0))


def build_weights(act, long_only=False):
    W = pd.DataFrame(0.0, index=formidx, columns=px.columns)
    for d, grp in act.groupby("d"):
        c = grp.set_index("ticker")
        cand = c.index[c.index.isin(px.columns)]
        if len(cand) < 50:
            continue
        p, a, v = px.loc[d, cand], adv.loc[d, cand], nvol.loc[d, cand]
        ok = (p >= PMIN) & (a >= ADVMIN) & v.notna() & (v > 0)
        cand = cand[ok.values]
        if len(cand) < 50:
            continue
        s = c.loc[cand, "sue"]
        qh = s.quantile(1 - Q)
        longs = s[s >= qh].index
        iv = 1.0 / nvol.loc[d, longs]
        wl = iv / iv.sum()
        if long_only:
            W.loc[d, longs] = wl.values
            uni = cand                                   # borrowable hedge = short EW universe basket
            W.loc[d, uni] = W.loc[d, uni].values - (1.0 / len(uni))
        else:
            ql = s.quantile(Q)
            shorts = s[s <= ql].index
            ivs = 1.0 / nvol.loc[d, shorts]
            ws = ivs / ivs.sum()
            W.loc[d, longs] = wl.values
            W.loc[d, shorts] = -ws.values
    return W


def leg(W, cost_bps):
    Wd = W.reindex(px.index, method="ffill").shift(1).fillna(0.0)
    gross = (Wd * rets).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).reindex(px.index).fillna(0.0)
    net = (gross - turn * cost_bps / 1e4).rename("pead")
    return net[net.index >= "2005-01-01"].dropna()


def sh(r): r = pd.Series(r).dropna(); return r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
def md(r): e = (1+pd.Series(r).dropna()).cumprod(); return float((e/e.cummax()-1).min())

for lo in (False, True):
    W = build_weights(act, long_only=lo)
    for cb in (30, 40):
        n = leg(W, cb)
        tag = "LONGONLY-hedged" if lo else "LONGSHORT"
        print("%s cost%d: full Sh %.2f ann %.1f%% mdd %.1f%% | search %.2f hold %.2f" %
              (tag, cb, sh(n), n.mean()*252*100, md(n)*100,
               sh(n[n.index < '2022-01-01']), sh(n[n.index >= '2022-01-01'])))
    if lo:
        Wlo = W
print("t=%.1fs" % (time.time() - t0))

# ---- trade book for the chosen primary (use long-only-hedged) ----
def trade_book(W, cost_bps, cap=100000.0):
    Wd = W.reindex(px.index, method="ffill").shift(1).fillna(0.0)
    pnl_daily = Wd * rets * cap
    trades = []
    pos_days = {}
    for tk in W.columns:
        w = Wd[tk]
        held = w.abs() > 1e-9
        if not held.any():
            continue
        grp = (held != held.shift()).cumsum()[held]
        for _, idx in w[held].groupby(grp):
            dts = idx.index
            run_pnl = float(pnl_daily.loc[dts, tk].sum())
            pv = float(w.loc[dts].abs().mean() * cap)
            trades.append(dict(ticker=tk, sector=SEC.get(tk, "Unknown"),
                               entry_date=str(dts[0].date()), exit_date=str(dts[-1].date()),
                               hold_days=int(len(dts)), position_value=pv, pnl=run_pnl))
            pos_days[tk] = pos_days.get(tk, 0) + len(dts)
    return trades, pos_days

trades, pos_days = trade_book(Wlo, 30)
secs = pd.Series([t["sector"] for t in trades]).value_counts()
tot_pd = sum(pos_days.values())
top_share = max(pos_days.values())/tot_pd
print("TRADES", len(trades), "| sectors", len(secs), "| max name pos-day share %.1f%%" % (top_share*100))
print("sector spread:", secs.to_dict())
print("t=%.1fs" % (time.time() - t0))
