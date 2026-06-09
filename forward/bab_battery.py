"""PRE-REGISTERED cross-market robustness battery for the defensive/BAB premium (2026-06-09).
Tests the SAME defensive premise in UNTOUCHED universes; reports ALL results (no cherry-picking).
Universes declared upfront: (A) US large-cap stocks [Sharadar]; (B) 11 sector ETFs [yfinance];
(C) ~28 cross-asset/international ETFs [yfinance]. Validated reference: mid-cap (holdout 0.80)."""
import sys; sys.path.insert(0,'/root/hephaestus'); sys.path.insert(0,'/root/hephaestus/forward')
import numpy as np, pandas as pd
from sdk.adapters import yf_panel
H = '2022-01-01'
def sh(r): r=pd.Series(r).dropna(); return round(float(r.mean()/r.std()*np.sqrt(252)),2) if len(r)>20 and r.std()>0 else None

def bab_panel(panel, beta_lb=252, vol_lb=63, target_vol=0.10, cost_bps=8.0, hold='ME'):
    px=panel.sort_index().ffill(limit=3); rets=px.pct_change()
    mkt=rets.mean(axis=1); varm=mkt.rolling(beta_lb,min_periods=beta_lb//2).var()
    betas=pd.DataFrame({c:rets[c].rolling(beta_lb,min_periods=beta_lb//2).cov(mkt)/varm for c in rets.columns})
    tgt=pd.DataFrame(index=betas.resample(hold).last().index, columns=rets.columns, dtype=float)
    for d in tgt.index:
        b=betas.loc[:d].iloc[-1].dropna()
        if len(b)<6: continue
        z=b.rank()-b.rank().mean(); wl=-z.clip(upper=0); ws=z.clip(lower=0)
        wl=wl/wl.sum() if wl.sum()>0 else wl*0; ws=ws/ws.sum() if ws.sum()>0 else ws*0
        bl=(wl*b).sum(); bh=(ws*b).sum(); row=pd.Series(0.0,index=rets.columns)
        if bl>1e-6: row[wl.index]+=wl/bl
        if bh>1e-6: row[ws.index]-=ws/bh
        tgt.loc[d]=row
    w=tgt.reindex(rets.index,method='ffill').shift(1).fillna(0.0)
    gret=(w*rets).sum(axis=1); rv=gret.rolling(vol_lb).std()*np.sqrt(252)
    scl=(target_vol/rv).clip(upper=3).shift(1).fillna(1.0); w=w.mul(scl,axis=0)
    g=(w*rets).sum(axis=1); cost=w.diff().abs().sum(axis=1)*(cost_bps/1e4)
    return (g-cost).fillna(0.0)

def report(name, ret, costs):
    ret=pd.Series(ret).dropna(); s=ret[ret.index<H]; h=ret[ret.index>=H]
    print(f"  {name:34s} search {sh(s)} | HOLDOUT {sh(h)} | full {sh(ret)} | costs@{costs}")

print("=== B) SECTOR ETFs (11 SPDR) — beta-neutral BAB ===")
sect=['XLE','XLF','XLK','XLV','XLI','XLP','XLY','XLU','XLB','XLRE','XLC']
ps=yf_panel(sect, start='2004-01-01')
for c in [3,8]:
    report(f"sector-ETF cost{c}bps", bab_panel(ps, cost_bps=c), c)
print("=== C) CROSS-ASSET / INTL ETFs (~28) — beta-neutral BAB ('BAB everywhere') ===")
xa=['SPY','QQQ','IWM','EFA','EEM','EWJ','EWG','EWU','EWZ','EWA','EWC','EWH','FXI',
    'TLT','IEF','SHY','LQD','HYG','TIP','GLD','SLV','DBC','USO','VNQ','XLU','XLP','XLE','XLK']
px=yf_panel(xa, start='2004-01-01')
for c in [3,8]:
    report(f"cross-asset cost{c}bps", bab_panel(px, cost_bps=c), c)
