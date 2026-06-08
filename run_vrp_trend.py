import sys; sys.path.insert(0, "/root/hephaestus")
from strategies.vrp_trend import SPEC, signal, load_data
from sdk.harness import run_experiment, _sharpe, _maxdd
import pandas as pd
# diagnostic: VRP-alone vs combined (the agent's pre-registered comparison)
panel = load_data()
import numpy as np
def vol_scale(r,t=0.10): v=r.std()*np.sqrt(252); return r*(t/v) if v>0 else r
# rebuild legs for the diagnostic
import warnings; warnings.filterwarnings("ignore")
combo, trades = signal(panel)
print("=== running agent's VRP+Trend through the SDK harness ===")
v = run_experiment(SPEC, write_wiki=True, alert=True)
print("\n=== VERDICT ===")
for k in ("tier","dsr","median_cpcv","pbo","search_sharpe","holdout_sharpe","holdout_pass","deployment_passed","deploy_peak","deploy_sectors","full_sharpe","full_maxdd","n_trades","PASSED_ALL_GATES"):
    print(f"  {k}: {v[k]}")
