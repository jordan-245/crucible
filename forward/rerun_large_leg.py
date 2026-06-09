import sys, json, traceback
sys.path.insert(0, '/root/hephaestus')
from importlib import import_module
MOD = 'strategies.auto_value_momentum_trend_overlay_large_cap_t_ithmtl_1968'
print("RERUN_START", MOD, flush=True)
try:
    m = import_module(MOD)
    from sdk.harness import run_experiment
    v = run_experiment(m.SPEC, write_wiki=True, alert=False)  # supervised: record, no auto-ping
    print('VERDICT_JSON=' + json.dumps({k: v[k] for k in v}, default=str), flush=True)
    print("RERUN_DONE", flush=True)
except Exception as e:
    traceback.print_exc()
    print("RERUN_FAIL", type(e).__name__, str(e)[:300], flush=True)
