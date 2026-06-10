"""SDK demo strategy: wraps the validated Boreas TSMOM as a StrategySpec to prove the harness."""
import sys
sys.path.insert(0, "/root/boreas/research")
from tsmom import run_tsmom, grid_configs
from sdk.harness import StrategySpec

def load_data():
    return None  # run_tsmom loads the Boreas futures panel internally

def signal(panel, **params):
    return run_tsmom(**params)  # (daily_returns, trades)

SPEC = StrategySpec(
    id="sdk-demo-trend",
    family="sdk_demo_tsmom",
    title="SDK demo — diversified TSMOM via the Crucible harness",
    markets=["futures"],
    data_desc="FREE (yfinance 21 continuous futures 2005-2026)",
    pre_registration="Standard 1/3/12m TSMOM sign blend, inverse-vol, weekly, 8bps. FROZEN. "
                     "Proves the SDK reproduces the Boreas trend verdict through the rails.",
    load_data=load_data, signal=signal,
    default_params={}, grid=grid_configs(),
    holdout_start="2022-01-01", deploy_max_positions=21,
)
