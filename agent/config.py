"""Shared forge-agent config — single source of truth for the LLM invocation all smiths use."""
import json
import os


def _policy_model(tier: str = "frontier", failsafe: str = "claude-opus-4-8") -> str:
    """Read the central model policy (/root/.pi/model-policy.json). Failsafe = $0-Max model."""
    try:
        with open("/root/.pi/model-policy.json") as fh:
            return json.load(fh)["tiers"][tier]
    except Exception:
        return failsafe


# The model every smith uses for propose / codegen / scout.
# Resolution order: FORGE_MODEL env (per-run override) > central policy > $0-Max failsafe.
MODEL = os.environ.get("FORGE_MODEL") or _policy_model()
SYS = "You are Claude Code, Anthropic's official CLI for Claude."


def pi_cmd() -> list[str]:
    """The pi invocation for ALL forge LLM calls.

    --no-tools is critical: these are PURE generation calls (given context -> return JSON/code).
    Without it, pi runs AGENTICALLY and the codegen step ran the entire backtest itself in a bash
    tool loop until the 15-min timeout -> crash, plus ~2x compute and heavy Max-quota burn (the
    issuance-factor run died exactly this way). --no-context-files skips AGENTS.md discovery."""
    return ["pi", "-p", "--model", MODEL, "--no-tools", "--no-context-files",
            "--system-prompt", SYS, "--mode", "json"]
