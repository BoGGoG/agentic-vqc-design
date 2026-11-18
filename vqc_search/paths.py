"""Repository paths used by the models, the agent tools and the experiment scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
AGENT_RUNS_DIR = RESULTS_DIR / "agent_runs"
BASELINES_DIR = RESULTS_DIR / "baselines"
# New tool calls (re-running the agent tools by hand) write here, not into the paper logs.
NEW_RUNS_DIR = RESULTS_DIR / "new_runs"
PLOTS_DIR = REPO_ROOT / "figures" / "model_plots"
# Re-running an experiment script writes its CSV here, so the paper's results in
# results/baselines/ are never overwritten.
RERUN_DIR = RESULTS_DIR / "rerun"
