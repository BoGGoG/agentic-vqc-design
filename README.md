# AI Agents for Variational Quantum Circuit Design

Code, data and agent logs for the paper

> M. Knipfer, A. Roman, K. T. Matchev, K. Matcheva, S. Gleyzer,
> *AI Agents for Variational Quantum Circuit Design*, arXiv:2602.19387.

A large language model (LLM) agent designs variational quantum circuits (VQCs) for hybrid
quantum-classical neural networks. It proposes a circuit as PennyLane code, a tool trains the
network and returns the test error, and the agent iterates. Task: predict the position of a
noisy Gaussian peak in a 1D signal of length 21 (metric: test RMSE).

This repository contains:

- `paper_figures.ipynb`: regenerates every data figure and every computed table of the paper
  from the shipped logs. No training; a few minutes on a laptop CPU.
- The complete logs of the four agent sessions reported in the paper: one row per circuit the
  agent trained (circuit code, test RMSE, training history, ...) and the full conversation
  transcripts (prompts, agent messages, tool calls and results, our messages).
- The training code: the three hybrid architectures, the tools the agent called (with the
  docstrings it saw), and the scripts for all baselines (random and evolutionary search,
  hardware-efficient and strongly entangling ansätze, scaling sweep, five-seed retraining of
  the agents' best circuits, classical baseline).
- The data set.

The agent framework, Orchestral AI, is **not** included; see [Orchestral](#orchestral).

## Quick start

```bash
conda env create -f environment.yml      # Python 3.12, CPU is enough
conda activate agentic-vqc
jupyter lab paper_figures.ipynb           # or: jupyter nbconvert --to notebook --execute paper_figures.ipynb
```

On Linux, `pip` installs the CUDA build of PyTorch by default. For a smaller CPU-only install,
run `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu` in the
environment before (or after) `pip install -r requirements.txt`.

The figures use SciencePlots with LaTeX text rendering if `latex` is on the `PATH`; without it
the notebook falls back to matplotlib's mathtext (same figures, different fonts).

## Repository layout

```
├── paper_figures.ipynb      all data figures and computed tables of the paper
├── data/
│   └── gaussian_peak_noise_0_01/{train,test}.h5   the data set (4000 train / 1000 test signals)
├── results/
│   ├── agent_runs/<session>/*.parquet   agent logs, one file per trained circuit
│   ├── transcripts/<session>/           the conversation of each session (Orchestral format)
│   └── baselines/                       results of all baseline experiments (CSV)
├── vqc_search/              models, VQC code-string compiler, search grammar
│   ├── models/              SimpleQNN.py, FullQuantum.py, Quanvolutional.py, SimpleQNN_Jax.py
│   ├── utils/quantum_circuit.py   VQC code string -> PennyLane QNode / TorchLayer
│   └── search/              circuit template grammar and sampler (random / evolutionary search)
├── agent_tools/             the tools the agent called, as used in the sessions
├── experiments/             baseline and retraining scripts, data generation
└── figures/
    ├── static/              Figs. 1-2 (drawn schematics)
    └── paper/               written by the notebook
```

## Figures and tables

| Paper | Produced by | Input |
|---|---|---|
| Fig. 1, Fig. 2 | drawn by hand | `figures/static/*.pdf` |
| Fig. 3 | notebook, Sec. 1 | `data/` |
| Figs. 4, 6, 8, 10 | notebook, Sec. 2 | `results/agent_runs/` |
| Figs. 5, 7, 9, 11 | notebook, Sec. 3 | logged circuit code in `results/agent_runs/` |
| Fig. 12 (Lie-EQGNN) | not in this repository | separate code base |
| Fig. 13 | notebook, Sec. 4 | `results/baselines/scaling_sweep_extended.csv` |
| Tables 1, 3, 4 | notebook, Sec. 5 | `results/baselines/` |
| Table 2 | literature comparison | none |
| Table 5 | compiled by hand | `results/transcripts/`, `results/agent_runs/` |
| Tables A1, A2 (Appendix D) | notebook, Sec. 6 | `results/agent_runs/`, `results/baselines/` |

## Agent sessions

| Log folder (`results/agent_runs/`, `results/transcripts/`) | LLM | Architecture | Circuits | Best (iteration, test RMSE) |
|---|---|---|---|---|
| `SimpleQNN_runs_log_Sonnet_3_7_2025_11_17_02` | Claude 3.7 Sonnet (`claude-3-7-sonnet-20250219`, Anthropic API) | Simple QNN | 26 | 25, 0.0326 |
| `SimpleQNN_runs_log_LLama3_3_2025_11_21` | Llama 3.3 70B (`llama-3.3-70b-versatile`, Groq) | Simple QNN | 26 | 20, 0.0207 |
| `FullQuantum_runs_log_LLama3_3_2025_11_21` | Llama 3.3 70B | Full Quantum | 17 | 16, 0.0811 |
| `QuanvNN_runs_log_LLama3_3_2025_11_24` | Llama 3.3 70B | Quanvolutional | 25 | 12, 0.1330 |

Iterations are counted from 0 in the time order of the log files, as in the paper. Only the
sessions reported in the paper are included.

**Run logs** (`*.parquet`, one row each): `VQC_code` (the circuit as proposed by the agent),
`VQC_weights_shape`, `epochs`, `test_MSE`, `test_RMSE`, `train_MSE_hist`, `val_MSE_hist`,
`n_trainable_params_total`, `n_trainable_params_VQC`, `n_gates_in_VQC`, `depth_of_VQC`,
`training_plus_test_time_sec`, `pennylane_version`, `python_version`, `platform`, and
`git_commit` (see [Orchestral](#orchestral)). Read a session with
`pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(".../*.parquet"))])`.

**Transcripts** (`context.json`): the list of messages of the session in Orchestral's format.
`role: system` is the system prompt, `role: user` are our messages, `type: response` are the
LLM's replies (with `tool_calls`, the exact arguments of each tool call) and `role: tool` the
tool results returned to the LLM. Every logged circuit can be matched to its tool call (code
and epochs, in order). `metadata.json` holds the session name and the model selected in the
app when the session was last saved. For the Llama 3.3 Simple QNN session this field reads
`claude-3-7-sonnet` because the app's model selector was changed later; all 16 LLM responses in
that transcript are from `llama-3.3-70b-versatile` (see the `model` field of each response).

The QNN tools did not fix a random seed, so re-training a logged circuit gives a test RMSE
that differs from the logged one by about the seed spread (see the five-seed retraining in the
paper).

## Orchestral

The agent sessions ran in the [Orchestral AI](https://arxiv.org/abs/2601.02577) framework
(A. Roman and J. Roman, arXiv:2601.02577), which handled the LLM API calls, tool calling,
conversation storage and the chat interface. Orchestral is **not part of this repository**:
the version we used is not publicly released.

- Version used for all four sessions: Orchestral **pre-release development version 0.1.0**
  (package `orchestral`, November 2025). The QNN code and tools lived inside that development
  repository at the time; the `git_commit` column of every run log records its commit
  (`a75a32a` for the Claude 3.7 session, `61325e8`, `aa65a9d`, `a96464d`, `573fb52`, `e549c94`
  for the Llama sessions).
- Runtime of the sessions: Python 3.13.9, PennyLane 0.43.0, PyTorch 2.9.0.
- Everything specific to this study is here: `agent_tools/` contains the four tools
  (`train_SimpleQNN.py`, `train_FullQuantum.py`, `train_QuanvNN.py`,
  `calc_hilbert_space_dimension.py`) as of commit `61325e8`, with their docstrings (= the tool
  descriptions the LLM saw) unchanged; only import and output paths were adapted.
  `agent_tools/launch_agent.py` shows how the agent was set up. The system prompt and all
  messages are in the transcripts.
- Without Orchestral, `define_tool` is replaced by a no-op (`agent_tools/_orchestral_compat.py`),
  so the tools are plain functions that can be called directly, e.g. to re-train a logged
  circuit:

```python
from agent_tools.train_SimpleQNN import TrainCustomSimpleQNNTool

result = TrainCustomSimpleQNNTool(
    VQC_code=code, VQC_weights_shape=[9, 5], q_enc_size=5, q_out_size=5, epochs=20
)  # writes a log row to results/new_runs/
```

To drive the tools with an LLM, register them with any tool-calling agent framework; the tool
signatures and docstrings are all an agent needs.

## Re-running the experiments

All scripts run from the repository root and write to `results/rerun/` (the paper's results in
`results/baselines/` are never overwritten). They use the same training settings as the agent
tools (150 training, 250 validation and 500 test signals, Adam with learning rate 0.1, batch
size 16, MultiStepLR at epochs 3, 8, 15 with factor 0.2, 7 epochs unless noted).

| Script | Paper | Output |
|---|---|---|
| `python -m experiments.random_search_baseline` | Tables 3, A2; Appendix A | `random_search_baseline.csv` |
| `python -m experiments.evolutionary_search_baseline` | Tables 3, A2; Appendix A | `evolutionary_search_baseline.csv` |
| `python -m experiments.fixed_ansatz_baselines` | Tables 1, 3, A2 (HEA, SEL; PyTorch) | `fixed_ansatz_baselines.csv` |
| `python -m experiments.scaling_sweep_extended` | Fig. 13, Table A2 (HEA, JAX) | `scaling_sweep_extended.csv` |
| `python -m experiments.retrain_agent_best` | Tables 1, 3, A2 (agent, JAX) | `agent_best_reseed.csv` |
| `python -m experiments.retrain_agent_best_pytorch` | Tables 4, A2 (agent, PyTorch, 7 epochs) | `agent_best_reseed_pytorch_7ep.csv` |
| same, with `FIXED_EPOCHS = None` in the script | Table 4 (20 epochs) | `agent_best_reseed_pytorch_origep.csv` |
| `python -m experiments.classical_baseline_simpleqnn` | Table 4 | `classical_baseline_simpleqnn.csv` |
| `python experiments/generate_data.py [OUT_DIR]` | data set (fixed seeds, reproduces `data/` exactly) | `train.h5`, `test.h5` |

The scaling sweep is the expensive one: on our CPU a single 20-qubit run took up to 1.5 hours.
The script stops runs that exceed 10 minutes (`--retry-timeout SECONDS` re-runs them with a
longer limit) or 40 GB of memory and records them with a non-`ok` status. The JAX scripts (`retrain_agent_best`,
`random_search_baseline`, `evolutionary_search_baseline`, `scaling_sweep_extended`) use the
batched JAX training path in `vqc_search/models/SimpleQNN_Jax.py`; the others use the PyTorch
path of the agent tools. The two give RMSE values that differ by about 0.005 for identical
circuits (see the footnote to Table 1 in the paper).

Notes on the result files:

- `scaling_sweep_extended.csv` contains one duplicated row (20 qubits, L = 3, seed 4). The
  heatmap of the paper includes it (cell mean 0.0360); without it the mean is 0.0365. The
  20-qubit, L = 8 cell has three seeds (time limit).
- `random_search_log.txt` and `evolutionary_search_log.txt` are the console logs of the two
  search runs.

## Citation

```bibtex
@misc{knipfer2026aiagentsvqc,
  title         = {AI Agents for Variational Quantum Circuit Design},
  author        = {Knipfer, Marco and Roman, Alexander and Matchev, Konstantin T. and
                   Matcheva, Katia and Gleyzer, Sergei},
  year          = {2026},
  eprint        = {2602.19387},
  archivePrefix = {arXiv}
}
```

## License

- Code (`vqc_search/`, `agent_tools/`, `experiments/`, `paper_figures.ipynb`): MIT, see
  [`LICENSE`](LICENSE).
- Data, logs and figures (`data/`, `results/`, `figures/`): CC BY 4.0, see
  [`LICENSE-DATA`](LICENSE-DATA). Please cite the paper when you use them.
- The transcripts and logged circuits contain LLM outputs; Llama 3.3 70B outputs are also
  subject to Meta's Llama 3.3 Community License (details in `LICENSE-DATA`).
# agentic-vqc-design
