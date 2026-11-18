"""
Purely classical baseline for the Simple QNN (MDPI referee point R1.3b).

Referee request: "add a purely classical baseline (training the linear embedding
and output layers without the VQC) to show how much the quantum layer actually
contributes."

We take the Simple QNN exactly as used by the agent tool
(`agent_tools/train_SimpleQNN.py` -> `SimpleQNN_CustomVQC`) and replace the VQC
by the identity. Everything else is unchanged:

    Linear(21, k) -> Dropout(0.1) -> (tanh(x) + 1) * pi/2 -> Linear(k, 1) -> Sigmoid

Trainable parameters: 22*k + k + 1 (k=5 -> 116, the count quoted by the referee
for Appendix C, Iteration 1).

Training goes through the shared `train_quantum_model` (n_train=150, Adam lr=0.10,
MultiStepLR [3, 8, 15] x 0.2, batch 16), with the same seeds (0-4) and seeding
as `retrain_agent_best_pytorch.py`, so the results are directly comparable to
`results/baselines/agent_best_reseed_pytorch_origep.csv`.

Run from repo root:
    python -m experiments.classical_baseline_simpleqnn
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from vqc_search.models.Quanvolutional import train_quantum_model
from vqc_search.models.SimpleQNN import create_final_dense

INPUT_SIZE = 21
DROPOUT = 0.1
SEEDS = [0, 1, 2, 3, 4]

# (k = q_enc_size = q_out_size, epochs, what the setting matches)
CONFIGS = [
    (5, 20, "Claude 3.7 best circuit (enc=out=5, 20 epochs)"),
    (12, 20, "Llama 3.3 best circuit (enc=out=12, 20 epochs)"),
    (9, 20, "tool default (enc=out=9), 20 epochs"),
    (5, 7, "Appendix C Iteration 1 (enc=out=5, 7 epochs)"),
    (12, 7, "Llama 3.3 best circuit at the 7-epoch reseed budget"),
]


class SimpleQNN_IdentityVQC(nn.Module):
    """SimpleQNN_CustomVQC with the VQC replaced by the identity."""

    def __init__(self, k: int):
        super().__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("linear1", nn.Linear(INPUT_SIZE, k))
        self.encoder.add_module("dropout_1", nn.Dropout(p=DROPOUT))
        self.final_dense = create_final_dense(
            k, [], 1, DROPOUT, activation=nn.LeakyReLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = (torch.tanh(x) + 1) * (math.pi / 2)  # same scaling as before the VQC
        return self.final_dense(x)


def run_one(k: int, epochs: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimpleQNN_IdentityVQC(k)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = train_quantum_model(
        model,
        epochs=epochs,
        show_progress=False,
        data_save_dir="gaussian_peak_noise_0_01",
        device="cpu",
    )
    return {"test_rmse": float(out["test_RMSE"]), "trainable_total_params": n_params}


if __name__ == "__main__":
    rows = []
    for k, epochs, note in CONFIGS:
        print(f"--- k={k}, epochs={epochs}: {note}")
        for s in SEEDS:
            t0 = time.time()
            r = run_one(k, epochs, s)
            print(
                f"  seed={s}: test_rmse={r['test_rmse']:.6f} "
                f"params={r['trainable_total_params']} ({time.time() - t0:.1f}s)"
            )
            rows.append(
                {
                    "method": "classical_identity_vqc",
                    "q_enc_size": k,
                    "q_out_size": k,
                    "epochs": epochs,
                    "seed": s,
                    "trainable_total_params": r["trainable_total_params"],
                    "test_rmse": r["test_rmse"],
                    "matches": note,
                }
            )

    df = pd.DataFrame(rows)
    print("\nSUMMARY (5-seed mean +/- std)")
    summary = (
        df.groupby(["q_enc_size", "epochs", "trainable_total_params"])["test_rmse"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    print(summary.to_string(index=False))

    csv_path = RERUN_DIR / "classical_baseline_simpleqnn.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
