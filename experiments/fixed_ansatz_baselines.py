"""
Fixed-ansatz baselines for SimpleQNN Gaussian-peak regression.

Trains Hardware-Efficient Ansatz (HEA) and StronglyEntanglingLayers (SEL)
under IDENTICAL conditions to the agent SimpleQNN runs, varying only the VQC.

The training path mirrors the exact code path used by the agent tool:
  tools/custom/train_SimpleQNN.py
    -> SimpleQNN_CustomVQC(hparams)
    -> train_quantum_model(model, epochs, ...) from models/Quanvolutional.py

Run from repo root:
    python -m experiments.fixed_ansatz_baselines
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from vqc_search.models.SimpleQNN import SimpleQNN_CustomVQC
from vqc_search.models.Quanvolutional import train_quantum_model

# ---------------------------------------------------------------------------
# VQC code-string generators (follow the code-string injection convention:
#   no imports, qml/np are injected, must define @qml.qnode(dev) circuit(inputs, weights))
# ---------------------------------------------------------------------------

def hea_code(n_layers: int) -> str:
    return f"""\
n_qubits = 9
n_layers = {n_layers}
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev)
def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[l, i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
"""


def sel_code(n_layers: int) -> str:
    return f"""\
n_qubits = 9
n_layers = {n_layers}
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev)
def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
"""


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

N_QUBITS = 9
Q_ENC_SIZE = 9
Q_OUT_SIZE = 9
INPUT_SIZE = 21
DROPOUT = 0.1
EPOCHS = 7
SEEDS = [0, 1, 2, 3, 4]

CONFIGS = [
    {"ansatz": "HEA", "layers": 1, "weights_shape": [1, 9],    "expected_vqc_params": 1 * 9},
    {"ansatz": "HEA", "layers": 2, "weights_shape": [2, 9],    "expected_vqc_params": 2 * 9},
    {"ansatz": "HEA", "layers": 3, "weights_shape": [3, 9],    "expected_vqc_params": 3 * 9},
    {"ansatz": "HEA", "layers": 5, "weights_shape": [5, 9],    "expected_vqc_params": 5 * 9},
    {"ansatz": "SEL", "layers": 1, "weights_shape": [1, 9, 3], "expected_vqc_params": 1 * 9 * 3},
    {"ansatz": "SEL", "layers": 2, "weights_shape": [2, 9, 3], "expected_vqc_params": 2 * 9 * 3},
    {"ansatz": "SEL", "layers": 3, "weights_shape": [3, 9, 3], "expected_vqc_params": 3 * 9 * 3},
    {"ansatz": "SEL", "layers": 5, "weights_shape": [5, 9, 3], "expected_vqc_params": 5 * 9 * 3},
]


def make_code(ansatz: str, layers: int) -> str:
    if ansatz == "HEA":
        return hea_code(layers)
    if ansatz == "SEL":
        return sel_code(layers)
    raise ValueError(f"Unknown ansatz: {ansatz}")


def run_single(config: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    code = make_code(config["ansatz"], config["layers"])
    hparams = {
        "input_size": INPUT_SIZE,
        "q_enc_size": Q_ENC_SIZE,
        "dropout": DROPOUT,
        "q_out_size": Q_OUT_SIZE,
        "VQC_code": code,
        "VQC_weights_shape": config["weights_shape"],
    }
    model = SimpleQNN_CustomVQC(hparams)

    n_vqc_params = sum(p.numel() for p in model.vqc.parameters() if p.requires_grad)
    assert n_vqc_params == config["expected_vqc_params"], (
        f"VQC param count mismatch for {config['ansatz']} L={config['layers']}: "
        f"expected {config['expected_vqc_params']}, got {n_vqc_params}"
    )

    # Exact same call as the agent tool (train_SimpleQNN.py line 143)
    results = train_quantum_model(
        model,
        epochs=EPOCHS,
        show_progress=True,
        data_save_dir="gaussian_peak_noise_0_01",
    )

    test_rmse = float(results["test_RMSE"])
    assert np.isfinite(test_rmse), f"Non-finite test RMSE: {test_rmse}"

    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "ansatz": config["ansatz"],
        "n_qubits": N_QUBITS,
        "layers": config["layers"],
        "trainable_vqc_params": n_vqc_params,
        "trainable_total_params": n_total_params,
        "seed": seed,
        "test_rmse": test_rmse,
    }


def print_config_report():
    print("\n## Training Configuration (byte-for-byte from train_quantum_model in Quanvolutional.py)")
    print(f"  Dataset:           gaussian_peak_noise_0_01")
    print(f"  n_train:           150")
    print(f"  n_val:             250")
    print(f"  n_test:            500")
    print(f"  Optimizer:         Adam(lr=0.10, weight_decay=1e-5)")
    print(f"  LR schedule:       MultiStepLR(milestones=[3, 8, 15], gamma=0.2)")
    print(f"  Loss:              MSE")
    print(f"  Batch size:        16")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Target:            peak position only (y[:, 0])")
    print(f"  Input norm:        per-sample min-max to [0, 1]")
    print(f"  VQC input scaling: (tanh(x)+1) * pi/2  ->  [0, pi]")
    print(f"  Encoder:           Linear({INPUT_SIZE}, {Q_ENC_SIZE}) -> Dropout({DROPOUT})")
    print(f"  Final head:        Linear({Q_OUT_SIZE}, 1) -> Sigmoid")
    print(f"  Seeds:             {SEEDS}")
    print()
    print("## Discrepancies vs. task expectations (resolved by inspecting actual code)")
    print("  - n_train = 150, NOT 1000.  The agent tool calls train_quantum_model()")
    print("    from Quanvolutional.py, which hardcodes n_train=150.  The n_train=1000")
    print("    value appears only in the SimpleQNN.py standalone __main__ block, which")
    print("    is demo code not used by the agent.")
    print("  - lr = 0.10, NOT 0.01.  train_quantum_model uses Adam(lr=0.10).")
    print("    The lr=0.01 value appears in SimpleQNN_Normal.py (ablation), not the")
    print("    agent path.")
    print("  - LR schedule: MultiStepLR(milestones=[3,8,15], gamma=0.2), NOT manual")
    print("    decay at epochs 8/12/15.  The manual schedule is in SimpleQNN_Normal.py")
    print("    and the SimpleQNN.py __main__ block; the agent path uses MultiStepLR.")
    print("  - epochs = 7.  All entries in SimpleQNN_runs_log.csv and")
    print("    SimpleQNN_runs_log_LLama3_3_2025_11_14.csv show epochs=7.")


if __name__ == "__main__":
    all_results = []
    total_runs = len(CONFIGS) * len(SEEDS)
    run_idx = 0

    for config in CONFIGS:
        for seed in SEEDS:
            run_idx += 1
            label = f"{config['ansatz']} L={config['layers']} seed={seed}"
            print(f"\n{'='*60}")
            print(f"  [{run_idx}/{total_runs}] {label}")
            print(f"{'='*60}")
            t0 = time.time()
            row = run_single(config, seed)
            dt = time.time() - t0
            print(f"  -> test_rmse = {row['test_rmse']:.6f}  ({dt:.1f}s)")
            all_results.append(row)

    df = pd.DataFrame(all_results)

    csv_path = RERUN_DIR / "fixed_ansatz_baselines.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    summary = (
        df.groupby(["ansatz", "n_qubits", "layers", "trainable_vqc_params"])["test_rmse"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "test_rmse_mean", "std": "test_rmse_std"})
    )

    print("\n## Summary (5 seeds per configuration)")
    print(
        "| ansatz | n_qubits | layers | trainable_vqc_params | test_rmse_mean | test_rmse_std |"
    )
    print(
        "|--------|----------|--------|----------------------|----------------|---------------|"
    )
    for _, r in summary.iterrows():
        print(
            f"| {r['ansatz']:6s} | {int(r['n_qubits']):8d} | {int(r['layers']):6d} | "
            f"{int(r['trainable_vqc_params']):20d} | "
            f"{r['test_rmse_mean']:14.6f} | {r['test_rmse_std']:13.6f} |"
        )

    print_config_report()
