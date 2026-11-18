"""
Retrain best LLM agent circuits over 5 seeds (PyTorch path).

Diagnostic counterpart to `retrain_agent_best.py` (which used the JAX path).
This script reseeds the same best circuits but through the ORIGINAL PyTorch
training path used by the agent runs: `SimpleQNN_CustomVQC` +
`train_quantum_model`. Unlike the JAX reseed, it uses each circuit's ORIGINAL
epoch count (read from the best run's parquet), not a fixed 7.

Goal: measure how much of the large PyTorch->JAX reseed gap is a framework/setup
artifact rather than a real property of the circuits.

Run from repo root:
    python -m experiments.retrain_agent_best_pytorch
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import polars as pl
import torch

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from vqc_search.models.Quanvolutional import train_quantum_model
from vqc_search.models.SimpleQNN import SimpleQNN_CustomVQC
from vqc_search.utils.quantum_circuit import create_pennylane_circuit

INPUT_SIZE = 21
DROPOUT = 0.1
RESEED_SEEDS = [0, 1, 2, 3, 4]

# If set to an int, override each circuit's original epoch count with this fixed
# value (matches the baselines / JAX reseed, isolating the framework effect from
# the epoch-budget effect). Set to None to use each circuit's original epochs.
FIXED_EPOCHS = 7

# Same session mapping and best-circuit selection as retrain_agent_best.py.
AGENT_SESSIONS = {
    # Only the sessions reported in the paper are in this repository. When the
    # experiment was run, earlier exploratory sessions of the same agents were also
    # searched; the best circuit came from the sessions below in both cases.
    "sonnet_3.7": {
        "dirs": [
            "SimpleQNN_runs_log_Sonnet_3_7_2025_11_17_02",
        ],
        "label": "Agent (Sonnet 3.7)",
    },
    "llama_3.3": {
        "dirs": [
            "SimpleQNN_runs_log_LLama3_3_2025_11_21",
        ],
        "label": "Agent (Llama 3.3)",
    },
}


def extract_best_circuits() -> dict:
    """Find the best circuit (lowest test_RMSE) per agent model."""
    runs_dir = AGENT_RUNS_DIR
    results = {}

    for model_name, info in AGENT_SESSIONS.items():
        all_dfs = []
        for d in info["dirs"]:
            session_dir = runs_dir / d
            for p in sorted(session_dir.glob("*.parquet")):
                df = pl.read_parquet(p).with_columns(pl.lit(d).alias("_session"))
                all_dfs.append(df)

        if not all_dfs:
            print(f"  WARNING: no parquets found for {model_name}")
            continue

        df = pl.concat(all_dfs)
        best = df.sort("test_RMSE").head(1)

        results[model_name] = {
            "code": best["VQC_code"][0],
            "weights_shape": best["VQC_weights_shape"][0].to_list(),
            "epochs": int(best["epochs"][0]),
            "original_rmse": float(best["test_RMSE"][0]),
            "original_total_params": int(best["n_trainable_params_total"][0]),
            "original_vqc_params": int(best["n_trainable_params_VQC"][0]),
            "num_operations": int(best["n_gates_in_VQC"][0]),
            "depth": int(best["depth_of_VQC"][0]),
            "source_session": best["_session"][0],
            "label": info["label"],
            "n_runs_total": len(df),
        }

    return results


def detect_sizes(
    code: str,
    weights_shape: list,
    original_total_params: int,
    original_vqc_params: int,
) -> dict:
    """Recover q_enc_size from param count; probe q_out_size by running the qnode.

    Original PyTorch model:
      encoder Linear(21, enc) -> 22*enc params
      vqc                     -> original_vqc_params
      final   Linear(out, 1)  -> out + 1 params
      total = 22*enc + vqc + out + 1
    """
    qnode, meta = create_pennylane_circuit(code, prefer_name="circuit")
    n_qubits = len(meta["wires"])

    dummy_inputs = np.zeros(n_qubits)
    dummy_weights = np.zeros(weights_shape)
    result = qnode(dummy_inputs, dummy_weights)
    out_size = len(result) if isinstance(result, (list, tuple, np.ndarray)) else 1

    enc_size = (original_total_params - original_vqc_params - out_size - 1) // 22
    expected = 22 * enc_size + original_vqc_params + out_size + 1
    if expected != original_total_params:
        raise ValueError(
            f"Cannot recover enc_size: {expected} != {original_total_params} "
            f"(enc={enc_size}, out={out_size}, vqc={original_vqc_params})"
        )

    return {"n_qubits": n_qubits, "q_enc_size": enc_size, "q_out_size": out_size}


def reseed_circuit_torch(
    code: str,
    weights_shape: list,
    q_enc_size: int,
    q_out_size: int,
    epochs: int,
    seed: int,
) -> dict:
    """Build a fresh SimpleQNN_CustomVQC for `seed` and train it (PyTorch path)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hparams = {
        "VQC_code": code,
        "VQC_weights_shape": weights_shape,
        "input_size": INPUT_SIZE,
        "q_enc_size": q_enc_size,
        "dropout": DROPOUT,
        "q_out_size": q_out_size,
    }
    model = SimpleQNN_CustomVQC(hparams)

    n_vqc = sum(p.numel() for p in model.vqc.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out = train_quantum_model(
        model,
        epochs=epochs,
        show_progress=False,
        data_save_dir="gaussian_peak_noise_0_01",
        device="cpu",
    )

    return {
        "test_rmse": float(out["test_RMSE"]),
        "trainable_vqc_params": int(n_vqc),
        "trainable_total_params": int(n_total),
    }


if __name__ == "__main__":
    epoch_mode = (
        f"fixed {FIXED_EPOCHS} epochs for all circuits"
        if FIXED_EPOCHS is not None
        else "each circuit's original epochs"
    )
    print("=" * 64)
    print("  Retrain best agent circuits over 5 seeds (PyTorch path)")
    print("=" * 64)
    print(f"  seeds={RESEED_SEEDS}  ({epoch_mode})")
    print()

    circuits = extract_best_circuits()

    print("Best circuits + recovered sizes:")
    for name, info in circuits.items():
        sizes = detect_sizes(
            info["code"],
            info["weights_shape"],
            info["original_total_params"],
            info["original_vqc_params"],
        )
        info.update(sizes)
        print(
            f"  {name:12s} RMSE={info['original_rmse']:.4f}  epochs={info['epochs']}  "
            f"n_qubits={sizes['n_qubits']}  enc={sizes['q_enc_size']}  "
            f"out={sizes['q_out_size']}  (from {info['source_session']})"
        )
    print()

    all_results = []
    for name, info in circuits.items():
        eff_epochs = FIXED_EPOCHS if FIXED_EPOCHS is not None else info["epochs"]
        print(f"{'=' * 64}")
        print(
            f"  {info['label']}  (n_qubits={info['n_qubits']}, "
            f"epochs={eff_epochs}, original RMSE={info['original_rmse']:.4f})"
        )
        print(f"{'=' * 64}")

        for s in RESEED_SEEDS:
            t0 = time.time()
            r = reseed_circuit_torch(
                info["code"],
                info["weights_shape"],
                q_enc_size=info["q_enc_size"],
                q_out_size=info["q_out_size"],
                epochs=eff_epochs,
                seed=s,
            )
            dt = time.time() - t0
            print(f"  seed={s}: test_rmse={r['test_rmse']:.6f}  ({dt:.1f}s)")

            all_results.append({
                "method": "agent_best_reseed_pytorch",
                "agent_model": name,
                "n_qubits": info["n_qubits"],
                "weights_shape": str(info["weights_shape"]),
                "q_enc_size": info["q_enc_size"],
                "q_out_size": info["q_out_size"],
                "trainable_vqc_params": r["trainable_vqc_params"],
                "trainable_total_params": r["trainable_total_params"],
                "epochs": eff_epochs,
                "orig_epochs": info["epochs"],
                "seed": s,
                "test_rmse": r["test_rmse"],
                "num_operations": info["num_operations"],
                "depth": info["depth"],
                "original_rmse": info["original_rmse"],
                "source_session": info["source_session"],
            })

    df = pd.DataFrame(all_results)

    print(f"\n{'=' * 64}")
    print("  SUMMARY  (original single run | PyTorch 5-seed mean +/- std)")
    print(f"{'=' * 64}")
    for name, info in circuits.items():
        subset = df[df["agent_model"] == name]
        mean = subset["test_rmse"].mean()
        std = subset["test_rmse"].std()
        print(
            f"  {info['label']:22s}  orig {info['original_rmse']:.4f}  |  "
            f"PyTorch {mean:.4f} +/- {std:.4f}"
        )

    suffix = f"_{FIXED_EPOCHS}ep" if FIXED_EPOCHS is not None else "_origep"
    csv_path = (
        RERUN_DIR / f"agent_best_reseed_pytorch{suffix}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")
