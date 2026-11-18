"""
Retrain best LLM agent circuits over 5 seeds (JAX path).

Extracts the single best VQC from each agent session (Sonnet 3.7, Llama 3.3), retrains each over 5 seeds using the same JAX training loop as the
random/evolutionary baselines, and saves results to CSV.

The JAX training path mirrors SimpleQNN_CustomVQC exactly:
  encoder: Linear(21, enc_size) -> Dropout(0.1)
  vqc scaling: (tanh(x) + 1) * pi/2 -> [0, pi]
  vqc: vmapped QNode
  final: Linear(out_size, 1) -> sigmoid

enc_size and out_size are detected per circuit (agent circuits use varying
qubit counts and output sizes).

Run from repo root:
    python -m experiments.retrain_agent_best
"""

import sys
import time
import warnings
from pathlib import Path

import jax
import jax.core

if not hasattr(jax.core, "is_concrete"):
    jax.core.is_concrete = lambda x: not isinstance(x, jax.core.Tracer)

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message=".*complex128.*")

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pennylane as qml
import polars as pl

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from vqc_search.utils.quantum_circuit import (
    create_pennylane_circuit,
    create_pennylane_circuit_jax,
)

# ---------------------------------------------------------------------------
# Constants — must match agent runs / baselines exactly
# ---------------------------------------------------------------------------
INPUT_SIZE = 21
DROPOUT = 0.1
EPOCHS = 7
BATCH_SIZE = 16
LR = 0.10
WEIGHT_DECAY = 1e-5
LR_MILESTONES = [3, 8, 15]
LR_GAMMA = 0.2
N_TRAIN = 150
N_VAL = 250
N_TEST = 500
DATA_DIR_NAME = "gaussian_peak_noise_0_01"
RESEED_SEEDS = [0, 1, 2, 3, 4]

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


# ---------------------------------------------------------------------------
# Circuit extraction and probing
# ---------------------------------------------------------------------------

def extract_best_circuits() -> dict:
    """Find the best circuit (lowest test_RMSE) per agent model."""
    runs_dir = AGENT_RUNS_DIR
    results = {}

    for model_name, info in AGENT_SESSIONS.items():
        all_dfs = []
        for d in info["dirs"]:
            session_dir = runs_dir / d
            for p in sorted(session_dir.glob("*.parquet")):
                df = pl.read_parquet(p).with_columns(
                    pl.lit(d).alias("_session")
                )
                all_dfs.append(df)

        if not all_dfs:
            print(f"  WARNING: no parquets found for {model_name}")
            continue

        df = pl.concat(all_dfs)
        best = df.sort("test_RMSE").head(1)

        results[model_name] = {
            "code": best["VQC_code"][0],
            "weights_shape": best["VQC_weights_shape"][0].to_list(),
            "original_rmse": float(best["test_RMSE"][0]),
            "original_total_params": int(best["n_trainable_params_total"][0]),
            "original_vqc_params": int(best["n_trainable_params_VQC"][0]),
            "source_session": best["_session"][0],
            "label": info["label"],
            "n_runs_total": len(df),
        }

    return results


def detect_circuit_sizes(
    code: str,
    weights_shape: list,
    original_total_params: int,
    original_vqc_params: int,
) -> dict:
    """Detect enc_size and out_size, recovering enc_size from original param count.

    The original PyTorch model architecture is:
      encoder: Linear(21, enc_size) -> 22 * enc_size params
      vqc: original_vqc_params
      final:  Linear(out_size, 1) -> out_size + 1 params
      total = 22 * enc_size + original_vqc_params + out_size + 1

    We know total and vqc params from the parquet. We probe out_size by running
    the circuit, then solve for enc_size.
    """
    qnode_jax, meta = create_pennylane_circuit_jax(code, prefer_name="circuit")
    n_qubits = len(meta["wires"])

    dummy_weights = jnp.zeros(weights_shape, dtype=jnp.float32)
    dummy_inputs = jnp.zeros(n_qubits, dtype=jnp.float32)
    result = qnode_jax(dummy_inputs, dummy_weights)
    out_size = len(result) if isinstance(result, (list, tuple)) else 1

    enc_size = (original_total_params - original_vqc_params - out_size - 1) // 22
    expected_total = 22 * enc_size + original_vqc_params + out_size + 1
    if expected_total != original_total_params:
        raise ValueError(
            f"Cannot recover enc_size: {expected_total} != {original_total_params} "
            f"(enc_size={enc_size}, out_size={out_size}, vqc={original_vqc_params})"
        )

    return {
        "n_qubits": n_qubits,
        "enc_size": enc_size,
        "out_size": out_size,
    }


def get_circuit_specs(code: str, weights_shape: list, n_inputs: int) -> dict:
    """Gate count and depth via qml.specs (non-JAX path)."""
    qnode, _ = create_pennylane_circuit(code)
    dummy_inputs = np.zeros(n_inputs)
    dummy_weights = np.zeros(weights_shape)
    specs = qml.specs(qnode)(dummy_inputs, dummy_weights)
    r = specs.resources
    return {"num_operations": r.num_gates, "depth": r.depth}


# ---------------------------------------------------------------------------
# Data loading (identical to random_search_baseline)
# ---------------------------------------------------------------------------

_DATA_CACHE = None


def load_data():
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    data_dir = DATA_DIR / DATA_DIR_NAME
    import h5py as h5

    with h5.File(data_dir / "train.h5", "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]
    with h5.File(data_dir / "test.h5", "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]

    X_train = X_train_full[:N_TRAIN]
    y_train = y_train_full[:N_TRAIN]
    X_val = X_train_full[N_TRAIN : N_TRAIN + N_VAL]
    y_val = y_train_full[N_TRAIN : N_TRAIN + N_VAL]
    X_test = X_test_full[:N_TEST]
    y_test = y_test_full[:N_TEST]

    def norm(X):
        return (X - X.min(axis=1, keepdims=True)) / (
            X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8
        )

    X_train, X_val, X_test = norm(X_train), norm(X_val), norm(X_test)
    y_train = y_train[:, 0] if y_train.ndim == 2 else y_train
    y_val = y_val[:, 0] if y_val.ndim == 2 else y_val
    y_test = y_test[:, 0] if y_test.ndim == 2 else y_test

    _DATA_CACHE = (
        jnp.array(X_train, dtype=jnp.float32),
        jnp.array(y_train, dtype=jnp.float32),
        jnp.array(X_val, dtype=jnp.float32),
        jnp.array(y_val, dtype=jnp.float32),
        jnp.array(X_test, dtype=jnp.float32),
        jnp.array(y_test, dtype=jnp.float32),
    )
    return _DATA_CACHE


# ---------------------------------------------------------------------------
# Dynamic-size JAX model
# ---------------------------------------------------------------------------

def _make_lr_schedule(base_lr: float) -> optax.Schedule:
    steps_per_epoch = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE
    boundaries = [m * steps_per_epoch for m in LR_MILESTONES]
    lr_values = [base_lr]
    for _ in LR_MILESTONES:
        lr_values.append(lr_values[-1] * LR_GAMMA)
    return optax.join_schedules(
        [optax.constant_schedule(v) for v in lr_values],
        boundaries,
    )


def init_params(key, enc_size, vqc_weights_shape, out_size):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "encoder_w": jax.random.normal(k1, (INPUT_SIZE, enc_size)) * 0.01,
        "encoder_b": jnp.zeros(enc_size),
        "vqc_weights": jax.random.normal(k2, tuple(vqc_weights_shape)) * 0.01,
        "final_w": jax.random.normal(k3, (out_size, 1)) * 0.01,
        "final_b": jnp.zeros(1),
    }


def forward(params, x, vqc_fn, *, training=False, dropout_key=None):
    x = x @ params["encoder_w"] + params["encoder_b"]
    if training and dropout_key is not None:
        mask = jax.random.bernoulli(dropout_key, 1.0 - DROPOUT, x.shape)
        x = x * mask / (1.0 - DROPOUT)
    x = (jnp.tanh(x) + 1.0) * (jnp.pi / 2.0)
    x = vqc_fn(x, params["vqc_weights"])
    x = jnp.stack(x, axis=-1)
    x = x @ params["final_w"] + params["final_b"]
    x = jax.nn.sigmoid(x)
    return x.squeeze(-1)


def retrain_circuit_jax(
    code: str,
    weights_shape: list,
    seed: int,
    enc_size: int,
    out_size: int,
) -> dict:
    """Train one circuit for one seed. Returns test_rmse and param counts."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, enc_size, weights_shape, out_size)

    n_vqc = params["vqc_weights"].size
    n_total = sum(p.size for p in jax.tree.leaves(params))

    qnode, _ = create_pennylane_circuit_jax(code, prefer_name="circuit")
    vqc_fn = jax.jit(jax.vmap(qnode, in_axes=(0, None)))

    lr_schedule = _make_lr_schedule(LR)
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(params)

    def loss_fn(params, xb, yb, dropout_key):
        pred = forward(params, xb, vqc_fn, training=True, dropout_key=dropout_key)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def train_step(params, opt_state, xb, yb, dropout_key):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb, dropout_key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for epoch in range(EPOCHS):
        key, shuffle_key, dropout_key = jax.random.split(key, 3)
        perm = jax.random.permutation(shuffle_key, N_TRAIN)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        for i in range(0, N_TRAIN, BATCH_SIZE):
            xb = X_shuf[i : i + BATCH_SIZE]
            yb = y_shuf[i : i + BATCH_SIZE]
            dropout_key, subkey = jax.random.split(dropout_key)
            params, opt_state, _ = train_step(params, opt_state, xb, yb, subkey)

    test_mse_sum = 0.0
    test_n = 0
    for i in range(0, N_TEST, BATCH_SIZE):
        xb = X_test[i : i + BATCH_SIZE]
        yb = y_test[i : i + BATCH_SIZE]
        pred = forward(params, xb, vqc_fn, training=False)
        batch_n = len(yb)
        batch_mse = float(jnp.mean((pred - yb) ** 2))
        test_mse_sum += batch_mse * batch_n
        test_n += batch_n

    test_rmse = float(np.sqrt(test_mse_sum / test_n))

    return {
        "test_rmse": test_rmse,
        "trainable_vqc_params": int(n_vqc),
        "trainable_total_params": int(n_total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Retrain best agent circuits over 5 seeds (JAX path)")
    print("=" * 60)
    print(f"  epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"  seeds={RESEED_SEEDS}")
    print()

    print("Extracting best circuits from agent sessions...")
    circuits = extract_best_circuits()
    print()

    for name, info in circuits.items():
        print(f"  {name}: RMSE={info['original_rmse']:.6f} "
              f"(from {info['source_session']}, {info['n_runs_total']} total runs)")
    print()

    print("Probing circuit sizes...")
    circuit_meta = {}
    for name, info in circuits.items():
        sizes = detect_circuit_sizes(
            info["code"],
            info["weights_shape"],
            info["original_total_params"],
            info["original_vqc_params"],
        )
        circuit_meta[name] = sizes
        specs = get_circuit_specs(
            info["code"], info["weights_shape"], sizes["enc_size"]
        )
        circuit_meta[name].update(specs)
        print(f"  {name}: n_qubits={sizes['n_qubits']}, "
              f"enc_size={sizes['enc_size']}, out_size={sizes['out_size']}, "
              f"gates={specs['num_operations']}, depth={specs['depth']}")
    print()

    all_results = []

    for name, info in circuits.items():
        sizes = circuit_meta[name]
        print(f"\n{'=' * 60}")
        print(f"  {info['label']}  (n_qubits={sizes['n_qubits']}, "
              f"original RMSE={info['original_rmse']:.6f})")
        print(f"{'=' * 60}")

        for s in RESEED_SEEDS:
            t0 = time.time()
            r = retrain_circuit_jax(
                info["code"],
                info["weights_shape"],
                seed=s,
                enc_size=sizes["enc_size"],
                out_size=sizes["out_size"],
            )
            dt = time.time() - t0
            print(f"  seed={s}: test_rmse={r['test_rmse']:.6f}  ({dt:.1f}s)")

            all_results.append({
                "method": "agent_best_reseed",
                "agent_model": name,
                "n_qubits": sizes["n_qubits"],
                "weights_shape": str(info["weights_shape"]),
                "enc_size": sizes["enc_size"],
                "out_size": sizes["out_size"],
                "trainable_vqc_params": r["trainable_vqc_params"],
                "trainable_total_params": r["trainable_total_params"],
                "seed": s,
                "test_rmse": r["test_rmse"],
                "num_operations": sizes["num_operations"],
                "depth": sizes["depth"],
                "original_rmse": info["original_rmse"],
                "source_session": info["source_session"],
            })

    df = pd.DataFrame(all_results)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name in circuits:
        subset = df[df["agent_model"] == name]
        mean = subset["test_rmse"].mean()
        std = subset["test_rmse"].std()
        orig = circuits[name]["original_rmse"]
        print(f"  {circuits[name]['label']:25s}  "
              f"JAX 5-seed: {mean:.6f} ± {std:.6f}  "
              f"(original PyTorch: {orig:.6f})")

    csv_path = RERUN_DIR / "agent_best_reseed.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")
