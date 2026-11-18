"""
Random VQC search baseline for SimpleQNN Gaussian-peak regression.

Samples random circuits from the same structural family the LLM agents use,
trains each under IDENTICAL conditions, and reports results. The control
experiment: if the LLM policy is doing useful work, it should beat uniform
sampling over the same action space at the same evaluation budget.

Search space A (per circuit):
  - n_qubits        ~ Uniform{2, ..., 14}  (matches agent freedom)
  - n_layers        ~ Uniform{1, 2, 3, 4, 5}
  - n_rots_per_qubit ~ Uniform{1, 2, 3}  (RY | RX,RY | RX,RY,RZ)
  - entangling[l]   ~ Uniform{ring_cnot, alt_cnot, ring_cz}  per layer

Training conditions match the agent path (train_quantum_model in
Quanvolutional.py) exactly:
  n_train=150, batch_size=16, Adam(lr=0.10, weight_decay=1e-5),
  MultiStepLR([3,8,15], gamma=0.2), MSE loss, 7 epochs.

Uses JAX backend (jax.vmap + jax.jit on the VQC) for speed.
Pure functional JAX + optax — no flax dependency.

Run from repo root:
    python -m experiments.random_search_baseline
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

import jax
import jax.core

# PennyLane 0.45 uses jax.core.is_concrete which was removed in JAX >=0.4.28.
# Polyfill: a Tracer is always abstract; everything else is concrete.
if not hasattr(jax.core, "is_concrete"):
    jax.core.is_concrete = lambda x: not isinstance(x, jax.core.Tracer)

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message=".*complex128.*")

import h5py as h5
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pennylane as qml

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from vqc_search.utils.quantum_circuit import (
    create_pennylane_circuit,
    create_pennylane_circuit_jax,
)

# ---------------------------------------------------------------------------
# Constants — must match agent runs exactly
# ---------------------------------------------------------------------------
QUBIT_RANGE = (2, 14)
INPUT_SIZE = 21
DROPOUT = 0.1
EPOCHS = 7
N_SEARCH = 26
RESEED_SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 16
LR = 0.10
WEIGHT_DECAY = 1e-5
LR_MILESTONES = [3, 8, 15]
LR_GAMMA = 0.2
N_TRAIN = 150
N_VAL = 250
N_TEST = 500
DATA_DIR_NAME = "gaussian_peak_noise_0_01"

ROTATION_OPTIONS = {
    1: ("RY",),
    2: ("RX", "RY"),
    3: ("RX", "RY", "RZ"),
}

ENTANGLING_OPTIONS = ["ring_cnot", "alt_cnot", "ring_cz"]


# ---------------------------------------------------------------------------
# Circuit code-string generator
# ---------------------------------------------------------------------------

def _entangling_block(pattern: str) -> str:
    if pattern == "ring_cnot":
        return "                qml.CNOT(wires=[i, (i + 1) % n_qubits])"
    elif pattern == "alt_cnot":
        return (
            "                if i < n_qubits - 1:\n"
            "                    qml.CNOT(wires=[i, i + 1])"
        )
    elif pattern == "ring_cz":
        return "                qml.CZ(wires=[i, (i + 1) % n_qubits])"
    raise ValueError(f"Unknown entangling pattern: {pattern}")


def generate_random_circuit(
    rng: np.random.Generator,
) -> Tuple[str, dict]:
    n_qubits = int(rng.integers(QUBIT_RANGE[0], QUBIT_RANGE[1] + 1))
    n_layers = int(rng.integers(1, 6))  # 1..5
    n_rots = int(rng.choice([1, 2, 3]))
    gate_names = ROTATION_OPTIONS[n_rots]
    entangling_per_layer = [rng.choice(ENTANGLING_OPTIONS) for _ in range(n_layers)]

    weights_shape = [n_layers, n_qubits, n_rots]

    rot_lines = "\n".join(
        f"            qml.{g}(weights[l, i, {r}], wires=i)"
        for r, g in enumerate(gate_names)
    )

    ent_cases = []
    for l_idx, pat in enumerate(entangling_per_layer):
        ent_block = _entangling_block(pat)
        ent_cases.append(
            f"        if l == {l_idx}:\n"
            f"            for i in range(n_qubits):\n"
            f"{ent_block}"
        )
    ent_section = "\n".join(ent_cases)

    code = f"""\
n_qubits = {n_qubits}
n_layers = {n_layers}
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev)
def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for l in range(n_layers):
        for i in range(n_qubits):
{rot_lines}
{ent_section}
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
"""

    meta = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_rots": n_rots,
        "gate_names": ",".join(gate_names),
        "entangling": ",".join(entangling_per_layer),
        "weights_shape": weights_shape,
    }
    return code, meta


# ---------------------------------------------------------------------------
# Circuit validation and specs
# ---------------------------------------------------------------------------

def get_circuit_specs(code: str, weights_shape: list, n_qubits: int | None = None) -> dict:
    qnode, _ = create_pennylane_circuit(code)
    if n_qubits is None:
        import re
        m = re.search(r'n_qubits\s*=\s*(\d+)', code)
        n_qubits = int(m.group(1)) if m else weights_shape[1]
    dummy_inputs = np.zeros(n_qubits)
    dummy_weights = np.zeros(weights_shape)
    specs = qml.specs(qnode)(dummy_inputs, dummy_weights)
    r = specs.resources
    return {
        "num_operations": r.num_gates,
        "depth": r.depth,
    }


# ---------------------------------------------------------------------------
# Data loading (matches train_quantum_model in Quanvolutional.py)
# ---------------------------------------------------------------------------

def _load_data():
    data_dir = DATA_DIR / DATA_DIR_NAME
    with h5.File(data_dir / "train.h5", "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]
    with h5.File(data_dir / "test.h5", "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]

    X_train = X_train_full[:N_TRAIN]
    y_train = y_train_full[:N_TRAIN]
    X_val = X_train_full[N_TRAIN:N_TRAIN + N_VAL]
    y_val = y_train_full[N_TRAIN:N_TRAIN + N_VAL]
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

    return (
        jnp.array(X_train, dtype=jnp.float32),
        jnp.array(y_train, dtype=jnp.float32),
        jnp.array(X_val, dtype=jnp.float32),
        jnp.array(y_val, dtype=jnp.float32),
        jnp.array(X_test, dtype=jnp.float32),
        jnp.array(y_test, dtype=jnp.float32),
    )


_DATA_CACHE = None

def load_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = _load_data()
    return _DATA_CACHE


# ---------------------------------------------------------------------------
# Pure-functional JAX model (mirrors SimpleQNN_CustomVQC architecture)
#
# Architecture:
#   encoder: Linear(21, enc_size) -> Dropout(0.1)
#   vqc input scaling: (tanh(x) + 1) * pi/2  -> [0, pi]
#   vqc: vmapped QNode
#   final: Linear(out_size, 1) -> sigmoid
# ---------------------------------------------------------------------------

def init_params(key, vqc_weights_shape, enc_size, out_size):
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "encoder_w": jax.random.normal(k1, (INPUT_SIZE, enc_size)) * 0.01,
        "encoder_b": jnp.zeros(enc_size),
        "vqc_weights": jax.random.normal(k2, tuple(vqc_weights_shape)) * 0.01,
        "final_w": jax.random.normal(k3, (out_size, 1)) * 0.01,
        "final_b": jnp.zeros(1),
    }
    return params


def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))


def count_vqc_params(params):
    return params["vqc_weights"].size


def forward(params, x, vqc_fn, *, training=False, dropout_key=None):
    x = x @ params["encoder_w"] + params["encoder_b"]
    if training and dropout_key is not None:
        mask = jax.random.bernoulli(dropout_key, 1.0 - DROPOUT, x.shape)
        x = x * mask / (1.0 - DROPOUT)
    x = (jnp.tanh(x) + 1.0) * (jnp.pi / 2.0)
    x = vqc_fn(x, params["vqc_weights"])
    x = jnp.stack(x, axis=-1)  # list of (B,) -> (B, n_qubits)
    x = x @ params["final_w"] + params["final_b"]
    x = jax.nn.sigmoid(x)
    return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop (matches train_quantum_model hyperparameters exactly)
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


def _detect_n_qubits(code: str, weights_shape: list) -> int:
    import re
    m = re.search(r'n_qubits\s*=\s*(\d+)', code)
    if m:
        return int(m.group(1))
    m2 = re.search(r'wires\s*=\s*(\d+)', code)
    if m2:
        return int(m2.group(1))
    return weights_shape[1]


def train_circuit_jax(code: str, weights_shape: list, seed: int,
                      n_qubits: int | None = None) -> dict:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    if n_qubits is None:
        n_qubits = _detect_n_qubits(code, weights_shape)
    enc_size = n_qubits
    out_size = n_qubits

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, weights_shape, enc_size, out_size)

    n_vqc = count_vqc_params(params)
    n_total = count_params(params)

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
            xb = X_shuf[i:i + BATCH_SIZE]
            yb = y_shuf[i:i + BATCH_SIZE]
            dropout_key, subkey = jax.random.split(dropout_key)
            params, opt_state, _ = train_step(params, opt_state, xb, yb, subkey)

    # Test: per-batch MSE -> weighted mean -> sqrt (matches PyTorch path)
    test_mse_sum = 0.0
    test_n = 0
    for i in range(0, N_TEST, BATCH_SIZE):
        xb = X_test[i:i + BATCH_SIZE]
        yb = y_test[i:i + BATCH_SIZE]
        pred = forward(params, xb, vqc_fn, training=False)
        batch_n = len(yb)
        batch_mse = float(jnp.mean((pred - yb) ** 2))
        test_mse_sum += batch_mse * batch_n
        test_n += batch_n

    test_rmse = float(np.sqrt(test_mse_sum / test_n))

    return {
        "test_rmse": test_rmse,
        "trainable_vqc_params": n_vqc,
        "trainable_total_params": n_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    search_rng = np.random.default_rng(seed=42)
    all_results = []
    circuit_bank = {}
    valid_count = 0
    attempt = 0

    print(f"Random VQC search: sampling {N_SEARCH} circuits from action space A")
    print(f"  n_qubits={QUBIT_RANGE[0]}..{QUBIT_RANGE[1]}, epochs={EPOCHS}, dataset={DATA_DIR_NAME}")
    print(f"  backend=JAX, lr={LR}, weight_decay={WEIGHT_DECAY}")
    print(f"  schedule=MultiStepLR({LR_MILESTONES}, gamma={LR_GAMMA})")
    print(f"  n_train={N_TRAIN}, batch_size={BATCH_SIZE}\n")

    pbar = tqdm(total=N_SEARCH, desc="Random search", unit="circuit")
    while valid_count < N_SEARCH:
        attempt += 1
        code, meta = generate_random_circuit(search_rng)

        try:
            specs = get_circuit_specs(code, meta["weights_shape"], meta["n_qubits"])
        except Exception as e:
            tqdm.write(f"  [attempt {attempt}] circuit compile failed: {e}")
            continue

        valid_count += 1
        circuit_bank[valid_count] = (code, meta, specs)

        label = (
            f"q={meta['n_qubits']} L={meta['n_layers']} rots={meta['n_rots']} "
            f"ent={meta['entangling']}"
        )

        t0 = time.time()
        train_result = train_circuit_jax(code, meta["weights_shape"], seed=0,
                                         n_qubits=meta["n_qubits"])
        dt = time.time() - t0

        test_rmse = train_result["test_rmse"]
        pbar.set_postfix(rmse=f"{test_rmse:.4f}", q=meta["n_qubits"])
        pbar.update(1)
        tqdm.write(f"  [{valid_count}/{N_SEARCH}] {label} -> rmse={test_rmse:.6f} ({dt:.1f}s)")

        row = {
            "method": "random",
            "circuit_id": valid_count,
            "n_qubits": meta["n_qubits"],
            "n_layers": meta["n_layers"],
            "n_rots_per_qubit": meta["n_rots"],
            "gate_names": meta["gate_names"],
            "entangling": meta["entangling"],
            "trainable_vqc_params": train_result["trainable_vqc_params"],
            "trainable_total_params": train_result["trainable_total_params"],
            "seed": 0,
            "test_rmse": test_rmse,
            "num_operations": specs["num_operations"],
            "depth": specs["depth"],
        }
        all_results.append(row)
    pbar.close()

    df = pd.DataFrame(all_results)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  RANDOM SEARCH SUMMARY ({N_SEARCH} circuits, seed=0 each)")
    print(f"{'='*60}")
    print(f"  Best  test_rmse: {df['test_rmse'].min():.6f}")
    print(f"  Median test_rmse: {df['test_rmse'].median():.6f}")
    print(f"  Mean  test_rmse: {df['test_rmse'].mean():.6f} +/- {df['test_rmse'].std():.6f}")

    # --- Retrain best circuit over 5 seeds ---
    best_idx = df["test_rmse"].idxmin()
    best_row = df.iloc[best_idx]
    best_circuit_id = int(best_row["circuit_id"])
    best_code, best_meta, best_specs = circuit_bank[best_circuit_id]

    print(f"\n  Retraining best circuit (id={best_circuit_id}) over {len(RESEED_SEEDS)} seeds...")

    reseed_results = []
    for s in tqdm(RESEED_SEEDS, desc="Reseed best", unit="seed"):
        t0 = time.time()
        r = train_circuit_jax(best_code, best_meta["weights_shape"], seed=s,
                              n_qubits=best_meta["n_qubits"])
        dt = time.time() - t0
        tqdm.write(f"    seed={s}: test_rmse={r['test_rmse']:.6f}  ({dt:.1f}s)")
        reseed_row = {
            "method": "random_best_reseed",
            "circuit_id": best_circuit_id,
            "n_qubits": best_meta["n_qubits"],
            "n_layers": best_meta["n_layers"],
            "n_rots_per_qubit": best_meta["n_rots"],
            "gate_names": best_meta["gate_names"],
            "entangling": best_meta["entangling"],
            "trainable_vqc_params": r["trainable_vqc_params"],
            "trainable_total_params": r["trainable_total_params"],
            "seed": s,
            "test_rmse": r["test_rmse"],
            "num_operations": best_specs["num_operations"],
            "depth": best_specs["depth"],
        }
        reseed_results.append(reseed_row)

    df_reseed = pd.DataFrame(reseed_results)
    print(f"\n  Best circuit 5-seed: mean={df_reseed['test_rmse'].mean():.6f} "
          f"+/- {df_reseed['test_rmse'].std():.6f}")

    # --- Save ---
    df_all = pd.concat([df, df_reseed], ignore_index=True)
    csv_path = RERUN_DIR / "random_search_baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")
