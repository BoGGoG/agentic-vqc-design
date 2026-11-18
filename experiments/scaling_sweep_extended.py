"""
Extended qubit/depth scaling sweep for SimpleQNN.

Trains HEA circuits at varying n_qubits and n_layers with 5 seeds per
configuration.  Adds failsafes for memory, timeout, and runtime errors.

Swept parameters:
  n_qubits in {5, 7, 9, 12, 16, 18, 20}
  n_layers in {1, 2, 3, 5, 8}
  seeds    in {0, 1, 2, 3, 4}
  (175 configurations total)

Run from repo root:
    python -m experiments.scaling_sweep_extended
"""

import sys
import time
import warnings
from pathlib import Path

from tqdm import tqdm

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
import h5py as h5

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from vqc_search.utils.quantum_circuit import (
    create_pennylane_circuit,
    create_pennylane_circuit_jax,
)

# ---------------------------------------------------------------------------
# Training constants (match agent path)
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

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
QUBIT_RANGE = [5, 7, 9, 12, 16, 18, 20]
LAYER_RANGE = [1, 2, 3, 5, 8]
SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Failsafe thresholds
# ---------------------------------------------------------------------------
MEMORY_LIMIT_BYTES = 40 * (1024 ** 3)  # 40 GB
TIMEOUT_SECONDS = 600  # 10 minutes per run


def estimate_peak_memory(n_qubits: int) -> int:
    """Rough estimate of peak training memory in bytes.

    Accounts for statevector (2^n complex128), batching (×BATCH_SIZE),
    and a ×10 safety factor for gradients, optimizer state, and JAX
    intermediates.
    """
    statevector_bytes = (2 ** n_qubits) * 16  # complex128
    return statevector_bytes * BATCH_SIZE * 10


# ---------------------------------------------------------------------------
# HEA code generator
# ---------------------------------------------------------------------------

def hea_code(n_qubits: int, n_layers: int) -> str:
    return f"""\
n_qubits = {n_qubits}
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


def get_circuit_specs(code: str, weights_shape: list) -> dict:
    qnode, _ = create_pennylane_circuit(code)
    dummy_inputs = np.zeros(weights_shape[1])
    dummy_weights = np.zeros(weights_shape)
    specs = qml.specs(qnode)(dummy_inputs, dummy_weights)
    r = specs.resources
    return {"num_operations": r.num_gates, "depth": r.depth}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_CACHE = None


def load_data():
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

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
# Model forward pass
# ---------------------------------------------------------------------------

def init_params(key, n_qubits, vqc_weights_shape):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "encoder_w": jax.random.normal(k1, (INPUT_SIZE, n_qubits)) * 0.01,
        "encoder_b": jnp.zeros(n_qubits),
        "vqc_weights": jax.random.normal(k2, tuple(vqc_weights_shape)) * 0.01,
        "final_w": jax.random.normal(k3, (n_qubits, 1)) * 0.01,
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


def train_config(n_qubits: int, n_layers: int, seed: int) -> dict:
    """Train a single HEA configuration. Returns metrics dict.

    Raises TimeoutError if wall-clock time exceeds TIMEOUT_SECONDS.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    code = hea_code(n_qubits, n_layers)
    weights_shape = [n_layers, n_qubits]

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, n_qubits, weights_shape)

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

    t_start = time.time()

    for epoch in range(EPOCHS):
        if time.time() - t_start > TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Exceeded {TIMEOUT_SECONDS}s after epoch {epoch}/{EPOCHS}"
            )

        key, shuffle_key, dropout_key = jax.random.split(key, 3)
        perm = jax.random.permutation(shuffle_key, N_TRAIN)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        for i in range(0, N_TRAIN, BATCH_SIZE):
            xb = X_shuf[i:i + BATCH_SIZE]
            yb = y_shuf[i:i + BATCH_SIZE]
            dropout_key, subkey = jax.random.split(dropout_key)
            params, opt_state, _ = train_step(params, opt_state, xb, yb, subkey)

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

def load_completed(csv_path: Path) -> set:
    """Return set of (n_qubits, n_layers, seed) already in the CSV."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["n_qubits"], df["n_layers"], df["seed"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended scaling sweep")
    parser.add_argument(
        "--retry-timeout", type=int, nargs="?", const=3600, default=None,
        metavar="SECONDS",
        help="Remove timed-out rows from CSV and re-run them. "
             "Optionally set a new timeout (default: 3600s).",
    )
    args = parser.parse_args()

    if args.retry_timeout is not None:
        TIMEOUT_SECONDS = args.retry_timeout

    csv_path = RERUN_DIR / "scaling_sweep_extended.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.retry_timeout is not None and csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        n_before = len(df_existing)
        df_existing = df_existing[df_existing["status"] != "skipped_timeout"]
        n_removed = n_before - len(df_existing)
        df_existing.to_csv(csv_path, index=False)
        print(f"Retry mode: removed {n_removed} timed-out rows, "
              f"new timeout = {TIMEOUT_SECONDS}s\n")

    completed = load_completed(csv_path)
    total = len(QUBIT_RANGE) * len(LAYER_RANGE) * len(SEEDS)

    print(f"Extended scaling sweep")
    print(f"  n_qubits = {QUBIT_RANGE}")
    print(f"  n_layers = {LAYER_RANGE}")
    print(f"  seeds    = {SEEDS}")
    print(f"  Total: {total}  |  Already done: {len(completed)}  |  "
          f"Remaining: {total - len(completed)}")
    print(f"  Memory limit: {MEMORY_LIMIT_BYTES / (1024**3):.0f} GB  |  "
          f"Timeout: {TIMEOUT_SECONDS}s per run\n")

    jobs = []
    for nq in QUBIT_RANGE:
        for nl in LAYER_RANGE:
            for seed in SEEDS:
                jobs.append((nq, nl, seed))

    n_ok = 0
    n_fail = 0
    specs_cache = {}
    memory_skipped_qubits = set()

    pbar = tqdm(jobs, desc="Sweep", unit="run",
                bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"))

    for nq, nl, seed in pbar:
        pbar.set_postfix_str(f"{nq}q {nl}L s{seed}  ok={n_ok} fail={n_fail}",
                             refresh=False)

        if (nq, nl, seed) in completed:
            n_ok += 1
            continue

        # --- memory guard ---
        if nq in memory_skipped_qubits:
            _skip = True
        else:
            peak_mem = estimate_peak_memory(nq)
            _skip = peak_mem > MEMORY_LIMIT_BYTES
            if _skip:
                memory_skipped_qubits.add(nq)
                tqdm.write(f"  SKIP n_qubits={nq}: estimated peak memory "
                           f"{peak_mem / (1024**3):.1f} GB > limit "
                           f"{MEMORY_LIMIT_BYTES / (1024**3):.0f} GB")

        if _skip:
            row = {
                "ansatz": "HEA", "n_qubits": nq, "n_layers": nl,
                "trainable_vqc_params": nl * nq,
                "trainable_total_params": None,
                "seed": seed, "test_rmse": None,
                "num_operations": None, "depth": None,
                "status": "skipped_memory", "wall_s": 0.0,
            }
            pd.DataFrame([row]).to_csv(
                csv_path, mode="a", index=False,
                header=not csv_path.exists(),
            )
            n_fail += 1
            continue

        # --- circuit specs (cached per qubit/layer combo) ---
        if (nq, nl) not in specs_cache:
            code = hea_code(nq, nl)
            weights_shape = [nl, nq]
            try:
                specs_cache[(nq, nl)] = get_circuit_specs(code, weights_shape)
            except Exception as e:
                tqdm.write(f"  Circuit compile failed for {nq}q/{nl}L: {e}")
                specs_cache[(nq, nl)] = None

        specs = specs_cache[(nq, nl)]

        if specs is None:
            row = {
                "ansatz": "HEA", "n_qubits": nq, "n_layers": nl,
                "trainable_vqc_params": nl * nq,
                "trainable_total_params": None,
                "seed": seed, "test_rmse": None,
                "num_operations": None, "depth": None,
                "status": "error", "wall_s": 0.0,
            }
            pd.DataFrame([row]).to_csv(
                csv_path, mode="a", index=False,
                header=not csv_path.exists(),
            )
            n_fail += 1
            continue

        # --- train ---
        t0 = time.time()
        try:
            result = train_config(nq, nl, seed)
            wall_s = time.time() - t0
            status = "ok"
            test_rmse = result["test_rmse"]
            n_vqc = result["trainable_vqc_params"]
            n_total = result["trainable_total_params"]
            n_ok += 1
            tqdm.write(f"  {nq}q {nl}L s{seed}  rmse={test_rmse:.4f}  "
                       f"({wall_s:.1f}s)")

        except TimeoutError as e:
            wall_s = time.time() - t0
            status = "skipped_timeout"
            test_rmse = None
            n_vqc = nl * nq
            n_total = None
            n_fail += 1
            tqdm.write(f"  {nq}q {nl}L s{seed}  TIMEOUT ({wall_s:.0f}s)")

        except Exception as e:
            wall_s = time.time() - t0
            status = "error"
            test_rmse = None
            n_vqc = nl * nq
            n_total = None
            n_fail += 1
            tqdm.write(f"  {nq}q {nl}L s{seed}  ERROR: "
                       f"{type(e).__name__}: {e}")

        row = {
            "ansatz": "HEA",
            "n_qubits": nq,
            "n_layers": nl,
            "trainable_vqc_params": n_vqc,
            "trainable_total_params": n_total,
            "seed": seed,
            "test_rmse": test_rmse,
            "num_operations": specs["num_operations"],
            "depth": specs["depth"],
            "status": status,
            "wall_s": round(wall_s, 1),
        }
        pd.DataFrame([row]).to_csv(
            csv_path, mode="a", index=False,
            header=not csv_path.exists(),
        )

    pbar.close()

    print(f"\nDONE. Results: {csv_path}")
    print(f"  ok: {n_ok}  |  skipped/error: {n_fail}")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  Total rows in CSV: {len(df)}")
