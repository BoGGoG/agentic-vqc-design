"""
Evolutionary VQC search baseline for SimpleQNN Gaussian-peak regression.

A simple (mu+lambda) evolutionary strategy over the same circuit search space
as the random search baseline. Represents the "evolutionary QAS" family cited
by Reviewer 2.

Algorithm:
  1. Initialize population of POP_SIZE random circuits.
  2. Train each, record fitness = -test_rmse.
  3. For N_GENERATIONS generations:
       a. Tournament selection (k=3) to pick a parent.
       b. Mutate parent -> offspring.
       c. Train offspring, record fitness.
       d. Replace worst individual if offspring is better.
  4. Total evaluations = POP_SIZE + N_GENERATIONS = 26 (matches agent budget).

Mutation operators (one chosen uniformly per mutation):
  - change_n_qubits: bump qubit count by +-1 or +-2 (clamped to QUBIT_RANGE)
  - change_n_rots: change rotation gate count (1/2/3) for all layers
  - change_entangling: swap one layer's entangling pattern
  - add_layer: append a new layer (if n_layers < 5)
  - remove_layer: remove last layer (if n_layers > 1)

Training conditions are IDENTICAL to the agent path and the random search
baseline (same JAX training loop with matched hyperparameters).

Run from repo root:
    python -m experiments.evolutionary_search_baseline
"""

import sys
import time
import warnings
from pathlib import Path
from copy import deepcopy

import jax
import jax.core

if not hasattr(jax.core, "is_concrete"):
    jax.core.is_concrete = lambda x: not isinstance(x, jax.core.Tracer)

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message=".*complex128.*")

import numpy as np
import pandas as pd
from tqdm import tqdm

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqc_search.paths import AGENT_RUNS_DIR, DATA_DIR, RERUN_DIR  # noqa: E402
from experiments.random_search_baseline import (
    ENTANGLING_OPTIONS,
    QUBIT_RANGE,
    ROTATION_OPTIONS,
    RESEED_SEEDS,
    generate_random_circuit,
    get_circuit_specs,
    train_circuit_jax,
)

# ---------------------------------------------------------------------------
# Evolutionary search parameters
# ---------------------------------------------------------------------------
POP_SIZE = 8
N_GENERATIONS = 18  # total evals = POP_SIZE + N_GENERATIONS = 26
TOURNAMENT_K = 3


# ---------------------------------------------------------------------------
# Genome representation
# ---------------------------------------------------------------------------
# A genome is a dict with:
#   n_layers: int (1..5)
#   n_rots: int (1..3)
#   entangling: list[str] of length n_layers

def random_genome(rng: np.random.Generator) -> dict:
    n_qubits = int(rng.integers(QUBIT_RANGE[0], QUBIT_RANGE[1] + 1))
    n_layers = int(rng.integers(1, 6))
    n_rots = int(rng.choice([1, 2, 3]))
    entangling = [rng.choice(ENTANGLING_OPTIONS) for _ in range(n_layers)]
    return {"n_qubits": n_qubits, "n_layers": n_layers, "n_rots": n_rots, "entangling": list(entangling)}


def genome_to_code(genome: dict) -> tuple:
    """Convert genome dict to (code_str, meta) using the same code builder
    as random search, but from explicit genome fields instead of rng."""
    from experiments.random_search_baseline import (
        _entangling_block,
    )

    n_qubits = genome["n_qubits"]
    n_layers = genome["n_layers"]
    n_rots = genome["n_rots"]
    gate_names = ROTATION_OPTIONS[n_rots]
    entangling_per_layer = genome["entangling"]

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
# Mutation operators
# ---------------------------------------------------------------------------

def mutate(genome: dict, rng: np.random.Generator) -> dict:
    child = deepcopy(genome)
    ops = ["change_n_qubits", "change_n_rots", "change_entangling", "add_layer", "remove_layer"]
    op = rng.choice(ops)

    if op == "change_n_qubits":
        delta = int(rng.choice([-2, -1, 1, 2]))
        new_q = child["n_qubits"] + delta
        child["n_qubits"] = max(QUBIT_RANGE[0], min(QUBIT_RANGE[1], new_q))
        if child["n_qubits"] == genome["n_qubits"]:
            return mutate(child, rng)

    elif op == "change_n_rots":
        options = [r for r in [1, 2, 3] if r != child["n_rots"]]
        child["n_rots"] = int(rng.choice(options))

    elif op == "change_entangling":
        idx = int(rng.integers(0, child["n_layers"]))
        options = [e for e in ENTANGLING_OPTIONS if e != child["entangling"][idx]]
        child["entangling"][idx] = rng.choice(options)

    elif op == "add_layer":
        if child["n_layers"] < 5:
            child["n_layers"] += 1
            child["entangling"].append(rng.choice(ENTANGLING_OPTIONS))
        else:
            return mutate(child, rng)

    elif op == "remove_layer":
        if child["n_layers"] > 1:
            child["n_layers"] -= 1
            child["entangling"].pop()
        else:
            return mutate(child, rng)

    return child


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_select(population: list, fitnesses: list, rng: np.random.Generator, k: int = TOURNAMENT_K) -> int:
    candidates = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(candidates, key=lambda i: fitnesses[i])
    return best


# ---------------------------------------------------------------------------
# Evaluate a genome
# ---------------------------------------------------------------------------

def evaluate_genome(genome: dict, seed: int = 0) -> dict:
    code, meta = genome_to_code(genome)
    specs = get_circuit_specs(code, meta["weights_shape"], meta["n_qubits"])
    result = train_circuit_jax(code, meta["weights_shape"], seed=seed,
                               n_qubits=meta["n_qubits"])
    return {
        "code": code,
        "meta": meta,
        "specs": specs,
        "test_rmse": result["test_rmse"],
        "trainable_vqc_params": result["trainable_vqc_params"],
        "trainable_total_params": result["trainable_total_params"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(seed=123)
    all_rows = []
    eval_count = 0

    print(f"Evolutionary VQC search: pop={POP_SIZE}, generations={N_GENERATIONS}")
    print(f"  Total evaluations: {POP_SIZE + N_GENERATIONS}")
    print(f"  Tournament k={TOURNAMENT_K}\n")

    # --- Phase 1: Initialize population ---
    print("Phase 1: Initializing population")
    population = []
    fitnesses = []
    eval_results = []

    for i in tqdm(range(POP_SIZE), desc="Init population", unit="ind"):
        genome = random_genome(rng)
        label = f"q={genome['n_qubits']} L={genome['n_layers']} rots={genome['n_rots']} ent={','.join(genome['entangling'])}"

        t0 = time.time()
        try:
            result = evaluate_genome(genome, seed=0)
        except Exception as e:
            tqdm.write(f"  FAILED: {e}")
            genome = random_genome(rng)
            result = evaluate_genome(genome, seed=0)

        dt = time.time() - t0
        eval_count += 1
        fitness = -result["test_rmse"]

        population.append(genome)
        fitnesses.append(fitness)
        eval_results.append(result)

        tqdm.write(f"  [init {i+1}/{POP_SIZE}] {label} -> rmse={result['test_rmse']:.6f} ({dt:.1f}s)")

        all_rows.append({
            "method": "evolutionary",
            "generation": 0,
            "eval_id": eval_count,
            "n_qubits": genome["n_qubits"],
            "n_layers": genome["n_layers"],
            "n_rots_per_qubit": genome["n_rots"],
            "gate_names": result["meta"]["gate_names"],
            "entangling": result["meta"]["entangling"],
            "trainable_vqc_params": result["trainable_vqc_params"],
            "trainable_total_params": result["trainable_total_params"],
            "seed": 0,
            "test_rmse": result["test_rmse"],
            "num_operations": result["specs"]["num_operations"],
            "depth": result["specs"]["depth"],
        })

    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    print(f"\n  Initial best: test_rmse={-fitnesses[best_idx]:.6f} "
          f"(individual {best_idx})")

    # --- Phase 2: Evolution ---
    print(f"\nPhase 2: Evolution ({N_GENERATIONS} generations)")

    for gen in tqdm(range(1, N_GENERATIONS + 1), desc="Evolution", unit="gen"):
        parent_idx = tournament_select(population, fitnesses, rng)
        parent = population[parent_idx]

        child = mutate(parent, rng)
        label = f"q={child['n_qubits']} L={child['n_layers']} rots={child['n_rots']} ent={','.join(child['entangling'])}"

        t0 = time.time()
        try:
            result = evaluate_genome(child, seed=0)
        except Exception as e:
            tqdm.write(f"  [gen {gen}] FAILED: {e}, skipping")
            continue

        dt = time.time() - t0
        eval_count += 1
        child_fitness = -result["test_rmse"]

        worst_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        replaced = ""
        if child_fitness > fitnesses[worst_idx]:
            replaced = f" [replaced #{worst_idx}]"
            population[worst_idx] = child
            fitnesses[worst_idx] = child_fitness
            eval_results[worst_idx] = result

        current_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        tqdm.write(f"  [gen {gen}/{N_GENERATIONS}] {label} -> rmse={result['test_rmse']:.6f} "
                   f"best={-fitnesses[current_best_idx]:.6f} ({dt:.1f}s){replaced}")

        all_rows.append({
            "method": "evolutionary",
            "generation": gen,
            "eval_id": eval_count,
            "n_qubits": child["n_qubits"],
            "n_layers": child["n_layers"],
            "n_rots_per_qubit": child["n_rots"],
            "gate_names": result["meta"]["gate_names"],
            "entangling": result["meta"]["entangling"],
            "trainable_vqc_params": result["trainable_vqc_params"],
            "trainable_total_params": result["trainable_total_params"],
            "seed": 0,
            "test_rmse": result["test_rmse"],
            "num_operations": result["specs"]["num_operations"],
            "depth": result["specs"]["depth"],
        })

    # --- Summary ---
    final_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    final_best_rmse = -fitnesses[final_best_idx]
    all_rmses = [-f for f in fitnesses]

    print(f"\n{'='*60}")
    print(f"  EVOLUTIONARY SEARCH SUMMARY ({eval_count} evaluations)")
    print(f"{'='*60}")
    print(f"  Best  test_rmse: {min(all_rmses):.6f}")
    print(f"  Population mean: {np.mean(all_rmses):.6f} +/- {np.std(all_rmses):.6f}")

    # --- Retrain best over 5 seeds ---
    best_genome = population[final_best_idx]
    print(f"\n  Retraining best genome over {len(RESEED_SEEDS)} seeds...")
    print(f"  q={best_genome['n_qubits']} L={best_genome['n_layers']} rots={best_genome['n_rots']} "
          f"ent={','.join(best_genome['entangling'])}")

    best_code, best_meta = genome_to_code(best_genome)
    best_specs = get_circuit_specs(best_code, best_meta["weights_shape"], best_meta["n_qubits"])

    reseed_rows = []
    for s in tqdm(RESEED_SEEDS, desc="Reseed best", unit="seed"):
        t0 = time.time()
        r = train_circuit_jax(best_code, best_meta["weights_shape"], seed=s,
                              n_qubits=best_meta["n_qubits"])
        dt = time.time() - t0
        tqdm.write(f"    seed={s}: test_rmse={r['test_rmse']:.6f}  ({dt:.1f}s)")
        reseed_rows.append({
            "method": "evolutionary_best_reseed",
            "generation": -1,
            "eval_id": -1,
            "n_qubits": best_genome["n_qubits"],
            "n_layers": best_genome["n_layers"],
            "n_rots_per_qubit": best_genome["n_rots"],
            "gate_names": best_meta["gate_names"],
            "entangling": best_meta["entangling"],
            "trainable_vqc_params": r["trainable_vqc_params"],
            "trainable_total_params": r["trainable_total_params"],
            "seed": s,
            "test_rmse": r["test_rmse"],
            "num_operations": best_specs["num_operations"],
            "depth": best_specs["depth"],
        })

    df_reseed = pd.DataFrame(reseed_rows)
    print(f"\n  Best genome 5-seed: mean={df_reseed['test_rmse'].mean():.6f} "
          f"+/- {df_reseed['test_rmse'].std():.6f}")

    # --- Save ---
    df_all = pd.concat([pd.DataFrame(all_rows), df_reseed], ignore_index=True)
    csv_path = RERUN_DIR / "evolutionary_search_baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")
