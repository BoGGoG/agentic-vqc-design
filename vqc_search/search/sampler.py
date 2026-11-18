"""
Random sampling of circuit templates from the search space.
"""

import random
from typing import Literal

from .templates import (
    ROTATION_GATES,
    CircuitTemplate,
    FullQuantumTemplate,
    QuanvNNConfig,
)


def _sample_rotation_gates(rng: random.Random, pool: tuple[str, ...] = ROTATION_GATES) -> tuple[str, ...]:
    k = rng.randint(1, len(pool))
    return tuple(rng.sample(pool, k))


def sample_circuit_template(rng: random.Random) -> CircuitTemplate:
    """Sample a random CircuitTemplate for SimpleQNN."""
    n_qubits = rng.randint(4, 9)
    return CircuitTemplate(
        n_qubits=n_qubits,
        n_layers=rng.randint(1, 4),
        rotation_gates=_sample_rotation_gates(rng),
        entangling_pattern=rng.choice(["ring", "linear", "none"]),
        entangling_gate=rng.choice(["CNOT", "CZ"]),
        data_reuploading=rng.choice([True, False]),
        encoding_gate=rng.choice(["RY", "RX"]),
        measurement_basis=rng.choice(["Z", "XYZ"]),
    )


def sample_fullquantum_template(rng: random.Random) -> FullQuantumTemplate:
    """Sample a random FullQuantumTemplate."""
    return FullQuantumTemplate(
        n_qubits=rng.randint(3, 7),
        n_cell_layers=rng.randint(1, 3),
        cell_rotation_gates=_sample_rotation_gates(rng, ("RY", "RZ", "RX")),
        cell_entangling=rng.choice(["ring", "linear"]),
        init_hadamard=rng.choice([True, False]),
        input_encoding_gate=rng.choice(["RY", "RX"]),
    )


def sample_quanvnn_config(rng: random.Random) -> QuanvNNConfig:
    """Sample a random QuanvNNConfig."""
    kernel_size = rng.choice([3, 5, 7])
    circuit = CircuitTemplate(
        n_qubits=kernel_size,
        n_layers=rng.randint(1, 3),
        rotation_gates=_sample_rotation_gates(rng),
        entangling_pattern=rng.choice(["ring", "linear"]),
        entangling_gate=rng.choice(["CNOT", "CZ"]),
        data_reuploading=rng.choice([True, False]),
        encoding_gate=rng.choice(["RY", "RX"]),
        measurement_basis=rng.choice(["Z", "XYZ"]),
    )
    return QuanvNNConfig(
        circuit=circuit,
        kernel_size=kernel_size,
        stride=rng.choice([1, 2]),
    )


def sample_template(
    architecture: Literal["SimpleQNN", "FullQuantum", "QuanvNN"],
    rng: random.Random,
):
    """Sample a random template for the given architecture."""
    if architecture == "SimpleQNN":
        return sample_circuit_template(rng)
    if architecture == "FullQuantum":
        return sample_fullquantum_template(rng)
    if architecture == "QuanvNN":
        return sample_quanvnn_config(rng)
    raise ValueError(f"Unknown architecture: {architecture}")
