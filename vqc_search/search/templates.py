"""
Template-based VQC search space.

Defines parametric circuit templates that compile to PennyLane code strings,
covering the same design space the LLM agent explores.
"""

from dataclasses import dataclass, field
from typing import Literal


ROTATION_GATES = ("RX", "RY", "RZ")
ENTANGLING_PATTERNS = ("ring", "linear", "none")
ENTANGLING_GATES = ("CNOT", "CZ")
ENCODING_GATES = ("RY", "RX")
MEASUREMENT_BASES = ("Z", "XYZ")


@dataclass(frozen=True)
class CircuitTemplate:
    """Template for SimpleQNN and QuanvNN VQC circuits."""

    n_qubits: int = 5
    n_layers: int = 2
    rotation_gates: tuple[str, ...] = ("RY", "RZ")
    entangling_pattern: Literal["ring", "linear", "none"] = "ring"
    entangling_gate: Literal["CNOT", "CZ"] = "CNOT"
    data_reuploading: bool = False
    encoding_gate: Literal["RY", "RX"] = "RY"
    measurement_basis: Literal["Z", "XYZ"] = "Z"

    def __post_init__(self):
        assert 2 <= self.n_qubits <= 9
        assert 1 <= self.n_layers <= 5
        assert all(g in ROTATION_GATES for g in self.rotation_gates)
        assert len(self.rotation_gates) >= 1

    @property
    def n_params_per_qubit_per_layer(self) -> int:
        return len(self.rotation_gates)

    def weights_shape(self) -> list[int]:
        return [self.n_layers, self.n_qubits, self.n_params_per_qubit_per_layer]

    @property
    def output_size(self) -> int:
        if self.measurement_basis == "Z":
            return self.n_qubits
        return 3 * self.n_qubits  # XYZ


@dataclass(frozen=True)
class FullQuantumTemplate:
    """Template for FullQuantum (recurrent/data-reuploading) VQC circuits."""

    n_qubits: int = 4
    n_cell_layers: int = 1
    cell_rotation_gates: tuple[str, ...] = ("RY", "RZ")
    cell_entangling: Literal["ring", "linear"] = "ring"
    init_hadamard: bool = True
    input_encoding_gate: Literal["RY", "RX"] = "RY"
    seq_len: int = 21

    def __post_init__(self):
        assert 3 <= self.n_qubits <= 9
        assert 1 <= self.n_cell_layers <= 3
        assert all(g in ROTATION_GATES for g in self.cell_rotation_gates)
        assert len(self.cell_rotation_gates) >= 1

    def weights_shape(self) -> list[int]:
        return [self.n_cell_layers, self.n_qubits, len(self.cell_rotation_gates)]

    @property
    def output_size(self) -> int:
        return self.n_qubits - 1  # hidden wires only


@dataclass(frozen=True)
class QuanvNNConfig:
    """QuanvNN = CircuitTemplate kernel + conv parameters."""

    circuit: CircuitTemplate = field(default_factory=CircuitTemplate)
    kernel_size: int = 5
    stride: int = 2

    def __post_init__(self):
        assert self.kernel_size in (3, 5, 7)
        assert self.stride in (1, 2)
        assert self.circuit.n_qubits == self.kernel_size


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _indent(code: str, level: int = 1) -> str:
    prefix = "    " * level
    return "\n".join(prefix + line if line.strip() else "" for line in code.splitlines())


def _gen_encoding(n_qubits: int, gate: str, indent: int = 1) -> str:
    prefix = "    " * indent
    return f"{prefix}for i in range(n_qubits):\n{prefix}    qml.{gate}(inputs[i], wires=i)"


def _gen_rotation_block(rotation_gates: tuple[str, ...], layer_idx: int, indent: int = 1) -> str:
    prefix = "    " * indent
    lines = [f"{prefix}for i in range(n_qubits):"]
    for gi, gate in enumerate(rotation_gates):
        lines.append(f"{prefix}    qml.{gate}(weights[{layer_idx}, i, {gi}], wires=i)")
    return "\n".join(lines)


def _gen_entangling(pattern: str, gate: str, indent: int = 1) -> str:
    prefix = "    " * indent
    if pattern == "none":
        return ""
    if pattern == "ring":
        return (
            f"{prefix}for i in range(n_qubits):\n"
            f"{prefix}    qml.{gate}(wires=[i, (i + 1) % n_qubits])"
        )
    if pattern == "linear":
        return (
            f"{prefix}for i in range(n_qubits - 1):\n"
            f"{prefix}    qml.{gate}(wires=[i, i + 1])"
        )
    raise ValueError(f"Unknown entangling pattern: {pattern}")


def _gen_measurement(n_qubits_expr: str, basis: str, indent: int = 1) -> str:
    prefix = "    " * indent
    if basis == "Z":
        return f"{prefix}return [qml.expval(qml.PauliZ(i)) for i in range({n_qubits_expr})]"
    # XYZ: 3 measurements per qubit
    lines = [
        f"{prefix}obs = []",
        f"{prefix}for i in range({n_qubits_expr}):",
        f"{prefix}    obs.append(qml.expval(qml.PauliX(i)))",
        f"{prefix}    obs.append(qml.expval(qml.PauliY(i)))",
        f"{prefix}    obs.append(qml.expval(qml.PauliZ(i)))",
        f"{prefix}return obs",
    ]
    return "\n".join(lines)


def generate_vqc_code(template: CircuitTemplate) -> tuple[str, list[int]]:
    """Generate a PennyLane VQC code string from a CircuitTemplate.

    Returns (code_str, weights_shape).
    """
    t = template
    lines = [
        f"n_qubits = {t.n_qubits}",
        f'dev = qml.device("default.qubit", wires=n_qubits, shots=None)',
        "",
        "@qml.qnode(dev)",
        "def circuit(inputs, weights):",
    ]

    # Encoding
    lines.append(_gen_encoding(t.n_qubits, t.encoding_gate))

    # Variational layers
    for layer in range(t.n_layers):
        lines.append("")
        lines.append(_gen_rotation_block(t.rotation_gates, layer))
        ent = _gen_entangling(t.entangling_pattern, t.entangling_gate)
        if ent:
            lines.append(ent)

        # Data reuploading: re-encode inputs after each layer (except last)
        if t.data_reuploading and layer < t.n_layers - 1:
            lines.append("")
            lines.append(_gen_encoding(t.n_qubits, t.encoding_gate))

    # Measurement
    lines.append("")
    lines.append(_gen_measurement("n_qubits", t.measurement_basis))

    code = "\n".join(lines)
    return code, t.weights_shape()


def generate_fullquantum_code(template: FullQuantumTemplate) -> tuple[str, list[int]]:
    """Generate a PennyLane VQC code string for FullQuantum (recurrent) architecture.

    Returns (code_str, weights_shape).
    """
    t = template
    lines = [
        f"n_qubits = {t.n_qubits}",
        "input_wire = 0",
        "hidden_wires = list(range(1, n_qubits))",
        f'dev = qml.device("default.qubit", wires=n_qubits, shots=None)',
        "",
        f"seq_len = {t.seq_len}",
        f"n_cell_layers = {t.n_cell_layers}",
        "def recurrent_cell(weights):",
        "    for l in range(n_cell_layers):",
    ]

    # Cell rotation gates
    for gi, gate in enumerate(t.cell_rotation_gates):
        lines.append(f"        for w in range(n_qubits):")
        lines.append(f"            qml.{gate}(weights[l, w, {gi}], wires=w)")

    # Cell entangling
    if t.cell_entangling == "ring":
        lines.append("        for w in range(n_qubits - 1):")
        lines.append("            qml.CNOT(wires=[w, w + 1])")
        lines.append("        qml.CNOT(wires=[n_qubits - 1, 0])")
    elif t.cell_entangling == "linear":
        lines.append("        for w in range(n_qubits - 1):")
        lines.append("            qml.CNOT(wires=[w, w + 1])")

    lines.append("")
    lines.append("@qml.qnode(dev)")
    lines.append("def circuit(inputs, weights):")

    # Init hidden qubits
    if t.init_hadamard:
        lines.append("    for w in hidden_wires:")
        lines.append("        qml.Hadamard(wires=w)")
        lines.append("")

    # Recurrent loop
    lines.append("    for t_step in range(seq_len):")
    lines.append(f"        qml.{t.input_encoding_gate}(inputs[t_step], wires=input_wire)")
    lines.append("        recurrent_cell(weights)")
    lines.append("")

    # Measurement on hidden wires
    lines.append("    return [qml.expval(qml.PauliZ(w)) for w in hidden_wires]")

    code = "\n".join(lines)
    return code, t.weights_shape()


def generate_quanvnn_code(config: QuanvNNConfig) -> tuple[str, list[int], dict]:
    """Generate VQC code + conv config for QuanvNN.

    Returns (code_str, weights_shape, conv_params) where conv_params has
    kernel_size, stride, VQC_output_dim.
    """
    # For QuanvNN, the qnode needs interface="torch"
    t = config.circuit
    code, ws = generate_vqc_code(t)
    # Patch the qnode decorator to add interface="torch"
    code = code.replace("@qml.qnode(dev)", '@qml.qnode(dev, interface="torch")')

    conv_params = {
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "VQC_output_dim": t.output_size,
    }
    return code, ws, conv_params
