import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.numpy as pnp
import pyarrow as pa
import pyarrow.parquet as pq

from agent_tools._orchestral_compat import define_tool
from vqc_search.paths import NEW_RUNS_DIR
from vqc_search.models.Quanvolutional import (
    train_quantum_model,
)
from vqc_search.models.FullQuantum import FullQuantumQNN


def get_git_commit():
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"
    return commit


def append_parquet(dir, df):
    dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{timestamp}.parquet"

    # write to parquet file using pq.write_table
    # but if exactly this file exists, change the name to _1, _2, ...
    if (dir / fname).exists():
        i = 1
        while (dir / f"{timestamp}_{i}.parquet").exists():
            i += 1
        fname = f"{timestamp}_{i}.parquet"
    pq.write_table(table, dir / fname)


@define_tool
def TrainCustomFullQuantumQNN(
    VQC_code: str,
    VQC_weights_shape: list,
    q_out_size: int,
    epochs: int,
) -> dict:
    """
                        Train a custom full quantum QNN consisting of:
                        - input of length 21
                        - a VQC defined by the user in `VQC_code` that takes 21 inputs and does something to them
                        - a linear layer that maps `q_out_size` to the final output size (1).
                        - a sigmoid activation at the end (y is in [0, 1]).
                        The data consists of gaussians plus noise (different sigmas and amplitudes).
                        The goal is to predict the peak positions of the gaussians (scaled to [0, 1]).
                        Everything is already set up and a QNN is trained.

                        The VQC can be arbitrary, but must follow these rules:
                        - VQC_code should define a PennyLane QNode function named 'circuit' that takes 'inputs' and 'weights' as arguments.
                        - No imports are allowed in the VQC code.
                        - can use any number of qubits, but should be <10, otherwise training will be very slow.
                        - the input size is 21. You don't have to use all of the inputs, but it is recommended. Since the number of inputs is larger than the number of qubits, you have to come up with some strategy of bringing them into the VQC
                        - can use any encoding and ansatz you like
                        - the output size is `q_out_size` and should be equal to the number of measurements.

                        Parameters:
                        VQC_code (str): The code defining the VQC. Pennylane syntax.
                        VQC_weights_shape (list): The shape of the weights for the VQC.
                        q_out_size (int): The number of outputs from the VQC.
                        epochs (int): Number of training epochs.

                        Output:
                        dict: A dictionary containing:
                        - test_RMSE (float): The test RMSE after training.
                        - val_RMSE_history (list): The validation RMSE history over epochs.
                        - train_RMSE_last_batch (float): The training RMSE of the last batch
                        - n_gates_in_VQC (int): Number of gates in the VQC.
                        - n_trainable_params_total (int): Total number of trainable parameters in the model
                        - n_trainable_params_VQC (int): Number of trainable parameters in the VQC.


                        Example code (watch out for indents, since it is interpreted as python code, remember that the input is already scaled to [0, pi]):
                        ---
    n_qubits = 4
    input_wire = 0
    hidden_wires = list(range(1, n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    seq_len = 21
    n_cell_layers = 2
    def recurrent_cell(weights):
        for l in range(n_cell_layers):
            # Local rotations
            for w in range(n_qubits):
                theta_y = weights[l, w, 0]
                theta_z = weights[l, w, 1]
                qml.RY(theta_y, wires=w)
                qml.RZ(theta_z, wires=w)
            # Entangling pattern: ring of CNOTs
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Optionally initialize hidden qubits in |+> to give them some "prior" state
        for w in hidden_wires:
            qml.Hadamard(wires=w)

        # Quantum RNN / data reuploading over time steps
        for t in range(seq_len):
            x_t = inputs[t]

            # Encode current input on the input qubit
            qml.RY(x_t, wires=input_wire)

            # Apply the recurrent cell with shared parameters
            recurrent_cell(weights)

            # Optional: approximately remove input imprint on q_in
            # (if you want a cleaner "input only" semantics)
            # qml.RY(-x_t, wires=input_wire)

        # Read out hidden qubits as features
        return [qml.expval(qml.PauliZ(w)) for w in hidden_wires]

                        ---
                        In this example:
                        "q_out_size": 3, # number of measurements
                        "VQC_weights_shape": [2, 9, 2],  # not optimal in this example, since some weights are unused. You might want to fix this

                        and for example:
                        epochs=10

                        This example does work in the sense that it runs and it learns something, but it is not good at all.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    hparams = {
        "VQC_code": VQC_code,
        "VQC_weights_shape": VQC_weights_shape,
        "input_size": (input_size := 21),
        "dropout": 0.1,
        "q_out_size": q_out_size,
    }
    model = FullQuantumQNN(
        hparams,
    )
    inputs = np.zeros(input_size)
    weights = np.zeros(VQC_weights_shape)

    # here the circuits can get too big to plot, so we skip this for now
    # fig, ax = qml.draw_mpl(model.vqc.qnode)(inputs, weights)
    # plt.title("Variational Quantum Circuit (VQC) Structure")
    # dir_fig = (
    #     NEW_RUNS_DIR / "circuits_FullQuantum"
    # )
    # dir_fig.mkdir(parents=True, exist_ok=True)
    # fig_path = f"VQC_circuit_plot_{timestamp}.png"
    # fig_path = dir_fig / fig_path
    # fig.savefig(fig_path)
    # plt.close(fig)

    # number of gates in the circuit
    x_tmp = pnp.random.rand(input_size)
    weights_tmp = pnp.random.rand(*VQC_weights_shape)
    specs = qml.specs(model.vqc.qnode)
    specs = specs(x_tmp, weights_tmp)
    resources = specs["resources"]
    n_gates = resources.num_gates
    depth = resources.depth
    # gate_types = resources.gate_types
    # gate_sizes = resources.gate_sizes

    # training
    start_time = time.time()
    train_results_dict = train_quantum_model(
        model, epochs, show_progress=True, data_save_dir="gaussian_peak_noise_0_01"
    )
    end_time = time.time()
    train_results_dict["n_trainable_params_total"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    train_results_dict["n_trainable_params_VQC"] = sum(
        p.numel() for p in model.vqc.parameters() if p.requires_grad
    )

    path = NEW_RUNS_DIR / "FullQuantum_runs_log"
    path_csv = NEW_RUNS_DIR / "FullQuantum_runs_log.csv"

    row = pd.DataFrame(
        [
            {
                # "run_id": run_id,
                "descriptiontimestamp": timestamp,
                "epochs": epochs,
                "VQC_code": VQC_code,
                "VQC_weights_shape": VQC_weights_shape,
                "test_MSE": train_results_dict["test_MSE"],
                "test_RMSE": train_results_dict["test_RMSE"],
                "train_MSE_hist": train_results_dict["train_MSE_hist"],
                "val_MSE_hist": train_results_dict["val_MSE_hist"],
                "git_commit": get_git_commit(),
                "pennylane_version": qml.__version__,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "training_plus_test_time_sec": end_time - start_time,
                "n_trainable_params_total": train_results_dict[
                    "n_trainable_params_total"
                ],
                "n_trainable_params_VQC": train_results_dict["n_trainable_params_VQC"],
                "n_gates_in_VQC": n_gates,
                "depth_of_VQC": depth,
            }
        ]
    )
    append_parquet(path, row)

    csv_row = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "epochs": epochs,
                "test_RMSE": train_results_dict["test_RMSE"],
                "val_RMSE": np.sqrt(train_results_dict["val_MSE_hist"][-1]),
                "n_trainable_params_total": train_results_dict[
                    "n_trainable_params_total"
                ],
                "n_trainable_params_VQC": train_results_dict["n_trainable_params_VQC"],
                "n_gates_in_VQC": n_gates,
                "depth_of_VQC": depth,
            }
        ]
    )

    # append to csv file
    csv_row.to_csv(path_csv, mode="a", header=not path_csv.exists(), index=False)

    out_dict = {}
    out_dict["test_RMSE"] = float(train_results_dict["test_RMSE"])
    out_dict["val_RMSE_history"] = np.sqrt(train_results_dict["val_MSE_hist"]).tolist()
    out_dict["train_RMSE_last_batch"] = float(
        np.sqrt(train_results_dict["train_MSE_hist"][-1])
    )
    out_dict["n_gates_in_VQC"] = n_gates
    out_dict["n_trainable_params_total"] = train_results_dict[
        "n_trainable_params_total"
    ]
    out_dict["n_trainable_params_VQC"] = train_results_dict["n_trainable_params_VQC"]

    return out_dict


def train_example():
    code = """
n_qubits = 4
input_wire = 0
hidden_wires = list(range(1, n_qubits))
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

seq_len = 21
n_cell_layers = 2
def recurrent_cell(weights):
    for l in range(n_cell_layers):
        # Local rotations
        for w in range(n_qubits):
            theta_y = weights[l, w, 0]
            theta_z = weights[l, w, 1]
            qml.RY(theta_y, wires=w)
            qml.RZ(theta_z, wires=w)
        # Entangling pattern: ring of CNOTs
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

@qml.qnode(dev)
def circuit(inputs, weights):
    # Optionally initialize hidden qubits in |+> to give them some "prior" state
    for w in hidden_wires:
        qml.Hadamard(wires=w)

    # Quantum RNN / data reuploading over time steps
    for t in range(seq_len):
        x_t = inputs[t]

        # Encode current input on the input qubit
        qml.RY(x_t, wires=input_wire)

        # Apply the recurrent cell with shared parameters
        recurrent_cell(weights)

        # Optional: approximately remove input imprint on q_in
        # (if you want a cleaner "input only" semantics)
        # qml.RY(-x_t, wires=input_wire)

    # Read out hidden qubits as features
    return [qml.expval(qml.PauliZ(w)) for w in hidden_wires]
    """
    # without Orchestral the tool is a plain function
    result = TrainCustomFullQuantumQNN(
        VQC_code=code,
        VQC_weights_shape=[2, 9, 5],
        q_out_size=3,
        epochs=10,
    )
    print(result)


if __name__ == "__main__":
    train_example()
