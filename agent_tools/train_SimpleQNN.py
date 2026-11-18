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
from vqc_search.models.SimpleQNN import (
    SimpleQNN_CustomVQC,
)


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
def TrainCustomSimpleQNNTool(
    VQC_code: str,
    VQC_weights_shape: list,
    q_enc_size: int,
    q_out_size: int,
    epochs: int,
) -> dict:
    """
                Train a custom simple QNN consisting of:
                - a linear layer that maps the input to `q_enc_size` and scales it to [0, pi].
                - a VQC defined by the user in `VQC_code` that maps `q_enc_size` inputs to `q_out_size` outputs.
                - a linear layer that maps `q_out_size` to the final output size (1).
                - a sigmoid activation at the end (y is in [0, 1]).
                The data consists of gaussians plus noise (different sigmas and amplitudes).
                The goal is to predict the peak positions of the gaussians (scaled to [0, 1]).
                Everything is already set up and a QNN is trained.


                The VQC can be arbitrary, but must follow these rules:
                - VQC_code should define a PennyLane QNode function named 'circuit' that takes 'inputs' and 'weights' as arguments.
                - No imports are allowed in the VQC code.
                - can use any number of qubits, but should be <10, otherwise training will be very slow.
                - the input size is `q_enc_size`. If course, you don't have to use all inputs, but then you could also make `q_enc_size` smaller.
                - can use any encoding and ansatz you like. Usually one chooses `n_qubits = q_enc_size`, but that does not have to be the case if you find a different encoding.
                - the output size is `q_out_size` and should be equal to the number of measurements.

                Parameters:
                VQC_code (str): The code defining the VQC. Pennylane syntax.
                VQC_weights_shape (list): The shape of the weights for the VQC.
                q_enc_size (int): The number of inputs to the VQC.
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
    n_qubits = 9
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Layer 1: RX, RY, RZ rotations
        for i in range(n_qubits):
            qml.RX(inputs[i % len(inputs)], wires=i)
            qml.RY(weights[i, 0], wires=i)
            qml.RZ(weights[i, 1], wires=i)
        # Entangling layer 1: CNOT ring
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        # Final measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
                ---
                In this example:
                tool.VQC_weights_shape = [9, 2]
                tool.q_enc_size = 9
                tool.q_out_size = 9

                and for example:
                epochs=10
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    hparams = {
        "VQC_code": VQC_code,
        "VQC_weights_shape": VQC_weights_shape,
        "input_size": 21,
        "q_enc_size": q_enc_size,
        "dropout": 0.1,
        "q_out_size": q_out_size,
    }
    model = SimpleQNN_CustomVQC(
        hparams,
    )
    inputs = np.zeros(q_enc_size)
    weights = np.zeros(VQC_weights_shape)

    fig, ax = qml.draw_mpl(model.vqc.qnode)(inputs, weights)
    plt.title("Variational Quantum Circuit (VQC) Structure")
    dir_fig = (
        NEW_RUNS_DIR / "circuits_SimpleQNN"
    )
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig_path = f"VQC_circuit_plot_{timestamp}.png"
    fig_path = dir_fig / fig_path
    fig.savefig(fig_path)

    # number of gates in the circuit
    x_tmp = pnp.random.rand(q_enc_size)
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

    path = NEW_RUNS_DIR / "SimpleQNN_runs_log"
    path_csv = (
        NEW_RUNS_DIR / "SimpleQNN_runs_log.csv"
    )

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


def train_good_example():
    code = """
n_qubits = 9
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev)
def circuit(inputs, weights):
    # Layer 1: RX, RY, RZ rotations
    for i in range(n_qubits):
        qml.RX(inputs[i % len(inputs)], wires=i)
        qml.RY(weights[i, 0], wires=i)
        qml.RZ(weights[i, 1], wires=i)
    # Entangling layer 1: CNOT ring
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
    # Layer 2: RX, RY, RZ rotations
    for i in range(n_qubits):
        qml.RX(weights[i, 2], wires=i)
        qml.RY(weights[i, 3], wires=i)
        qml.RZ(weights[i, 4], wires=i)
    # Entangling layer 2: CNOT every other qubit
    for i in range(0, n_qubits, 2):
        qml.CNOT(wires=[i, (i + 2) % n_qubits])
    # Final measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    """
    # without Orchestral the tool is a plain function
    result = TrainCustomSimpleQNNTool(
        VQC_code=code,
        VQC_weights_shape=[9, 5],
        q_enc_size=9,
        q_out_size=9,
        epochs=10,
    )
    print(result)


if __name__ == "__main__":
    train_good_example()
#     code = """
# n_qubits = 9
# dev = qml.device("default.qubit", wires=n_qubits, shots=None)
#
# @qml.qnode(dev)
# def circuit(inputs, weights):
#     # Layer 1: RX, RY, RZ rotations
#     for i in range(n_qubits):
#         qml.RX(inputs[i % len(inputs)], wires=i)
#         qml.RY(weights[i, 0], wires=i)
#         qml.RZ(weights[i, 1], wires=i)
#     # Entangling layer 1: CNOT ring
#     for i in range(n_qubits):
#         qml.CNOT(wires=[i, (i + 1) % n_qubits])
#     # Final measurement
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
#     """
#     tool = TrainCustomSimpleQNNTool
#     tool.VQC_code = code
#     tool.VQC_weights_shape = [9, 2]
#     tool.q_enc_size = 9
#     tool.q_out_size = 9
#     tool.epochs = 10
#     result = tool._run()
#     print(result)
