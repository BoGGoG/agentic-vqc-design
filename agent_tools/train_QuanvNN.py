from vqc_search.models.Quanvolutional import (
    train_quantum_model_custom_VQC,
)
from agent_tools._orchestral_compat import define_tool
from vqc_search.paths import NEW_RUNS_DIR
import pandas as pd
import time
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
import subprocess
import pennylane as qml
import pennylane.numpy as pnp
import sys
import platform
import numpy as np
import matplotlib.pyplot as plt
from vqc_search.models.Quanvolutional import QuanvCNN1DCustomVQC
import os


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
def TrainCustomQuanvNNTool(
    VQC_code: str,
    VQC_weights_shape: list,
    kernel_size: int,
    stride: int,
    VQC_output_dim: int,
    epochs: int,
) -> dict:
    """
            Train a custom quanvolutional quantum neural network (QNN) on a fixed 1D regression task.

            High-level architecture:
            - Input: 1D real-valued vector of length 21 (Gaussian peaks plus noise, targets scaled to [0, 1]).
            - Quanvolution layer:
              - A sliding window of size `kernel_size` and stride `stride` is moved over the input.
              - Each window is fed into a variational quantum circuit (VQC) defined by `VQC_code` and `VQC_weights_shape`.
              - The VQC must return exactly `VQC_output_dim` expectation values (observables) per window.
              - These `VQC_output_dim` features per window are treated as channels of a feature map.
            - Classical head:
              - A small 1D CNN with two residual conv layers.
              - Adaptive pooling over the sequence length to obtain a fixed-size feature vector.
              - A final fully-connected network with a single sigmoid output neuron (regression target y ∈ [0, 1]).

            Task:
            - Synthetic data consisting of Gaussian peaks plus noise (varying amplitudes and sigmas).
            - The goal is to predict the peak positions of the Gaussians (scaled to [0, 1]).
            - The dataset, optimizer, and training loop are already set up; the tool runs a full training of the QNN.

            Requirements on the VQC (what you must satisfy in VQC_code):
            - You must define a PennyLane QNode named `circuit` with signature:
                `def circuit(inputs, weights):`
              where:
                * `inputs` is a 1D tensor of length `kernel_size` (one sliding window),
                * `weights` is a trainable tensor whose shape is exactly `VQC_weights_shape`.
            - The circuit must return a collection (list or tuple) of exactly `VQC_output_dim`
              expectation values, e.g. `[qml.expval(...), ...]` of length `VQC_output_dim`.
            - No import statements are allowed inside `VQC_code`. The symbol `qml` (PennyLane)
              is already available in the execution environment.
            - You may use any number of qubits, but for efficiency it is recommended to use fewer
              than 10 qubits and a shallow depth, otherwise training will be very slow.
            - The input of length 21 is chopped into windows of length `kernel_size`. Usually
              `kernel_size == n_qubits`, but other choices are possible as long as your `circuit`
              can handle an `inputs` vector of length `kernel_size`.
            - You are free to choose any data encoding and ansatz, as long as the above interface
              and output-size constraints are satisfied.

            Parameters:
            - VQC_code (str): The Python code (as a string) defining the VQC. Must contain a
              PennyLane QNode named `circuit(inputs, weights)` and obey the shape constraints above.
            - VQC_weights_shape (list): Shape of the trainable weights for the VQC. This must be
              consistent with how you index `weights` in `VQC_code`. For example, if you use
              `weights[l, i, 0]` in the circuit, a natural choice is `[n_layers, n_qubits, 3]`.
            - kernel_size (int): Size of the sliding window (length of `inputs` seen by the VQC).
            - stride (int): Stride of the sliding window over the length-21 input.
            - VQC_output_dim (int): Number of expectation values (features) returned by the VQC per window.
            - epochs (int): Number of training epochs.

            Returns:
            - dict with the following entries:
              - test_RMSE (float): Test RMSE after training.
              - val_RMSE_history (list[float]): Validation RMSE per epoch.
              - train_RMSE_last_batch (float): Training RMSE of the last batch.
              - n_gates_in_VQC (int): Number of gates in the VQC.
              - n_trainable_params_total (int): Total number of trainable parameters in the model.
              - n_trainable_params_VQC (int): Number of trainable parameters in the VQC.

            Example code (watch out for indents, since it is interpreted as python code):
                    ---
    n_qubits = 5
    input_length = 5
    q_out = 10
    n_layers = 1
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        for i in range(min(input_length, n_qubits)):
            qml.RY(inputs[i], wires=i)

        for l in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        obs = [qml.PauliZ(i) for i in range(n_qubits)]
        return [qml.expval(o) for o in obs * ((q_out + n_qubits - 1) // n_qubits)][:q_out]
                ---
                In this example:
                VQC_weights_shape=[1, 5, 3]  # n_layers, n_qubits, 3 params per qubit
                kernel_size=5
                stride=2
                VQC_output_dim=10

                and for example:
                epochs=10
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # also plot the VQC. For this we have to recreate the circuit
    hparams = {
        "kernel_size": kernel_size,  # 1D kernel size
        "input_length": (seq_len := 21),  # input sequence length
        "stride": stride,  # stride for the convolution
        "VQC_output_dim": VQC_output_dim,
        "output_dim": 1,  # output dimension of the final FC NN
        "hidden_layers_dims": [10, 5],
        "dropout": 0.1,  # dropout bate
        "VQC_code": VQC_code,
        "VQC_weights_shape": VQC_weights_shape,
    }
    model = QuanvCNN1DCustomVQC(
        hparams,
    )
    inputs = np.zeros(kernel_size)
    weights = np.zeros(VQC_weights_shape)

    fig, ax = qml.draw_mpl(model.quanvolution1.qpatch.qnode)(inputs, weights)
    plt.title("Variational Quantum Circuit (VQC) Structure")
    dir_fig = NEW_RUNS_DIR / "circuits"
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig_path = f"VQC_circuit_plot_{timestamp}.png"
    fig_path = dir_fig / fig_path
    fig.savefig(fig_path)

    # number of gates in the circuit
    x_tmp = pnp.random.rand(kernel_size)
    weights_tmp = pnp.random.rand(*VQC_weights_shape)
    specs = qml.specs(model.quanvolution1.qpatch.qnode)
    specs = specs(x_tmp, weights_tmp)
    resources = specs["resources"]
    n_gates = resources.num_gates
    depth = resources.depth
    # gate_types = resources.gate_types
    # gate_sizes = resources.gate_sizes

    # training
    start_time = time.time()
    train_results_dict = train_quantum_model_custom_VQC(
        VQC_code=VQC_code,
        VQC_weights_shape=VQC_weights_shape,
        kernel_size=kernel_size,
        stride=stride,
        VQC_output_dim=VQC_output_dim,
        epochs=epochs,
        show_progress=True,
    )
    end_time = time.time()

    path = NEW_RUNS_DIR / "QuanvNN_runs_log"
    path_csv = (
        NEW_RUNS_DIR / "QuanvNN_runs_log.csv"
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


if __name__ == "__main__":
    code = """
n_qubits = 7
input_length = 7 # this is the kernel_size. It's how many inputs each VQC gets. The size of the sliding window.
q_out = 21 # VQC_output_dim (number of outputs from the VQC)
dev = qml.device("default.qubit", wires=n_qubits)
n_layers = 2

@qml.qnode(dev, interface="torch", diff_method='best')
def circuit(inputs, weights):
    inputs_scaled = inputs
    for i in range(input_length):
        qml.RY(inputs_scaled[i], wires=i)
        qml.RZ(
            inputs_scaled[i] * 0.5, wires=i
        )  # small extra encoding in a different axis

    # variational blocks with data re-uploading and alternating entanglers
    for l in range(n_layers):
        # parametric single-qubit rotations (trainable)
        for i in range(n_qubits):
            # Rot is expressive (3 params per qubit)
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

        # alternating entanglement:
        if (l % 2) == 0:
            # even layer: ring entangler 0-1,1-2,...,n-1-0
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        else:
            # odd layer: ladder / pair entangler (0-1,2-3,4-5,...). wrap last if odd count
            i = 0
            while i + 1 < n_qubits:
                qml.CNOT(wires=[i, i + 1])
                i += 2
            if n_qubits % 2 == 1:
                # connect last qubit with qubit 0 to avoid isolation
                qml.CNOT(wires=[n_qubits - 1, 0])

        # data re-uploading: re-inject the classical inputs (as small rotations)
        for i in range(input_length):
            qml.RZ(
                inputs_scaled[i] * 0.5, wires=i
            )  # small extra encoding in a different axis

    # build outputs: X, Y, Z per qubit (ordered per qubit)
    outputs = []
    for i in range(n_qubits):
        outputs.append(qml.PauliX(i))
        outputs.append(qml.PauliY(i))
        outputs.append(qml.PauliZ(i))

    # if more outputs required than 3*n_qubits, add ZZ correlators
    base = 3 * n_qubits
    if q_out > base:
        extra_needed = q_out - base
        k = 0
        while k < extra_needed:
            a = k % n_qubits
            b = (a + 1) % n_qubits
            outputs.append(qml.PauliZ(a) @ qml.PauliZ(b))
            k += 1

    # return only up to q_out features (if caller set q_out smaller)
    outputs = [qml.expval(obs) for obs in outputs[:q_out]]
    return outputs
"""
    # without Orchestral the tool is a plain function
    result = TrainCustomQuanvNNTool(
        VQC_code=code,
        VQC_weights_shape=[2, 7, 3],  # n_layers, n_qubits, 3 params per qubit
        kernel_size=7,
        stride=2,
        epochs=15,
        VQC_output_dim=21,
    )
    print(result)
