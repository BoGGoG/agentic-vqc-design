"""
Run directly: From the repository root, run:
`python -m vqc_search.models.SimpleQNN`
"""

import gc
import math
import time
from pathlib import Path

from vqc_search.paths import DATA_DIR, PLOTS_DIR

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vqc_search.utils.quantum_circuit import (
    create_pennylane_circuit,
    make_quantum_layer_custom_code,
)


class SimpleQNN_NoVQC(nn.Module):
    """ """

    def __init__(self, hparams):
        super().__init__()
        self.q_enc_size = hparams["q_enc_size"]
        self.n_qubits = hparams["n_qubits"]
        self.dropout_val = hparams.get("dropout", 0.2)
        self.q_out_size = hparams.get("q_out_size")

        # layer norm as normalizer
        # self.nrm = nn.LayerNorm(hparams["input_size"])

        self.encoder = nn.Sequential()
        self.encoder.add_module(
            "linear1", nn.Linear(hparams["input_size"], hparams["q_enc_size"])
        )
        self.encoder.add_module("relu1", nn.LeakyReLU())
        self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_val))

        # placeholder for now, will be quantum
        self.vqc = nn.Sequential()
        self.vqc.add_module(
            "linear1", nn.Linear(hparams["q_enc_size"], self.q_out_size)
        )
        self.vqc.add_module("relu1", nn.LeakyReLU())
        self.vqc.add_module("dropout1", nn.Dropout(p=self.dropout_val))
        self.vqc.add_module("linear2", nn.Linear(self.q_out_size, self.q_out_size))
        self.vqc.add_module("relu2", nn.LeakyReLU())

        in_dim = self.q_out_size
        hidden_dims = [50, 50, 40]
        out_dim = 1
        self.final_dense = create_final_dense(
            in_dim, hidden_dims, out_dim, self.dropout_val, activation=nn.LeakyReLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L]
        returns: [B, 1]
        """
        # x = self.nrm(x)
        x = self.encoder(x)  # [B, q_enc_size]
        x = self.vqc(x)  # [B, q_out_size]
        x = self.final_dense(x)  # [B, 1]
        return x


def create_final_dense(in_dim, hidden_dims, out_dim, dropout_val, activation=nn.ReLU):
    layers = []
    prev_dim = in_dim
    for i, hdim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hdim))
        layers.append(activation())
        layers.append(nn.Dropout(p=dropout_val))
        prev_dim = hdim
    layers.append(nn.Linear(prev_dim, out_dim))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class SimpleQNN_CustomVQC(nn.Module):
    """ """

    def __init__(self, hparams):
        super().__init__()
        self.q_enc_size = hparams["q_enc_size"]
        self.dropout_val = hparams.get("dropout", 0.2)
        self.q_out_size = hparams.get("q_out_size")

        # layer norm as normalizer
        # self.nrm = nn.LayerNorm(hparams["input_size"])

        self.encoder = nn.Sequential()
        self.encoder.add_module(
            "linear1", nn.Linear(hparams["input_size"], hparams["q_enc_size"])
        )
        # self.encoder.add_module("relu1", nn.LeakyReLU())
        self.encoder.add_module("dropout_1", nn.Dropout(p=self.dropout_val))

        VQC_code = hparams["VQC_code"]
        VQC_weights_shape = hparams["VQC_weights_shape"]

        self.vqc = make_quantum_layer_custom_code(
            custom_VQC_code=VQC_code,
            weights_shape=VQC_weights_shape,
        )

        in_dim = self.q_out_size
        hidden_dims = []
        out_dim = 1
        self.final_dense = create_final_dense(
            in_dim, hidden_dims, out_dim, self.dropout_val, activation=nn.LeakyReLU
        )

        # print number of parameters in all layers
        n_params_encoder = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        n_params_vqc = sum(p.numel() for p in self.vqc.parameters() if p.requires_grad)
        n_params_final = sum(
            p.numel() for p in self.final_dense.parameters() if p.requires_grad
        )
        print(f"{n_params_encoder=} {n_params_vqc=} {n_params_final=}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L]
        returns: [B, 1]
        """
        # x = self.nrm(x)
        x = self.encoder(x)  # [B, q_enc_size]
        # bring x to [0, pi]
        x = (torch.tanh(x) + 1) * (math.pi / 2)
        x = [self.vqc(xx) for xx in x]
        x = torch.stack(x, dim=0)  # [B, q_out_size]
        x = self.final_dense(x)  # [B, 1]
        return x


def train_simple_QNNNoVQC():
    this_dir = Path(__file__).parent
    data_save_dir = DATA_DIR / "gaussian_peak_noise_0_01"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    plot_dir = PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    show_progress = True
    epochs = 10

    with h5.File(data_save_path_train, "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]

    with h5.File(data_save_path_test, "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]

    n_train = 1000
    n_val = 250  # validation is much faster than training
    n_test = 500  # testing is much faster than training
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train : n_train + n_val]
    y_val = y_train_full[n_train : n_train + n_val]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    # normalize time series data to [0, 1] individually
    X_train = (X_train - X_train.min(axis=1, keepdims=True)) / (
        X_train.max(axis=1, keepdims=True) - X_train.min(axis=1, keepdims=True) + 1e-8
    )
    X_val = (X_val - X_val.min(axis=1, keepdims=True)) / (
        X_val.max(axis=1, keepdims=True) - X_val.min(axis=1, keepdims=True) + 1e-8
    )
    X_test = (X_test - X_test.min(axis=1, keepdims=True)) / (
        X_test.max(axis=1, keepdims=True) - X_test.min(axis=1, keepdims=True) + 1e-8
    )

    # train the model
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    if len(y_val.shape) == 2:
        y_val = y_val[:, 0]  # make sure y_val is 2D

    # train the model
    batch_size = 16
    loss_hist = []
    loss_hist_val = []
    # MSE loss function
    loss_fn = torch.nn.MSELoss()

    hparams = {
        "input_size": X_train.shape[1],
        "q_enc_size": 50,
        "n_qubits": 50,
        "dropout": 0.1,
        "q_out_size": 50,
    }
    model = SimpleQNN_NoVQC(hparams)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-5,
    )

    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print(
        "Number of trainable parameters in quantum layer:",
        sum(p.numel() for p in model.vqc.parameters() if p.requires_grad),
    )
    if show_progress:
        print(f"Training for {epochs} epochs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        if epoch == 8:
            optimizer.param_groups[0]["lr"] *= 0.2
        if epoch == 12:
            optimizer.param_groups[0]["lr"] *= 0.2
            # print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
        if epoch == 15:
            optimizer.param_groups[0]["lr"] *= 0.2
            # print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
        model.train()
        # shuffle X_train and y_train
        idxs = np.arange(len(X_train))
        idxs = np.random.permutation(idxs)
        X_train = X_train[idxs]
        y_train = y_train[idxs]

        epoch_losses = []
        for i in tqdm(
            range(0, len(X_train), batch_size),
            desc=f"Epoch {epoch + 1}",
            leave=False,
            disable=not show_progress,
        ):
            model.train()
            optimizer.zero_grad()
            x_batch = torch.tensor(X_train[i : i + batch_size], dtype=torch.float32)
            x_batch = x_batch.to(device)
            y_batch = torch.tensor(
                y_train[i : i + batch_size, 0],
                dtype=torch.float32,  # only predict peak positions
            ).to(device)
            predicted = model.forward(x_batch)
            loss = loss_fn(predicted.squeeze(1), y_batch)
            # tqdm.write(
            #     f"Batch {i // batch_size + 1} loss: {loss.item():.4f}, "
            #     f"RMSE: {np.sqrt(loss.item()):.4f}"
            # )
            loss.backward()
            optimizer.step()
            gc.collect()
            epoch_losses.append(loss.item())
            loss_hist.append(loss.item())
            torch.cuda.empty_cache()
        epoch_loss = np.mean(epoch_losses)

        # validation
        with torch.no_grad():
            model.eval()
            optimizer.zero_grad()
            x_val = x_val.to(device)
            predicted = model.forward(x_val)
            val_loss = loss_fn(predicted.squeeze(1), y_val.to(device))
            loss_hist_val.append(val_loss.item())
        if show_progress:
            tqdm.write(
                f"Epoch {epoch + 1}: train RMSE: {np.sqrt(epoch_loss):.4f}, "
                f"validation RMSE: {np.sqrt(val_loss.item()):.4f}"
            )
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()
    loss_hist = np.array(loss_hist)
    loss_hist = np.sqrt(loss_hist)
    loss_hist_val = np.array(loss_hist_val)
    loss_hist_val = np.sqrt(loss_hist_val)

    # test the model on test set. Batch size 16
    test_losses = []

    # no mean, just individual losses
    indiv_loss_f = nn.MSELoss(reduction="none")
    y_test_pred = []
    if show_progress:
        print(f"Testing on test set of size {len(X_test)}...")
    for i in tqdm(range(0, len(X_test), batch_size), disable=not show_progress):
        model.eval()
        x_batch = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32).to(
            device
        )
        y_batch = torch.tensor(
            y_test[i : i + batch_size, 0],
            dtype=torch.float32,  # only predict peak positions
        )
        with torch.no_grad():
            predicted = model.forward(x_batch)
        y_test_pred.append(predicted.squeeze(1).detach().cpu().numpy())
        loss = indiv_loss_f(predicted.squeeze(1), y_batch.to(device))
        test_losses.append(loss.detach().cpu().numpy())
    test_losses = np.concatenate(test_losses, axis=0)
    test_losses = np.mean(test_losses)
    # tqdm.write(f"Test MSE: {test_losses:.4f}, RMSE: {np.sqrt(test_losses):.4f}")
    y_test_pred = np.concatenate(y_test_pred, axis=0)

    test_RMSE = np.sqrt(test_losses)
    print(f"Final Test MSE: {test_losses:.6f}, RMSE: {test_RMSE:.6f}")

    # plot loss history
    plt.figure(figsize=(10, 5))
    x_axis_train = list(range(len(loss_hist)))
    x_axis_val = np.linspace(0, len(loss_hist), len(loss_hist_val))
    plt.plot(x_axis_train, loss_hist, label="Train MSE")
    plt.plot(x_axis_val, loss_hist_val, label="Validation MSE")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss History")
    plt.legend()
    plt.grid()
    path = plot_dir / "SimpleQNN_1_loss_history.png"
    plt.savefig(path)
    print(f"Loss history plot saved to {path}")

    # plot a few test predictions
    n_plots = 7
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))
    for i in range(n_plots):
        axs[i].plot(X_test[i], label="Input Signal")
        axs[i].axvline(
            y_test[i][0] * (X_test.shape[1] - 1), color="g", label="True Peak Pos"
        )
        axs[i].axvline(
            y_test_pred[i] * (X_test.shape[1] - 1),
            color="r",
            label="Predicted Peak Pos",
        )
    plt.suptitle("Test Set Predictions")
    plt.legend()
    path = plot_dir / "SimpleQNN_1_test_predictions.png"
    plt.savefig(path)
    print(f"Test predictions plot saved to {path}")


if __name__ == "__main__":
    this_dir = Path(__file__).parent
    data_save_dir = DATA_DIR / "gaussian_peak_noise_0_01"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    plot_dir = PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    show_progress = True
    epochs = 10

    with h5.File(data_save_path_train, "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]

    with h5.File(data_save_path_test, "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]

    n_train = 1000
    n_val = 250  # validation is much faster than training
    n_test = 500  # testing is much faster than training
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train : n_train + n_val]
    y_val = y_train_full[n_train : n_train + n_val]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    # normalize time series data to [0, 1] individually
    X_train = (X_train - X_train.min(axis=1, keepdims=True)) / (
        X_train.max(axis=1, keepdims=True) - X_train.min(axis=1, keepdims=True) + 1e-8
    )
    X_val = (X_val - X_val.min(axis=1, keepdims=True)) / (
        X_val.max(axis=1, keepdims=True) - X_val.min(axis=1, keepdims=True) + 1e-8
    )
    X_test = (X_test - X_test.min(axis=1, keepdims=True)) / (
        X_test.max(axis=1, keepdims=True) - X_test.min(axis=1, keepdims=True) + 1e-8
    )

    # train the model
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    if len(y_val.shape) == 2:
        y_val = y_val[:, 0]  # make sure y_val is 2D

    # train the model
    batch_size = 16
    loss_hist = []
    loss_hist_val = []
    # MSE loss function
    loss_fn = torch.nn.MSELoss()

    hparams = {
        "input_size": X_train.shape[1],
        "q_enc_size": 9,
        "dropout": 0.1,
        "q_out_size": 9,
        "VQC_code": """
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
            """,
        "VQC_weights_shape": [9, 5],
    }
    model = SimpleQNN_CustomVQC(hparams)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-5,
    )

    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print(
        "Number of trainable parameters in quantum layer:",
        sum(p.numel() for p in model.vqc.parameters() if p.requires_grad),
    )
    if show_progress:
        print(f"Training for {epochs} epochs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    t0 = time.time()
    for epoch in range(epochs):
        if epoch == 8:
            optimizer.param_groups[0]["lr"] *= 0.2
        if epoch == 12:
            optimizer.param_groups[0]["lr"] *= 0.2
            # print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
        if epoch == 15:
            optimizer.param_groups[0]["lr"] *= 0.2
            # print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
        model.train()
        # shuffle X_train and y_train
        idxs = np.arange(len(X_train))
        idxs = np.random.permutation(idxs)
        X_train = X_train[idxs]
        y_train = y_train[idxs]

        epoch_losses = []
        for i in tqdm(
            range(0, len(X_train), batch_size),
            desc=f"Epoch {epoch + 1}",
            leave=False,
            disable=not show_progress,
        ):
            model.train()
            optimizer.zero_grad()
            x_batch = torch.tensor(X_train[i : i + batch_size], dtype=torch.float32)
            x_batch = x_batch.to(device)
            y_batch = torch.tensor(
                y_train[i : i + batch_size, 0],
                dtype=torch.float32,  # only predict peak positions
            ).to(device)
            predicted = model.forward(x_batch)
            loss = loss_fn(predicted.squeeze(1), y_batch)
            # tqdm.write(
            #     f"Batch {i // batch_size + 1} loss: {loss.item():.4f}, "
            #     f"RMSE: {np.sqrt(loss.item()):.4f}"
            # )
            loss.backward()
            optimizer.step()
            gc.collect()
            epoch_losses.append(loss.item())
            loss_hist.append(loss.item())
            torch.cuda.empty_cache()
        epoch_loss = np.mean(epoch_losses)

        # validation
        with torch.no_grad():
            model.eval()
            optimizer.zero_grad()
            x_val = x_val.to(device)
            predicted = model.forward(x_val)
            val_loss = loss_fn(predicted.squeeze(1), y_val.to(device))
            loss_hist_val.append(val_loss.item())
        if show_progress:
            tqdm.write(
                f"Epoch {epoch + 1}: train RMSE: {np.sqrt(epoch_loss):.4f}, "
                f"validation RMSE: {np.sqrt(val_loss.item()):.4f}"
            )
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()
    loss_hist = np.array(loss_hist)
    loss_hist = np.sqrt(loss_hist)
    loss_hist_val = np.array(loss_hist_val)
    loss_hist_val = np.sqrt(loss_hist_val)
    t1 = time.time()
    print(f"Training time for {epochs} epochs: {t1 - t0:.2f} seconds")

    # test the model on test set. Batch size 16
    test_losses = []

    # no mean, just individual losses
    indiv_loss_f = nn.MSELoss(reduction="none")
    y_test_pred = []
    if show_progress:
        print(f"Testing on test set of size {len(X_test)}...")
    for i in tqdm(range(0, len(X_test), batch_size), disable=not show_progress):
        model.eval()
        x_batch = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32).to(
            device
        )
        y_batch = torch.tensor(
            y_test[i : i + batch_size, 0],
            dtype=torch.float32,  # only predict peak positions
        )
        with torch.no_grad():
            predicted = model.forward(x_batch)
        y_test_pred.append(predicted.squeeze(1).detach().cpu().numpy())
        loss = indiv_loss_f(predicted.squeeze(1), y_batch.to(device))
        test_losses.append(loss.detach().cpu().numpy())
    test_losses = np.concatenate(test_losses, axis=0)
    test_losses = np.mean(test_losses)
    # tqdm.write(f"Test MSE: {test_losses:.4f}, RMSE: {np.sqrt(test_losses):.4f}")
    y_test_pred = np.concatenate(y_test_pred, axis=0)

    test_RMSE = np.sqrt(test_losses)
    print(f"Final Test MSE: {test_losses:.6f}, RMSE: {test_RMSE:.6f}")

    # plot loss history
    plt.figure(figsize=(10, 5))
    x_axis_train = list(range(len(loss_hist)))
    x_axis_val = np.linspace(0, len(loss_hist), len(loss_hist_val))
    plt.plot(x_axis_train, loss_hist, label="Train MSE")
    plt.plot(x_axis_val, loss_hist_val, label="Validation MSE")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss History")
    plt.legend()
    plt.grid()
    path = plot_dir / "SimpleQNN_1_loss_history.png"
    plt.savefig(path)
    print(f"Loss history plot saved to {path}")

    # plot a few test predictions
    n_plots = 7
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))
    for i in range(n_plots):
        axs[i].plot(X_test[i], label="Input Signal")
        axs[i].axvline(
            y_test[i][0] * (X_test.shape[1] - 1), color="g", label="True Peak Pos"
        )
        axs[i].axvline(
            y_test_pred[i] * (X_test.shape[1] - 1),
            color="r",
            label="Predicted Peak Pos",
        )
    plt.suptitle("Test Set Predictions")
    plt.legend()
    path = plot_dir / "SimpleQNN_1_test_predictions.png"
    plt.savefig(path)
    print(f"Test predictions plot saved to {path}")
