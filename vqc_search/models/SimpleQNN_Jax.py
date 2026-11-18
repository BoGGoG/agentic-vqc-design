"""
Run directly: From the repository root, run:
`python -m vqc_search.models.SimpleQNN_Jax`
"""

import gc
import math
import time
from pathlib import Path

from vqc_search.paths import DATA_DIR, PLOTS_DIR

import h5py as h5
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from flax import nnx
from jax.tree_util import tree_map
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from vqc_search.utils.quantum_circuit import (
    create_pennylane_circuit,
    create_pennylane_circuit_jax,
    make_quantum_layer_custom_code,
)


def load_data(data_dir):
    data_save_path_train = data_dir / "train.h5"
    data_save_path_test = data_dir / "test.h5"
    with h5.File(data_save_path_train, "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]
    with h5.File(data_save_path_test, "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]
    return X_train_full, y_train_full, X_test_full, y_test_full


def normalize_series(X):
    return (X - X.min(axis=1, keepdims=True)) / (
        X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8
    )


def prepare_tensors(X, y):
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if len(y_tensor.shape) == 2:
        y_tensor = y_tensor[:, 0]
    return x_tensor, y_tensor


def numpy_collate(batch):
    xb, yb = default_collate(batch)
    return jnp.asarray(xb, dtype=jnp.float32), jnp.asarray(yb, dtype=jnp.float32)


class SimpleNN_Jax(nnx.Module):
    def __init__(self, num_hidden_1, num_hidden_2, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(0.01, rngs=rngs)
        self.relu = nnx.leaky_relu
        self.linear1 = nnx.Linear(21, num_hidden_1, rngs=rngs)
        self.linear2 = nnx.Linear(num_hidden_1, num_hidden_2, rngs=rngs)
        self.linear3 = nnx.Linear(num_hidden_2, 1, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.linear3(x)
        return out


def make_vqc(n_qubits: int, n_layers: int = 2):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(x, weights):
        # x: shape (n_qubits,)
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

        # weights: shape (n_layers, n_qubits, 3) for StronglyEntanglingLayers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Vectorize over batch dimension: x_batch shape (B, n_qubits)
    vmap_circuit = jax.vmap(circuit, in_axes=(0, None))
    return vmap_circuit


class VQCModule(nnx.Module):
    def __init__(self, n_qubits: int, n_layers: int, rngs: nnx.Rngs):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Initialize trainable quantum weights
        key = rngs.params()
        w0 = 0.01 * jax.random.normal(key, (n_layers, n_qubits, 3))
        self.qweights = nnx.Param(w0)

        # Build vmapped qnode
        self.vqc_fn = make_vqc(n_qubits, n_layers)

    def __call__(self, x):
        # x: (B, n_qubits) or (n_qubits,) depending on upstream
        if x.ndim == 1:
            x = x[None, :]
        y = self.vqc_fn(x, self.qweights[...])  # (B, n_qubits)
        return jnp.stack(y, axis=-1)  # (B, n_qubits)


class VQCModuleCustomCode(nnx.Module):
    qnode: object = nnx.data()
    vqc_fn: object = nnx.data()
    meta: dict = nnx.data()

    def __init__(self, VQC_code: str, VQC_weights_shape: tuple, rngs: nnx.Rngs):
        qnode, meta = create_pennylane_circuit_jax(VQC_code, prefer_name="circuit")
        self.qnode = qnode
        self.meta = meta

        key = rngs.params()
        w0 = 0.01 * jax.random.normal(key, VQC_weights_shape)
        self.qweights = nnx.Param(w0)

        # compile once
        self.vqc_fn = jax.jit(jax.vmap(self.qnode, in_axes=(0, None)))

    def __call__(self, x):
        # x: (B, n_qubits) or (n_qubits,) depending on upstream
        if x.ndim == 1:
            x = x[None, :]
        y = self.vqc_fn(x, self.qweights[...])  # (B, n_qubits)
        return jnp.stack(y, axis=-1)  # (B, n_qubits)


class FinalDense(nnx.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        dropout_val=0.0,
        activation=nnx.relu,
        rngs: nnx.Rngs = None,
    ):
        self.activation = activation
        self.dropout = nnx.Dropout(dropout_val, rngs=rngs)

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.linears = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

    def __call__(self, x):
        # hidden layers: Linear -> activation -> Dropout
        for lin in self.linears[:-1]:
            x = lin(x)
            x = self.activation(x)
            x = self.dropout(x)

        # output layer: Linear -> Sigmoid
        x = self.linears[-1](x)
        return nnx.sigmoid(x)


class SimpleQNN_Jax(nnx.Module):
    def __init__(self, hparams, rngs: nnx.Rngs):
        self.dr = hparams.get("dropout")  # dropout rate
        self.encoder = nnx.Linear(
            hparams["input_size"], hparams["q_enc_size"], rngs=rngs
        )
        self.dropout = nnx.Dropout(self.dr, rngs=rngs)
        self.vqc = VQCModule(n_qubits=hparams["q_enc_size"], n_layers=2, rngs=rngs)

        in_dim = hparams["q_enc_size"]
        hidden_dims = []  # right now no hidden layers after VQC
        out_dim = 1
        self.final_dense = FinalDense(
            in_dim, hidden_dims, out_dim, self.dr, activation=nnx.leaky_relu, rngs=rngs
        )

    def __call__(self, x):
        x = self.encoder(x)
        x = (jnp.tanh(x) + 1.0) * (jnp.pi / 2.0)  # bring x to [0, pi] by using tanh
        x = self.vqc(x)
        out = self.final_dense(x)
        return out


class SimpleQNNCustomVQC_Jax(nnx.Module):
    def __init__(self, hparams, rngs: nnx.Rngs):
        self.dr = hparams.get("dropout")  # dropout rate
        self.encoder = nnx.Linear(
            hparams["input_size"], hparams["q_enc_size"], rngs=rngs
        )
        self.dropout = nnx.Dropout(self.dr, rngs=rngs)
        self.vqc = VQCModuleCustomCode(
            VQC_code=hparams["VQC_code"],
            VQC_weights_shape=hparams["VQC_weights_shape"],
            rngs=rngs,
        )

        in_dim = hparams["q_enc_size"]
        hidden_dims = []  # right now no hidden layers after VQC
        out_dim = 1
        self.final_dense = FinalDense(
            in_dim, hidden_dims, out_dim, self.dr, activation=nnx.leaky_relu, rngs=rngs
        )

    def __call__(self, x):
        x = self.encoder(x)
        x = (jnp.tanh(x) + 1.0) * (jnp.pi / 2.0)  # bring x to [0, pi] by using tanh
        x = self.vqc(x)  # (B, n_qubits)
        out = self.final_dense(x)
        return out


def train(model, train_dataloader, eval_dataloader, num_epochs):
    # optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.01))
    optimizer = nnx.ModelAndOptimizer(model, optax.adam(learning_rate=0.01))
    train_rmse_history = []
    val_rmse_history = []

    for epoch in (pbar := tqdm(range(num_epochs))):
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        epoch_train_loss = 0.0
        for batch in train_dataloader:
            batch_loss = train_step(model, optimizer, batch)
            epoch_train_loss += batch_loss.item() * batch[0].shape[0]
        epoch_train_loss /= len(train_dataloader.dataset)

        epoch_val_loss = eval(model, eval_dataloader)
        train_rmse_history.append(epoch_train_loss)
        val_rmse_history.append(epoch_val_loss)

        pbar.set_postfix(
            train_rmse=epoch_train_loss,
            val_rmse=epoch_val_loss,
        )
    out = {
        "train_rmse_history": train_rmse_history,
        "val_rmse_history": val_rmse_history,
    }
    return out


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        pred = model(batch[0])
        loss = jnp.sqrt(jnp.mean((pred.squeeze() - batch[1]) ** 2))
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model, batch):
    pred = model(batch[0])
    return jnp.sqrt(jnp.mean((pred.squeeze() - batch[1]) ** 2))


def eval(model, eval_dataloader):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in eval_dataloader:
        loss = eval_step(model, (xb, yb))
        total = total + loss * xb.shape[0]
        n += xb.shape[0]
    return (total / n).item()  # one sync at end


def test(model, test_dataloader):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in test_dataloader:
        loss = eval_step(model, (xb, yb))
        total = total + loss * xb.shape[0]
        n += xb.shape[0]
    return (total / n).item()


if __name__ == "__main__":
    this_dir = Path(__file__).parent
    data_save_dir = DATA_DIR / "gaussian_peak_noise_0_01"
    plot_dir = PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    show_progress = True
    epochs = 10
    batch_size = 16

    X_train_full, y_train_full, X_test_full, y_test_full = load_data(data_save_dir)

    n_train, n_val, n_test = 1000, 250, 500
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train : n_train + n_val]
    y_val = y_train_full[n_train : n_train + n_val]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    X_train = normalize_series(X_train)
    X_val = normalize_series(X_val)
    X_test = normalize_series(X_test)

    x_train, y_train = prepare_tensors(X_train, y_train)
    x_val, y_val = prepare_tensors(X_val, y_val)
    x_test, y_test = prepare_tensors(X_test, y_test)
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")

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

    # hparams = {
    #     "input_size": 21,
    #     "dropout": 0.1,
    #     "q_enc_size": 7,
    # }
    model = SimpleQNNCustomVQC_Jax(hparams, rngs=nnx.Rngs(42))
    nnx.display(model)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numpy_collate,  # drop_last=True
    )  # might want drop_last=True for jax
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate
    )

    t0 = time.time()
    train_dict = train(model, train_dataloader, val_dataloader, num_epochs=epochs)
    t1 = time.time()
    print(
        f"Training time for {epochs} epochs: {t1 - t0:.2f} seconds (including compilation)"
    )
    test_rmse = test(model, test_dataloader)
    print(f"Test RMSE: {test_rmse:.4f}")
