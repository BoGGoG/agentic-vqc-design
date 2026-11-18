"""
Run directly: From the repository root, run:
`python -m vqc_search.models.Quanvolutional`
"""

import h5py as h5
from pathlib import Path

from vqc_search.paths import DATA_DIR, PLOTS_DIR

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import gc
from tqdm import tqdm
import numpy as np
from vqc_search.utils.quantum_circuit import (
    create_pennylane_circuit,
    make_quantum_layer_custom_code,
)


class Quanv1DCustomVQC(nn.Module):
    """
    A 1D quanvolution layer:
      - Extracts sliding windows along time axis (like Conv1d kernel),
      - Feeds each window through a quantum patch TorchLayer, VQC input values are in [0, π] after tanh-based scaling.
      - Returns a feature map [B, C_out, L_out].

    Parameters
    ----------
    in_channels : int
        Must be 1 for this implementation (extend by channel mixing if needed).
    out_channels : int
        Number of quantum features per window.
    kernel_size : int
        Window length along time axis.
    stride : int
        Window stride.
    padding : int
        Zero-padding on both sides of the sequence. Not implemented for now.
    VQC_code : str
        The PennyLane quantum circuit code as a string.
    VQC_weights_shape : list
        The shape of the weights tensor for the quantum circuit. This is needed to build the TorchLayer, because the circuit is defined dynamically from code.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        VQC_code: str = "",
        VQC_weights_shape: list = [],
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "This Quanv1D implementation expects in_channels=1. "
                "You can pre-mix channels with a 1x1 Conv1d before Quanv1D."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        # Build the quantum patch module
        self.qpatch = make_quantum_layer_custom_code(
            custom_VQC_code=VQC_code,
            weights_shape=VQC_weights_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        returns: [B, out_channels, L_out]
        """
        B, C, L = x.shape
        assert C == 1, "Use a 1x1 Conv1d to collapse to one channel first."

        # Use 2D unfold trick treating time as width (W) and height=1
        # Input for unfold must be [B, C, H, W]
        x4 = x.unsqueeze(2)  # [B,1,1,L]
        # Extract patches of size (1, kernel_size) with stride (1, stride)
        patches = F.unfold(
            x4,
            kernel_size=(1, self.kernel_size),
            dilation=(1, 1),
            padding=(0, self.padding),
            stride=(1, self.stride),
        )  # [B, C*1*kernel_size, L_out] = [B, kernel_size, L_out]
        # reshape to [B*L_out, kernel_size], because each patch is processed independently
        # note: here we break the batch into B*L_out independent patches
        B_, KL, Lout = patches.shape
        patches = patches.permute(0, 2, 1).contiguous()  # [B, L_out, kernel_size]
        patches = patches.view(-1, self.kernel_size)  # [B*L_out, kernel_size]

        # Normalize patches to [0,pi]
        patches_tanh = torch.tanh(patches)  # [-1, 1]
        patches_norm = (patches_tanh + 1.0) * 0.5 * torch.pi  # [0, π]

        # manually loop over batch dimensions
        qfeat = [self.qpatch(patch) for patch in patches_norm]
        qfeat = torch.stack(qfeat, dim=0)
        # qfeat = qfeat + self.bias  # affine tweak

        # back to [B, out_channels, L_out]
        qfeat = qfeat.view(B, Lout, self.out_channels).permute(0, 2, 1).contiguous()

        return qfeat


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class QuanvCNN1DCustomVQC(nn.Module):
    """ """

    def __init__(self, hparams: dict):
        super().__init__()
        self.seq_len = hparams["input_length"]
        self.kernel_size = hparams["kernel_size"]
        self.stride = hparams["stride"]
        self.padding = 0  # no padding for now
        self.VQC_output_dim = hparams.get("VQC_output_dim")
        self.quanvolution1 = Quanv1DCustomVQC(
            in_channels=1,
            VQC_code=hparams["VQC_code"],
            VQC_weights_shape=hparams["VQC_weights_shape"],
            out_channels=self.VQC_output_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.dropout_val = hparams["dropout"]
        self.quanv_out_len = (
            self.seq_len + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        # classical convolutional layer acting on output of quanvolution
        self.resblock1 = ResBlock1D(channels=self.VQC_output_dim, kernel_size=3)
        self.resblock2 = ResBlock1D(channels=self.VQC_output_dim, kernel_size=3)

        # use adaptive avg pooling to reduce length to fixed size (pool output size is out_channels of prev layer)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # set up final FC NN
        hidden_layers_dims = hparams.get("hidden_layers_dims", [])
        self.fc_final = nn.Sequential()
        # output length of the convolution
        current_dim = self.VQC_output_dim
        for dim in hidden_layers_dims:
            self.fc_final.append(nn.Linear(current_dim, dim, dtype=torch.float32))
            self.fc_final.append(nn.Dropout(self.dropout_val))
            self.fc_final.append(nn.ReLU())
            current_dim = dim
        self.output_dim = hparams["output_dim"]
        self.fc_final.append(
            nn.Linear(current_dim, self.output_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input shape [B, Cin, Lin]
        """
        # if no Cin, then add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # shape (batch_size, 1, seq_len)
        x = self.quanvolution1(x)  # shape (batch_size, conv1_channels, conv_out_len)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.adaptive_pool(x)  # shape (batch_size, 16, 1)
        x = x.squeeze(2)  # shape (batch_size, 16)
        x = self.fc_final(x)  # shape (batch_size, output_dim)
        return x


def train_quantum_model(
    model,
    epochs: int,
    show_progress: bool = True,
    data_save_dir="gaussian_peak_noise_0_01",
    device: str = "cpu",  # "cpu"
) -> dict:
    device = torch.device(device)
    this_dir = Path(__file__).parent
    data_save_ddir = DATA_DIR / data_save_dir
    data_save_path_train = data_save_ddir / "train.h5"
    data_save_path_test = data_save_ddir / "test.h5"

    with h5.File(data_save_path_train, "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]

    with h5.File(data_save_path_test, "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]

    n_train = 150
    n_val = 250  # validation is much faster than training
    n_test = 500  # testing is much faster than training
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train : n_train + n_val]
    y_val = y_train_full[n_train : n_train + n_val]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    # normalize data to [0, 1]
    X_train = (X_train - X_train.min(axis=1, keepdims=True)) / (
        X_train.max(axis=1, keepdims=True) - X_train.min(axis=1, keepdims=True) + 1e-8
    )
    X_val = (X_val - X_val.min(axis=1, keepdims=True)) / (
        X_val.max(axis=1, keepdims=True) - X_val.min(axis=1, keepdims=True) + 1e-8
    )
    X_test = (X_test - X_test.min(axis=1, keepdims=True)) / (
        X_test.max(axis=1, keepdims=True) - X_test.min(axis=1, keepdims=True) + 1e-8
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # train the QCNN model
    # train the model
    if len(y_val.shape) == 2:
        y_val = y_val[:, 0]

    # train the model
    batch_size = 16
    loss_hist = []
    loss_hist_val = []
    # MSE loss function
    loss_fn = torch.nn.MSELoss()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.10,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 8, 15], gamma=0.2
    )

    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    if show_progress:
        print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
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
            x_batch = X_train[i : i + batch_size].to(device)
            y_batch = y_train[i : i + batch_size, 0].to(device)
            predicted = model(x_batch)
            loss = loss_fn(predicted.squeeze(1), y_batch)
            # tqdm.write(
            #     f"Batch {i // batch_size + 1} loss: {loss.item():.4f}, "
            #     f"RMSE: {np.sqrt(loss.item()):.4f}"
            # )
            epoch_losses.append(loss.item())
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(epoch_losses)

        # validation
        with torch.no_grad():
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            model.eval()
            optimizer.zero_grad()
            predicted = model(X_val)
            val_loss = loss_fn(predicted.squeeze(1), y_val)
            loss_hist_val.append(val_loss.item())
        if show_progress:
            tqdm.write(
                f"Epoch {epoch + 1}: train RMSE: {np.sqrt(epoch_loss):.4f}, "
                f"validation RMSE: {np.sqrt(val_loss.item()):.4f}"
            )
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()
    loss_hist = np.array(loss_hist)
    # print(f"Final loss: {loss_hist_val[-1]}")
    # print(f"Final loss (RMSE): {np.sqrt(loss_hist_val[-1])}")

    # test the model on test set. Batch size 16
    test_losses = []

    # no mean, just individual losses
    indiv_loss_f = nn.MSELoss(reduction="none")
    if show_progress:
        print(f"Testing on test set of size {len(X_test)}...")
    for i in tqdm(range(0, len(X_test), batch_size), disable=not show_progress):
        model.eval()
        x_batch = X_test[i : i + batch_size].to(device)
        y_batch = y_test[i : i + batch_size, 0].to(device)
        with torch.no_grad():
            predicted = model(x_batch)
        loss = indiv_loss_f(predicted.squeeze(1), y_batch)
        test_losses.append(loss.detach().numpy())
    test_losses = np.concatenate(test_losses, axis=0)
    test_losses = np.mean(test_losses)
    # tqdm.write(f"Test MSE: {test_losses:.4f}, RMSE: {np.sqrt(test_losses):.4f}")

    test_RMSE = np.sqrt(test_losses)

    out = {
        "train_MSE_hist": np.array(loss_hist),
        "val_MSE_hist": np.array(loss_hist_val),
        "test_MSE": test_losses,
        "test_RMSE": test_RMSE,
    }

    return out


def train_quantum_model_custom_VQC(
    VQC_code: str,
    VQC_weights_shape: list,
    kernel_size: int,
    stride: int,
    VQC_output_dim: int,
    epochs: int,
    show_progress: bool = True,
):
    """
    Given a VQC code string and weights shape, train a QuanvCNN1DCustomVQC model on the Gaussian peak dataset.

    Parameters:
    VQC_code (str): The PennyLane quantum circuit code as a string.
    VQC_weights_shape (list): The shape of the weights tensor for the quantum circuit.
    kernel_size: Size of the 1D q convolution kernel (of the sliding window).
    VQC_output_dim (int): Number of outputs/features from the quantum layer.
    epochs (int): Number of training epochs.
    show_progress (bool): Whether to display training progress.

    Returns:
    Dict: A dictionary containing training history and test metrics. Keys include:
    - n_trainable_params_total: Total number of trainable parameters in the model.
    - n_trainable_params_VQC: Number of trainable parameters in the quantum layer.
    - train_MSE_hist: Training MSE history over epochs.
    - val_MSE_hist: Validation MSE history over epochs.
    - test_MSE: Test MSE on the test dataset.
    - test_RMSE: Test RMSE on the test dataset.

    Notes:
    The VQC_code should define a PennyLane QNode that accepts inputs and weights as parameters.
    The weights_shape should match the expected shape of the weights parameter in the VQC_code.
    """
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

    # number of trainable parameters (total)
    n_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_qpatch = sum(
        p.numel() for p in model.quanvolution1.qpatch.parameters() if p.requires_grad
    )
    out_dict = train_quantum_model(model, epochs=epochs, show_progress=show_progress)

    out_dict["n_trainable_params_total"] = n_params_total
    out_dict["n_trainable_params_VQC"] = n_params_qpatch

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

    out_dict = train_quantum_model_custom_VQC(
        VQC_code=code,
        VQC_weights_shape=[2, 7, 3],
        kernel_size=7,
        stride=2,
        VQC_output_dim=21,
        epochs=10,
        show_progress=True,
    )
    print(f"Test RMSE with custom VQC: {out_dict['test_RMSE']:.4f}")
