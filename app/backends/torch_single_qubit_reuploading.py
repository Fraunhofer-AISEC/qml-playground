import logging

import torch
import torch.nn as nn


logger = logging.getLogger(" [Backends]")


def fidelity(state1, state2):
    """Calculate the fidelity between two quantum states.

    Fidelity is a measure of similarity between two quantum states, ranging from 0
    (orthogonal states) to 1 (identical states). This function computes the square of
    the inner product between the two states.

    Args:
        state1 (torch.Tensor): First quantum state tensor
        state2 (torch.Tensor): Second quantum state tensor

    Returns:
        torch.Tensor: Fidelity values between corresponding states
    """
    f = torch.conj(state1) * state2
    f = torch.square(torch.abs(torch.sum(f, axis=1)))
    return f


class SingleQubitReuploadingTorch(nn.Module):
    """PyTorch implementation of a single-qubit data re-uploading quantum circuit.

    This class implements a quantum variational circuit with a data re-uploading
    pattern, where classical data is encoded multiple times (layers) with
    trainable parameters. Each layer consists of single-qubit rotations parametrized
    by the input data and trainable weights and biases.

    Attributes:
        num_layers (int): Number of re-uploading layers in the circuit
        weights (nn.Parameter): Trainable weights for each layer
        biases (nn.Parameter): Trainable biases for each layer
        initial_state (torch.Tensor): Initial quantum state |0⟩
    """
    def __init__(self, num_layers):
        """Initialize the single-qubit re-uploading quantum circuit model.

        Args:
            num_layers (int): Number of layers in the circuit
        """
        super(SingleQubitReuploadingTorch, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.randn((self.num_layers, 3)), requires_grad=True)
        self.biases = nn.Parameter(torch.randn((self.num_layers, 3)), requires_grad=True)

        self.initial_state = torch.view_as_complex(torch.Tensor([[1, 0], [0, 0]]))

    def forward(self, X):
        """Forward pass through the quantum circuit.

        Applies the parametrized quantum circuit to the input data X, processing it
        through multiple re-uploading layers. Each layer applies a single-qubit
        unitary rotation based on the input data and trainable parameters.

        Args:
            X (torch.Tensor): Input data tensor of shape [batch_size, feature_dim]

        Returns:
            list: List of quantum states after each layer, with each state being
                  a complex tensor of shape [batch_size, 2]
        """
        X = nn.functional.pad(X, (0, 1), "constant", 1)
        XR = torch.transpose(X.T.repeat(self.num_layers, 1, 1), 0, 2)
        W = self.weights.T.repeat(X.shape[0], 1, 1)
        B = self.biases.T.repeat(X.shape[0], 1, 1)
        angles = W * XR + B

        ctheta = torch.cos(angles[:, 1, :] / 2)
        stheta = torch.sin(angles[:, 1, :] / 2)
        phi_plus_omega = (angles[:, 0, :] + angles[:, 2, :]) / 2
        phi_minus_omega = (angles[:, 0, :] - angles[:, 2, :]) / 2

        m00 = torch.exp(-1j * phi_plus_omega) * ctheta
        m01 = -torch.exp(1j * phi_minus_omega) * stheta
        m10 = torch.exp(-1j * phi_minus_omega) * stheta
        m11 = torch.exp(1j * phi_plus_omega) * ctheta
        M = torch.transpose(torch.stack((m00, m01, m10, m11)), 0, 2).reshape(self.num_layers, -1, 2, 2)

        S = torch.unsqueeze(self.initial_state.repeat(X.shape[0], 1), -1)
        states = []

        for u in M:
            S = torch.bmm(u, S)
            states.append(torch.squeeze(S))

        return states