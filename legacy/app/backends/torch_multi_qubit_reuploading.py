import logging

import torch

from .torch_state_vector_simulator import StateVecSimTorch


logger = logging.getLogger(" [Backends]")


class MultiQubitReuploadingTorch(StateVecSimTorch):
    """PyTorch implementation of a multi-qubit data re-uploading quantum circuit.

    This class extends the state vector simulator to implement a quantum variational
    circuit with a data re-uploading pattern for multiple qubits. The circuit consists
    of alternating layers of parameterized rotations (encoding the classical data)
    and entangling CNOT gates arranged in a ring topology.

    Attributes:
        Inherits all attributes from StateVecSimTorch base class.
    """
    def __init__(self, n_qubits,
                 n_layers,
                 meas_qubits=None,
                 init_weights_scale=1.,
                 gpu=False,
                 seed=42,
                 ):
        """Initialize the multi-qubit re-uploading quantum circuit model.

        Args:
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of re-uploading layers
            meas_qubits (list, optional): List of qubits to measure. Defaults to None.
            init_weights_scale (float, optional): Scale for initializing weights. Defaults to 1.0.
            gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__(n_qubits, n_layers, meas_qubits, init_weights_scale, gpu, seed)

    def forward(self, X):
        """Forward pass through the multi-qubit quantum circuit.

        Creates a quantum circuit based on the input data and applies it to the initial
        state |0...0⟩. The circuit consists of multiple layers, each containing
        parameterized rotation gates followed by entangling CNOT gates in a ring topology.

        Args:
            X (torch.Tensor): Input data tensor of shape [batch_size, feature_dim]

        Returns:
            list: List of quantum states after each layer, with each state being
                  a complex tensor representing the full quantum state vector
        """
        qubit_list = [i for i in range(self.n_qubits)]
        self.state = torch.zeros(2 ** self.n_qubits)
        self.state[0] = 1.
        self.state = self.state.unsqueeze(1).repeat(1, X.shape[0]).T.type(torch.complex64)
        angles = self.get_angles(X)

        states = []

        for l in range(self.n_layers):
            self.Rot(angles[:, l, :, :])
            self.CNOT("ring", n_samples=X.shape[0])
            states.append(self.state)

        return states
