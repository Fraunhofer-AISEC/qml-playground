import logging
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

from backends.torch_single_qubit_reuploading import SingleQubitReuploadingTorch
from backends.torch_multi_qubit_reuploading import MultiQubitReuploadingTorch
from models.reuploading_model_base import QuantumReuploadingModelBase

logger = logging.getLogger(" [Model]")


class QuantumReuploadingRegressor(QuantumReuploadingModelBase):
    """Quantum re-uploading regression model supporting 1 or 2 qubits.

    - Input is univariate x; broadcasting to all angle inputs is handled by the backends.
    - Prediction is expectation value of Z on the first measured qubit mapped to R.
    - Loss is mean squared error (MSE) between y_true and y_pred.
    - For two qubits we still measure only the first qubit's Z expectation.
    """

    def __init__(self, name: str, num_qubits: int, layers: int):
        super().__init__(name, num_qubits, layers)
        self.loss = self.mse_loss

    def _initialize_model(self):
        """Initialize the quantum model backend."""
        if self.num_qubits == 1:
            self.model = SingleQubitReuploadingTorch(num_layers=self.layers)
        elif self.num_qubits == 2:
            # measure first qubit; backend will keep full state vector
            self.model = MultiQubitReuploadingTorch(n_qubits=2, n_layers=self.layers, meas_qubits=[0])
        else:
            raise NotImplementedError("Regressor currently supports 1 or 2 qubits")

    # ----------------- Core ops -----------------
    @staticmethod
    def _expval_z_from_statevec(statevec: torch.Tensor) -> torch.Tensor:
        """Compute <Z> for first qubit for a batched state vector.
        For single qubit, statevec shape: [batch, 2]; for multi, full state vector is flattened per sample.
        """
        if statevec.ndim == 2 and statevec.shape[1] == 2:
            # single qubit: <Z> = |0|^2 - |1|^2
            p0 = torch.square(torch.abs(statevec[:, 0]))
            p1 = torch.square(torch.abs(statevec[:, 1]))
            return p0 - p1
        # multi-qubit: compute marginal probabilities for first qubit
        n_states = statevec.shape[1]
        half = n_states // 2
        p0 = torch.square(torch.abs(statevec[:, :half])).sum(dim=1)
        p1 = torch.square(torch.abs(statevec[:, half:])).sum(dim=1)
        return p0 - p1

    def mse_loss(self, output_states, y):
        """Method for computing the cost function for
        a given sample (in the datasets), using MSE.
        """
        cost_fn = nn.MSELoss()
        y_pred = self._expval_z_from_statevec(output_states)

        cost = cost_fn(torch.as_tensor(y_pred, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))

        return cost

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Return y_pred as <Z> and list of states per layer."""
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        states = self.model(X)
        output_state = states[-1]
        # MultiQubitReuploadingTorch keeps state as shape [batch, 2^n], SingleQubit as [batch, 2]
        if isinstance(output_state, list):
            # some backends could return list per sample; not the case here
            output_state = torch.stack(output_state)
        y_pred = self._expval_z_from_statevec(output_state)
        return y_pred, states

    def evaluate(self, X, y_true):
        """Evaluate model performance on regression data.
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            dict: Dictionary with 'loss', 'y_pred', and 'states'
        """
        X = torch.Tensor(X)
        y_true_t = torch.Tensor(y_true).flatten()
        y_pred, states = self.predict(X)

        cost_fn = nn.MSELoss()
        loss = float(cost_fn(y_pred, y_true_t).detach())

        results = {"loss": loss,
                   "y_pred": y_pred.detach(),
                   "states": states}

        return results


    # Inference-time MC with weight noise
    def predict_mc(self, X: torch.Tensor, sigma: float = 0.0, M: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run M stochastic forward passes by injecting Gaussian noise N(0, sigma) into weights and biases.
        Returns mean and std tensors of shape [N].
        """
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        if sigma <= 0.0 or M <= 1:
            y_pred, _ = self.predict(X)
            return y_pred, torch.zeros_like(y_pred)

        weights_orig = self.model.state_dict()["weights"].detach().clone()
        biases_orig = self.model.state_dict()["biases"].detach().clone()
        preds = []
        for _ in range(M):
            with torch.no_grad():
                noisy = OrderedDict()
                noisy["weights"] = weights_orig + sigma * torch.randn_like(weights_orig)
                noisy["biases"] = biases_orig + sigma * torch.randn_like(biases_orig)
                self.model.load_state_dict(noisy, strict=False)
                y_hat, _ = self.predict(X)
                preds.append(y_hat.detach())
        # restore
        self.model.load_state_dict(OrderedDict(weights=weights_orig, biases=biases_orig), strict=False)
        P = torch.stack(preds, dim=0)
        mean = P.mean(dim=0)
        std = P.std(dim=0)
        return mean, std
