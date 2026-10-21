import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from data.datasets_torch import batch_loader

logger = logging.getLogger(" [Model]")


class QuantumReuploadingModelBase(ABC):
    """Base class for quantum re-uploading models supporting 1 or 2 qubits.
    
    This abstract base class provides common functionality for both classification
    and regression tasks using quantum re-uploading circuits.
    """

    def __init__(self, name: str, num_qubits: int, layers: int):
        """Initialize the quantum re-uploading model.
        
        Args:
            name (str): Name of the problem/dataset
            num_qubits (int): Number of qubits (1 or 2)
            layers (int): Number of re-uploading layers
        """
        self.name = name
        self.num_qubits = num_qubits
        self.layers = layers
        self.current_epoch = 0

        self.loss = nn.MSELoss()
        
        # Initialize the quantum model backend
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        """Initialize the quantum model backend. Must be implemented by subclasses."""
        pass

    # ----------------- Serialization -----------------
    def load_model(self, model_parameters: dict):
        """Load model parameters from a dictionary.
        
        This method loads a previously saved model configuration and parameters,
        restoring the model state including weights and biases.
        
        Args:
            model_parameters (dict): Dictionary containing model configuration and parameters,
                with keys 'config', 'weights', and 'biases'.
        """
        config = model_parameters["config"]

        self.name = config["name"]
        self.num_qubits = config["num_qubits"]
        self.layers = config["num_layers"]
        self.current_epoch = config["epoch"]

        weights = model_parameters["weights"]
        biases = model_parameters["biases"]

        if self.num_qubits > 1:
            weights = torch.reshape(weights, (self.layers, self.num_qubits, 3))
            biases = torch.reshape(biases, (self.layers, self.num_qubits, 3))

        state_dict = OrderedDict()
        state_dict["weights"] = weights
        state_dict["biases"] = biases

        self.model.load_state_dict(state_dict)
        
        # Call hook for subclass-specific loading
        self._post_load_hook()
        
        logger.debug(f"{self.__class__.__name__} model loaded. "
                     f"Qubits={self.num_qubits}, Layers={self.layers}, Epoch={self.current_epoch}")

    def _post_load_hook(self):
        """Hook for subclass-specific operations after loading. Override if needed."""
        pass

    def save_model(self) -> dict:
        """Save the current model configuration and parameters.
        
        This method creates a dictionary with the model's configuration and
        parameters, which can be used to later restore the model's state using
        the load_model method.
        
        Returns:
            dict: Dictionary containing model configuration and parameters,
                with keys 'config', 'weights', and 'biases'.
        """
        config = {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_layers": self.layers,
            "epoch": self.current_epoch,
        }

        parameters = self.model.state_dict()
        weights = parameters["weights"].detach().numpy()
        biases = parameters["biases"].detach().numpy()

        if self.num_qubits > 1:
            weights = np.reshape(weights, (self.layers * self.num_qubits, 3))
            biases = np.reshape(biases, (self.layers * self.num_qubits, 3))

        model_parameters = {
            "config": config,
            "weights": pd.DataFrame(weights).to_json(orient='values'),
            "biases": pd.DataFrame(biases).to_json(orient='values')
        }
        
        logger.debug(f"{self.__class__.__name__} model saved. "
                     f"Qubits={self.num_qubits}, Layers={self.layers}, Epoch={self.current_epoch}")
        
        return model_parameters

    # ----------------- Training -----------------
    @abstractmethod
    def predict(self, X):
        """Make predictions on input data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate model on data. Must be implemented by subclasses."""
        pass

    def train_single_epoch(self, X, y, lr, batch_size, reg_type, reg_strength):
        """Train for one epoch. Must be implemented by subclasses."""
        self.model.train()
        opt = Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        for Xbatch, ybatch in batch_loader(X, y, batch_size=batch_size):
            opt.zero_grad()
            output_states = self.model(Xbatch)[-1]
            loss = self.loss(output_states, ybatch)

            # Apply regularization if selected
            loss += self._apply_regularization(reg_type, reg_strength)

            loss.backward()
            opt.step()

        self.current_epoch += 1

    def _apply_regularization(self, reg_type: str, reg_strength: float) -> torch.Tensor:
        """Apply L1 or L2 regularization penalty.
        
        Args:
            reg_type (str): Type of regularization ('none', 'l1', 'l2')
            reg_strength (float): Strength of regularization
            
        Returns:
            torch.Tensor: Regularization penalty value
        """
        penalty = torch.tensor(0.0)
        if reg_type == "l1":
            for p in self.model.parameters():
                penalty = penalty + reg_strength * torch.sum(torch.abs(p))
        elif reg_type == "l2":
            for p in self.model.parameters():
                penalty = penalty + reg_strength * torch.sum(p * p)
        return penalty
