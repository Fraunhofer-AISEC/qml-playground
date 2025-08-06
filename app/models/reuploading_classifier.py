import logging
from collections import OrderedDict

import pandas as pd
import numpy as np

import torch
from torch.optim import Adam

from backends.torch_single_qubit_reuploading import SingleQubitReuploadingTorch, fidelity
from backends.torch_multi_qubit_reuploading import MultiQubitReuploadingTorch

from data.datasets_torch import create_target, batch_loader

logger = logging.getLogger(" [Model]")


class QuantumReuploadingClassifier:

    def __init__(self, name, num_qubits, layers):
        """Class with all computations needed for classification.

        Args:
            name (str): Name of the problem to create the dataset, to choose between
                ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
            num_qubits (int): Number of qubits.
            layers (int): Number of layers to use in the classifier.

        Returns:
            Dataset for the given problem (x, y).
        """
        self.name = name
        self.num_qubits = num_qubits
        self.layers = layers

        if self.num_qubits == 1:
            self.model = SingleQubitReuploadingTorch(num_layers=layers)
            self.loss = self.fidelity_loss
        elif self.num_qubits == 2:
            self.model = MultiQubitReuploadingTorch(n_qubits=2, n_layers=self.layers, meas_qubits=[0, 1])
            self.loss = self.cross_entropy_loss
        else:
            raise NotImplementedError

        self.target = create_target(name=name, num_qubits=self.num_qubits)
        self.states = []

        self.current_epoch = 0


    def load_model(self, model_parameters):
        """Load model parameters into the classifier.
        
        This method loads a previously saved model configuration and parameters,
        restoring the model state including weights and biases.
        
        Args:
            model_parameters (dict): Dictionary containing model configuration and parameters,
                with keys 'config', 'weights', and 'biases'.
                
        Returns:
            None
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

        self.target = create_target(self.name, num_qubits=self.num_qubits)
        logger.debug(f"Model for {self.name} loaded from state_dict. "
                     f"Number of qubits: {self.num_qubits}. "
                    f"Number of layers: {self.layers}. "
                    f"Current epoch: {self.current_epoch}."
                    )


    def save_model(self):
        """Save the current model configuration and parameters.
        
        This method creates a dictionary with the model's configuration and
        parameters, which can be used to later restore the model's state using
        the load_model method.
        
        Returns:
            dict: Dictionary containing model configuration and parameters,
                with keys 'config', 'weights', and 'biases'.
        """
        config = { "name": self.name,
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

        logger.debug(f"Model for {self.name} saved to state_dict."
                    f" Number of qubits: {self.num_qubits}. "
                    f"Number of layers: {self.layers}. "
                    f"Current epoch: {self.current_epoch}."
                    )
        return model_parameters


    def renormalize_probabilities(self, probs):
        """Renormalize probability distribution to account for valid class indices.
        
        This method adjusts the probability distribution when the number of classes
        is less than the total number of possible quantum states (2^num_qubits).
        It redistributes probabilities so they sum to 1 over valid class indices.
        
        Args:
            probs (torch.Tensor): Probability distribution tensor of shape (batch_size, 2^num_qubits)
            
        Returns:
            torch.Tensor: Renormalized probability distribution tensor with shape (batch_size, num_classes)
        """
        num_classes = self.target[0]

        if num_classes ==  2 ** self.num_qubits:
            return probs

        class_idx = self.target[1]
        non_class_idx = [i for i in range(2 ** self.num_qubits) if i not in class_idx]

        non_class_prob = 1 - torch.sum(probs[:, non_class_idx], axis=1)
        renorm_probs = probs[:, class_idx] / torch.unsqueeze(non_class_prob, 1)

        return renorm_probs


    def fidelity_loss(self, output_states, y):
        """Method for computing the cost function for
        a given sample (in the datasets), using fidelity.

        Args:
            x (array): Point to create the circuit.
            y (int): label of x.

        Returns:
            float with the cost function.
        """
        target_states = self.target[y]

        fidelities = fidelity(output_states, target_states)
        cost = 1 - fidelities

        return torch.mean(cost)


    def cross_entropy_loss(self, output_states, target_labels):
        """Calculate cross-entropy loss between quantum output states and target labels.
        
        This method is used as the loss function for multi-qubit classification tasks.
        It converts quantum state amplitudes to probabilities, renormalizes them,
        and calculates the negative log-likelihood loss against target labels.
        
        Args:
            output_states (torch.Tensor): Output quantum states from the model
            target_labels (torch.Tensor): Target class labels
            
        Returns:
            torch.Tensor: Calculated cross-entropy loss
        """
        probs = torch.square(torch.abs(output_states))
        renorm_probs = self.renormalize_probabilities(probs)

        cost_fn = torch.nn.NLLLoss()
        cost = cost_fn(torch.log(renorm_probs), target_labels)

        return cost


    def predict(self, X):
        output_states = self.model(X)[-1]

        if self.num_qubits == 1:
            scores = []

            for target_state in self.target:
                fidelities = fidelity(output_states, target_state)
                scores.append(fidelities)

            scores = torch.stack(scores)
            predictions = torch.argmax(scores, dim=0)

        elif self.num_qubits == 2:
            probs = torch.square(torch.abs(output_states))
            scores = self.renormalize_probabilities(probs)

            predictions = torch.argmax(scores, dim=1)

        else:
            raise NotImplementedError

        return predictions, scores


    def evaluate(self, X, y):
        """Evaluate model performance on given data.

        This method computes predictions, loss, accuracy, and intermediate quantum
        states for the provided data points and labels. It's useful for assessing
        model performance on training or test datasets.

        Args:
            X (numpy.ndarray): Input features array of shape (n_samples, feature_dim)
            y (numpy.ndarray): Target labels array of shape (n_samples,)

        Returns:
            dict: Dictionary containing evaluation results with keys:
                - "loss": Loss value (float)
                - "accuracy": Accuracy as a proportion of correct predictions (float)
                - "scores": Prediction scores for each class (torch.Tensor)
                - "predictions": Predicted class labels (torch.Tensor)
                - "states": Intermediate quantum states from all layers (list of torch.Tensor)
        """
        X = torch.Tensor(X)
        y = torch.from_numpy(y)

        predictions, scores = self.predict(X)
        states = self.model(X)
        output_states = states[-1]

        loss = float(self.loss(output_states, y).detach())
        accuracy = float(torch.sum(y == predictions) / y.shape[0])

        results = {
            "loss": loss,
            "accuracy": accuracy,
            "scores": scores,
            "predictions": predictions,
            "states": states,
        }

        return results

    def train_single_epoch(self, X, y, lr=0.1, batch_size=32, reg_type="none", reg_strength=0.01):
        """Train the quantum model for a single epoch.

        This method performs one epoch of training using mini-batch gradient descent.
        It updates the model parameters to minimize the loss function on the provided
        training data.

        Args:
            X (numpy.ndarray): Training features of shape (n_samples, feature_dim)
            y (numpy.ndarray): Training labels of shape (n_samples,)
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            batch_size (int, optional): Size of mini-batches for training. Defaults to 32.
            reg_type (str, optional): Type of regularization ('none', 'l1', 'l2'). Defaults to "none".
            reg_strength (float, optional): Strength of regularization. Defaults to 0.01.

        Returns:
            None
        """
        # Configure weight decay (L2 regularization) if selected
        weight_decay = 0.0
        if reg_type == "l2":
            weight_decay = reg_strength
            
        opt = Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        for Xbatch, ybatch in batch_loader(X, y, batch_size=batch_size):
            opt.zero_grad()
            output_states = self.model(Xbatch)[-1]
            loss = self.loss(output_states, ybatch)
            
            # Apply L1 regularization if selected
            if reg_type == "l1":
                l1_loss = 0
                for param in self.model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                loss += reg_strength * l1_loss
                
            loss.backward()
            opt.step()

        self.current_epoch += 1