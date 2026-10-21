import logging

import torch
from torch.optim import Adam

from backends.torch_single_qubit_reuploading import SingleQubitReuploadingTorch, fidelity
from backends.torch_multi_qubit_reuploading import MultiQubitReuploadingTorch

from data.datasets_torch import create_target, batch_loader
from models.reuploading_model_base import QuantumReuploadingModelBase

logger = logging.getLogger(" [Model]")


class QuantumReuploadingClassifier(QuantumReuploadingModelBase):
    """Quantum re-uploading classification model supporting 1 or 2 qubits."""

    def __init__(self, name, num_qubits, layers):
        """Class with all computations needed for classification.

        Args:
            name (str): Name of the problem to create the dataset, to choose between
                ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
            num_qubits (int): Number of qubits.
            layers (int): Number of layers to use in the classifier.
        """
        super().__init__(name, num_qubits, layers)
        self.target = create_target(name=name, num_qubits=self.num_qubits)
        self.states = []

    def _initialize_model(self):
        """Initialize the quantum model backend and loss function."""
        if self.num_qubits == 1:
            self.model = SingleQubitReuploadingTorch(num_layers=self.layers)
            self.loss = self.fidelity_loss
        elif self.num_qubits == 2:
            self.model = MultiQubitReuploadingTorch(n_qubits=2, n_layers=self.layers, meas_qubits=[0, 1])
            self.loss = self.cross_entropy_loss
        else:
            raise NotImplementedError("Classifier currently supports 1 or 2 qubits")

    def _post_load_hook(self):
        """Recreate target states after loading model."""
        self.target = create_target(self.name, num_qubits=self.num_qubits)

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
            output_states (torch.Tensor): Output quantum states from the model
            y (int): label of x.

        Returns:
            torch.Tensor: Calculated fidelity loss
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
        """Make predictions on input data.
        
        Args:
            X (torch.Tensor): Input features
            
        Returns:
            tuple: (predictions, scores) tensors
        """
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
