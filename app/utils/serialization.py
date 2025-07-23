import logging

from io import StringIO
import json

import numpy as np
import pandas as pd

import torch

from .qstate_representations import convert_state_vector_to_bloch_vector

logger = logging.getLogger(" [Serialization]")

def unserialize_model_dict(model_json: str) -> dict:
    """Convert serialized model parameters from JSON string to Python dictionary with PyTorch tensors.

    This function takes a JSON string representation of model parameters and converts it
    to a dictionary containing PyTorch tensors for weights and biases, which can be
    loaded into a quantum model.

    Args:
        model_json (str): JSON string containing serialized model parameters

    Returns:
        dict: Dictionary containing model configuration and PyTorch tensors for
            weights and biases
    """
    model_parameters = json.loads(model_json)
    model_parameters["weights"] = torch.Tensor(np.array(json.loads(model_parameters["weights"])))
    model_parameters["biases"] = torch.Tensor(np.array(json.loads(model_parameters["biases"])))

    return model_parameters


def serialize_quantum_states(num_qubits: int, states: list) -> str:
    """Serialize quantum states to a JSON string format based on the number of qubits.

    This function routes the serialization process to the appropriate helper function
    based on the number of qubits in the system - either to serialize_bloch_vectors
    for single-qubit systems or serialize_multiqubit_states for multi-qubit systems.

    Args:
        num_qubits (int): Number of qubits in the quantum system
        states (list): List of quantum state tensors to serialize

    Returns:
        str: JSON string representation of the serialized quantum states

    Raises:
        ValueError: If num_qubits is not a positive integer
    """
    if num_qubits == 1:
        return serialize_bloch_vectors(states)
    elif num_qubits > 1:
        return serialize_multiqubit_states(states)
    else:
        raise ValueError("num_qubits must be a positive integer.")


def unserialize_quantum_states(num_qubits: int, json_str: str) -> np.ndarray:
    """Convert serialized quantum states from JSON string back to numpy arrays.

    This function routes the deserialization process to the appropriate helper function
    based on the number of qubits in the system - either to unserialize_bloch_vectors
    for single-qubit systems or unserialize_multiqubit_states for multi-qubit systems.

    Args:
        num_qubits (int): Number of qubits in the quantum system
        json_str (str): JSON string representation of quantum states

    Returns:
        np.ndarray: Numpy array containing the deserialized quantum states

    Raises:
        ValueError: If num_qubits is not a positive integer
    """
    if num_qubits == 1:
        return unserialize_bloch_vectors(json_str)
    elif num_qubits > 1:
        return unserialize_multiqubit_states(json_str, num_qubits)
    else:
        raise ValueError("num_qubits must be a positive integer.")


def serialize_bloch_vectors(layer_state_tensors: list) -> str:
    """Convert quantum state tensors to Bloch vector representation and serialize to JSON.

    This function takes a list of PyTorch tensors representing quantum states across
    different layers, converts each state to its Bloch vector representation,
    and serializes the result to a JSON string for storage or transmission.

    Args:
        layer_state_tensors (list): List of PyTorch tensors containing quantum states
            for each layer

    Returns:
        str: JSON string representation of the Bloch vectors
    """
    layer_states = [s.detach().numpy() for s in layer_state_tensors]
    layer_bloch_vectors = []

    for layer_state in layer_states:
        bloch_vectors = np.array([convert_state_vector_to_bloch_vector(s) for s in layer_state])
        layer_bloch_vectors.append(bloch_vectors)

    layer_bloch_vectors = np.array(layer_bloch_vectors)
    layer_bloch_vectors = layer_bloch_vectors.reshape(len(layer_states), -1)

    json_str = pd.DataFrame(layer_bloch_vectors).to_json(orient='values')

    return json_str

def unserialize_bloch_vectors(json_str: str) -> np.ndarray:
    """Convert serialized Bloch vectors from JSON string back to numpy arrays.

    This function takes a JSON string representation of Bloch vectors and converts
    it back to a structured numpy array where each Bloch vector is represented
    as a 3D point (x, y, z) on the Bloch sphere.

    Args:
        json_str (str): JSON string containing serialized Bloch vectors

    Returns:
        np.ndarray: Numpy array of shape (num_layers, num_states, 3) containing
            the Bloch vector coordinates for each state in each layer
    """
    layer_bloch_vectors = pd.read_json(StringIO(json_str)).values
    layer_bloch_vectors = layer_bloch_vectors.reshape(layer_bloch_vectors.shape[0], -1, 3)

    return layer_bloch_vectors


def serialize_multiqubit_states(layer_state_tensors: list) -> str:
    """Serialize multi-qubit quantum state tensors to JSON string format.

    This function takes a list of PyTorch tensors representing multi-qubit
    quantum states across different layers, flattens the complex state vectors,
    and serializes them to a JSON string for storage or transmission.

    Args:
        layer_state_tensors (list): List of PyTorch tensors containing quantum states
            for each layer in the circuit

    Returns:
        str: JSON string representation of the multi-qubit quantum states
    """
    layer_states = [s.detach().numpy() for s in layer_state_tensors]
    layer_states = np.array(layer_states)

    layer_states = layer_states.reshape(len(layer_states), -1)

    json_str = pd.DataFrame(layer_states).to_json(orient='values')

    return json_str


def unserialize_multiqubit_states(json_str: str, num_qubits: int) -> np.ndarray:
    """Convert serialized multi-qubit states from JSON string back to numpy arrays.

    This function takes a JSON string representation of multi-qubit quantum states
    and converts it back to a structured numpy array. It handles complex numbers
    stored in JSON format by reconstructing them from their real and imaginary parts.

    Args:
        json_str (str): JSON string containing serialized multi-qubit quantum states
        num_qubits (int): Number of qubits in the quantum system

    Returns:
        np.ndarray: Numpy array of shape (num_layers, num_states, 2^num_qubits)
            containing the quantum state amplitudes for each state in each layer
    """
    def dict2complex(d):
        return d["real"] + d["imag"] * 1j

    layer_state_list = pd.read_json(StringIO(json_str)).values.tolist()
    layer_states = []
    for layer_state in layer_state_list:
        layer_states.append(list(map(dict2complex, layer_state)))

    layer_states = np.array(layer_states)
    layer_states = layer_states.reshape(layer_states.shape[0], -1, int(2 ** num_qubits))

    return layer_states