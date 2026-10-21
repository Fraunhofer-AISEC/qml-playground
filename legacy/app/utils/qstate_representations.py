import numpy as np


def compute_tangle(states):
    """Compute the tangle (entanglement measure) for a set of two-qubit states.

    The tangle is a measure of entanglement for two-qubit systems, derived from
    the squared concurrence. For pure states, it quantifies the amount of
    non-local quantum correlation.

    Args:
        states (numpy.ndarray): Array of two-qubit state vectors, where each state
            is represented as a complex array of amplitudes [a00, a01, a10, a11].
            Shape can be (4,) for a single state or (n, 4) for multiple states.

    Returns:
        numpy.ndarray: Tangle values for each input state. For separable states,
            the tangle is 0. For maximally entangled states, the tangle is 1.
    """
    states = np.atleast_2d(states)
    c = 2 * np.abs(states[:, 0] * states[:, 3] - states[:, 1] * states[:, 2])
    #c = 2 * np.abs(a00 * a11 - a01 * a10)
    return c ** 2


def convert_to_bloch_vector(rho):
    """Convert a density matrix to a Bloch vector.

    The Bloch vector representation maps a qubit state to a 3D real vector,
    providing a geometric visualization of the state on the Bloch sphere.

    Args:
        rho (numpy.ndarray): 2x2 density matrix representing a qubit state.

    Returns:
        list: Bloch vector [x, y, z] coordinates. Pure states lie on the sphere
            surface (vector of length 1), while mixed states are inside the sphere.
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real

    return [ax, ay, az]


def convert_state_vector_to_bloch_vector(psi):
    """Convert a pure state vector to a Bloch vector.

    This function first converts the pure state vector to its density matrix
    representation, then calculates the corresponding Bloch vector.

    Args:
        psi (numpy.ndarray): Complex vector representing a pure qubit state,
            typically of the form [alpha, beta] where |alpha|² + |beta|² = 1.

    Returns:
        list: Bloch vector [x, y, z] coordinates representing the state on the
            Bloch sphere surface.
    """
    rho = np.outer(psi, np.conj(psi))
    return convert_to_bloch_vector(rho)