import logging
import torch
import numpy as np

from itertools import product

logger = logging.getLogger(" [Data]")


regression_datasets = {
    'fourier_1': 'Fourier series degree 1',
    'fourier_2': 'Fourier series degree 2',
    'fourier_3': 'Fourier series degree 3',
    'fourier_4': 'Fourier series degree 4',
    'fourier_5': 'Fourier series degree 5',
}

classification_datasets = {
    'circle': 'Circle',
    '3_circles': '3 Circles',
    'square': 'Square',
    '4_squares': '4 Squares',
    'crown': 'Crown',
    'tricrown': 'Tricrown',
    'wavy_lines': 'Wavy Lines',
}

def batch_loader(inputs, labels, batch_size):

    # Convert inputs and labels to tensors
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    labels = torch.as_tensor(labels)

    # Generate shuffled indices
    indices = torch.randperm(inputs.shape[0])

    # Shuffle inputs and labels
    inputs = inputs[indices]
    labels = labels[indices]

    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], labels[idxs]


def create_dataset(name, samples=1000, seed=0):
    """Function to create datasets.

    Classification: returns random 2D points with labels.
    Regression (fourier_*): returns univariate x and y=f(x) pairs (as tensors).

    Args:
        name (str): Name of the problem to create the dataset, classification names
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy_lines']
            or regression names ['fourier_1','fourier_2','fourier_3'].
        samples (int): Number of points in the set.
        seed (int): Random seed

    Returns:
        Tuple (x, y):
          - For classification: x is (N,2) tensor of points in [-1,1]^2, y is int labels tensor of shape (N,)
          - For regression: x is (N,1) tensor of x in [-1,1], y is float tensor (N,) with function values
    """
    logger.debug(f'Create new dataset "{name}" with n={samples} points. (Random Seed: {seed})')
    torch.manual_seed(seed)

    if name.startswith('fourier_'):
        return _fourier(name, samples, seed)

    points = 1 - 2 * torch.rand(samples, 2)

    creator = globals()[f"_{name}"]

    x, y = creator(points)
    return x, y


def _fourier(name, samples=500, seed=0):
    """
    Build a real Fourier sum from the complex series:
        coeff0 = 0.1
        coeff_k = 0.15 + 0.15j  ->  a_k = 0.3, b_k = -0.3  for k >= 1
    Dataset name must be 'fourier_<n>' with 1 <= n <= 5.
    """
    # Parse and validate name
    if not isinstance(name, str) or not name.startswith("fourier_"):
        raise NotImplementedError(f"Unknown fourier dataset {name}")
    try:
        degree = int(name.split("_", 1)[1])
    except Exception:
        raise NotImplementedError(f"Unknown fourier dataset {name}")
    if degree < 1 or degree > 5:
        raise NotImplementedError(f"Only orders 1–5 are implemented (got {degree})")

    # Generate univariate x in [-1,1]
    torch.manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, steps=samples).unsqueeze(1)

    # Map [-1,1] to [-pi,pi]
    pi = torch.pi
    t = x.squeeze(1) * pi

    # Coefficients from target_function
    a0 = 0.1       # zero-frequency term
    a_k = 0.3      # cos coefficients for k >= 1
    b_k = -0.3     # sin coefficients for k >= 1

    # Construct series through the requested degree
    y = a0 * torch.ones_like(t)
    for k in range(1, degree + 1):
        y = y + a_k * torch.cos(k * t) + b_k * torch.sin(k * t)

    # Normalize to [-1, 1]
    max_abs = y.abs().max()
    if max_abs > 0:
        y = y / max_abs

    return x, y


def create_target(name, num_qubits=1):
    """Function to create target states for classification.

    Args:
        name (str): Name of the problem to create the target states, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines']

    Returns:
        List of numpy arrays encoding target states that depend only on the number of classes of the given problem
    """
    if num_qubits == 1:
        if name in ['circle', 'square', 'crown']:
            targets = torch.view_as_complex(torch.Tensor([[[1, 0], [0, 0]],
                                                          [[0, 0], [1, 0]]]
                                                         )
                                            )
        elif name in ['tricrown']:
            targets = torch.view_as_complex(torch.Tensor([[[1, 0], [0, 0]],
                                                          [[np.cos(torch.pi / 3), 0], [np.sin(torch.pi / 3), 0]],
                                                          [[np.cos(torch.pi / 3), 0], [-np.sin(torch.pi / 3), 0]]]
                                                         )
                                            )
        elif name in ['4_squares', 'wavy_lines', '3_circles']:
            targets = torch.view_as_complex(torch.Tensor([[[1, 0], [0, 0]],
                                                          [[1 / np.sqrt(3), 0], [np.sqrt(2 / 3), 0]],
                                                          [[1 / np.sqrt(3), 0],
                                                           [np.sqrt(2 / 3) * np.exp(1j * 2 * np.pi / 3).real,
                                                            np.sqrt(2 / 3) * np.exp(1j * 2 * np.pi / 3).imag]],
                                                          [[1 / np.sqrt(3), 0],
                                                           [np.sqrt(2 / 3) * np.exp(-1j * 2 * np.pi / 3).real,
                                                            np.sqrt(2 / 3) * np.exp(-1j * 2 * np.pi / 3).imag]],
                                                          ]
                                                         )
                                            )
        else:
            raise NotImplementedError('This dataset is not implemented')
    elif num_qubits == 2:
        if name in ['circle', 'square', 'crown']:
            targets = [2, [0, 1]]
        elif name in ['tricrown']:
            targets = [3, [0, 1, 2]]
        elif name in ['4_squares', 'wavy_lines', '3_circles']:
            targets = [4, [0, 1, 2, 3]]
        else:
            raise NotImplementedError('This dataset is not implemented')
    else:
        raise  NotImplementedError(f'Target for {num_qubits} not implemented!')

    return targets


def _circle(points):
    labels = torch.where(torch.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi), 1, 0)
    return points, labels


def _3_circles(points):
    centers = torch.tensor([[-1, 1], [1, 0], [-.5, -.5]])
    radii = torch.tensor([1, np.sqrt(6 / np.pi - 1), 1 / 2])
    labels = torch.zeros(len(points), dtype=torch.long)

    for j, (c, r) in enumerate(zip(centers, radii)):
        labels = torch.where(torch.linalg.norm(points - c, axis=1) < r, 1 + j, labels)

    return points, labels


def _square(points):
    labels = torch.where(torch.max(torch.abs(points), axis=1).values > .5 * np.sqrt(2), 1, 0)
    return points, labels


def _4_squares(points):
    labels = torch.zeros(len(points), dtype=torch.long)
    ids = torch.where(torch.logical_and(points[:, 0] < 0, points[:, 1] > 0))
    labels[ids] = 1
    ids = torch.where(torch.logical_and(points[:, 0] > 0, points[:, 1] < 0))
    labels[ids] = 2
    ids = torch.where(torch.logical_and(points[:, 0] > 0, points[:, 1] > 0))
    labels[ids] = 3

    return points, labels


def _crown(points):
    c = torch.tensor([[0, 0], [0, 0]])
    r = torch.tensor([np.sqrt(.8), np.sqrt(.8 - 2/np.pi)])
    labels = torch.where(torch.logical_and(torch.linalg.norm(points - c[0], axis=1) < r[0],
                                           torch.linalg.norm(points - c[1], axis=1) > r[1]),
                         1, 0)
    return points, labels


def _tricrown(points):
    c = torch.tensor([[0, 0], [0, 0]])
    r = torch.tensor([np.sqrt(.8), np.sqrt(.8 - 2 / np.pi)])

    labels = torch.zeros(len(points), dtype=torch.long)
    ids = torch.where(torch.linalg.norm(points - c[0], axis=1) > r[0])
    labels[ids] = 2
    ids = torch.where(torch.logical_and(torch.linalg.norm(points - c[0], axis=1) < r[0],
                                        torch.linalg.norm(points - c[1], axis=1) > r[1])
                      )
    labels[ids] = 1

    return points, labels


def _wavy_lines(points):
    freq = 1

    def fun1(s):
        return s + np.sin(freq * np.pi * s)

    def fun2(s):
        return -s + np.sin(freq * np.pi * s)

    labels = torch.zeros(len(points), dtype=torch.long)

    ids = torch.where(torch.logical_and(points[:, 1] < fun1(points[:, 0]),
                                        points[:, 1] > fun2(points[:, 0])
                                        )
                      )
    labels[ids] = 1
    ids = torch.where(torch.logical_and(points[:, 1] > fun1(points[:, 0]),
                                        points[:, 1] < fun2(points[:, 0])
                                        )
                      )
    labels[ids] = 2
    ids = torch.where(torch.logical_and(points[:, 1] > fun1(points[:, 0]),
                                        points[:, 1] > fun2(points[:, 0])
                                        )
                      )
    labels[ids] = 3

    return points, labels
