import logging
import torch
import numpy as np

from itertools import product

logger = logging.getLogger(" [Data]")


def batch_loader(inputs, labels, batch_size):

    inputs = torch.Tensor(inputs)
    labels = torch.from_numpy(labels)

    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], labels[idxs]


def create_dataset(name, samples=1000, seed=0):
    """Function to create training and test sets for classifying.

    Args:
        name (str): Name of the problem to create the dataset, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
        samples (int): Number of points in the set, randomly located.
            This argument is ignored if grid is specified.
        seed (int): Random seed

    Returns:
        Dataset for the given problem (x, y)
    """
    logger.debug(f'Create new dataset "{name}" with n={samples} points. (Random Seed: {seed})')
    torch.manual_seed(seed)
    points = 1 - 2 * torch.rand(samples, 2)

    creator = globals()[f"_{name}"]

    x, y = creator(points)
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
