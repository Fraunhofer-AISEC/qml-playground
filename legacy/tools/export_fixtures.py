"""Export golden test fixtures from the PyTorch reference implementation.

The web app (web/) re-implements the simulator, losses, gradients and Adam in TypeScript. This script runs
the original modules in legacy/app/ on fixed inputs and parameters and writes the results to
web/tests/fixtures/reference.json, which the Vitest suite compares against.

Usage (from the repository root, with legacy/requirements.txt installed):
    python legacy/tools/export_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "app"))

from data import datasets_torch as ds  # noqa: E402
from models.reuploading_classifier import QuantumReuploadingClassifier  # noqa: E402
from models.reuploading_regressor import QuantumReuploadingRegressor  # noqa: E402

OUT = ROOT / "web" / "tests" / "fixtures" / "reference.json"

CASES = [
    # (num_qubits, task, dataset, layers)
    (1, "classification", "circle", 3),
    (1, "classification", "tricrown", 4),
    (1, "classification", "4_squares", 2),
    (2, "classification", "circle", 3),
    (2, "classification", "tricrown", 4),
    (2, "classification", "wavy_lines", 5),
    (1, "regression", "fourier_2", 3),
    (2, "regression", "fourier_3", 4),
]
REGS = [("none", 0.0), ("l1", 0.05), ("l2", 0.05)]
N_SAMPLES = 8
ADAM_LR = 0.05
ADAM_STEPS = 3


def flat(t):
    return t.detach().numpy().astype(np.float64).ravel().tolist()


def complex_states(states):
    """List over layers of [n, D] complex tensors -> nested [layer][sample][re0, im0, re1, im1, ...]."""
    out = []
    for s in states:
        s = s.detach().reshape(N_SAMPLES, -1).numpy()
        inter = np.stack([s.real, s.imag], axis=-1).reshape(N_SAMPLES, -1)
        out.append(inter.astype(np.float64).tolist())
    return out


def make_model(num_qubits, task, dataset, layers):
    cls = QuantumReuploadingRegressor if task == "regression" else QuantumReuploadingClassifier
    return cls(name=dataset, num_qubits=num_qubits, layers=layers)


def set_params(qmodel, w, b):
    shape = tuple(qmodel.model.weights.shape)
    with torch.no_grad():
        qmodel.model.weights.copy_(torch.tensor(w, dtype=torch.float32).reshape(shape))
        qmodel.model.biases.copy_(torch.tensor(b, dtype=torch.float32).reshape(shape))


def loss_fn(qmodel, task, output_states, y):
    if task == "regression":
        return qmodel.mse_loss(output_states, y)
    return qmodel.loss(output_states, y)


def run_case(num_qubits, task, dataset, layers, seed):
    rs = np.random.RandomState(seed)
    n_params = layers * num_qubits * 3
    w = rs.normal(0, 1.2, n_params)
    b = rs.normal(0, 1.2, n_params)

    if task == "regression":
        X = torch.tensor(rs.uniform(-1, 1, (N_SAMPLES, 1)), dtype=torch.float32)
        y = torch.tensor(rs.uniform(-1, 1, N_SAMPLES), dtype=torch.float32)
    else:
        X = torch.tensor(rs.uniform(-1, 1, (N_SAMPLES, 2)), dtype=torch.float32)
        n_classes = ds.create_target(dataset, num_qubits=2)[0]
        y = torch.tensor(rs.randint(0, n_classes, N_SAMPLES), dtype=torch.long)

    qmodel = make_model(num_qubits, task, dataset, layers)
    set_params(qmodel, w, b)
    params = qmodel.model.state_dict()

    case = {
        "numQubits": num_qubits,
        "task": task,
        "dataset": dataset,
        "layers": layers,
        "x": flat(X),
        "y": flat(y),
        "weights": flat(params["weights"]),
        "biases": flat(params["biases"]),
    }

    states = qmodel.model(X)
    case["states"] = complex_states(states)

    # Loss + gradients with each regularization
    case["grads"] = {}
    for reg_type, strength in REGS:
        qmodel.model.zero_grad()
        out = qmodel.model(X)[-1]
        loss = loss_fn(qmodel, task, out, y) + qmodel._apply_regularization(reg_type, strength)
        loss.backward()
        case["grads"][reg_type] = {
            "strength": strength,
            "loss": float(loss.detach()),
            "weights": flat(qmodel.model.weights.grad),
            "biases": flat(qmodel.model.biases.grad),
        }

    # Predictions (no regularization)
    if task == "regression":
        y_pred, _ = qmodel.predict(X)
        case["predictions"] = flat(y_pred)
        case["evalLoss"] = qmodel.evaluate(X.numpy(), y.numpy())["loss"]
    else:
        preds, scores = qmodel.predict(X)
        case["predictions"] = flat(preds)
        # 1 qubit: scores is [classes, n]; 2 qubits: [n, classes]. Export as [n, classes].
        scores = scores.T if num_qubits == 1 else scores
        case["scores"] = scores.detach().numpy().astype(np.float64).tolist()
        case["evalLoss"] = qmodel.evaluate(X.numpy(), y.numpy())["loss"]

    # Adam with persistent state over several full-batch steps
    set_params(qmodel, w, b)
    opt = torch.optim.Adam(qmodel.model.parameters(), lr=ADAM_LR, betas=(0.9, 0.999))
    adam_steps = []
    for _ in range(ADAM_STEPS):
        opt.zero_grad()
        out = qmodel.model(X)[-1]
        loss = loss_fn(qmodel, task, out, y)
        loss.backward()
        opt.step()
        adam_steps.append({"weights": flat(qmodel.model.weights), "biases": flat(qmodel.model.biases)})
    case["adam"] = {"lr": ADAM_LR, "steps": adam_steps}
    return case


def dataset_labels():
    """Label functions on fixed points for every classification dataset."""
    rs = np.random.RandomState(7)
    points = torch.tensor(rs.uniform(-1, 1, (400, 2)), dtype=torch.float32)
    out = {"points": flat(points), "labels": {}}
    for name in ds.classification_datasets:
        _, labels = getattr(ds, f"_{name}")(points)
        out["labels"][name] = flat(labels)
    return out


def targets():
    out = {}
    for name in ds.classification_datasets:
        t = ds.create_target(name, num_qubits=1).numpy()
        out[name] = np.stack([t.real, t.imag], axis=-1).reshape(t.shape[0], -1).astype(np.float64).tolist()
    return out


def fourier():
    out = {}
    for name in ds.regression_datasets:
        x, y = ds._fourier(name, samples=100)
        out[name] = {"x": flat(x), "y": flat(y)}
    return out


def main():
    fixtures = {
        "cases": [run_case(*c, seed=100 + i) for i, c in enumerate(CASES)],
        "datasets": dataset_labels(),
        "targets": targets(),
        "fourier": fourier(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, {len(fixtures['cases'])} cases)")


if __name__ == "__main__":
    main()
