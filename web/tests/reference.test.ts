import { describe, expect, it } from 'vitest';
import reference from './fixtures/reference.json';
import { QuantumModel } from '../src/model/qmodel';
import type { DataSet, DatasetName, TaskType } from '../src/data/datasets';
import { fourierCurve, labelPoint, classificationDatasets, type ClassificationDataset } from '../src/data/datasets';
import { singleQubitTargets } from '../src/data/targets';
import type { RegType } from '../src/model/regularization';
import type { NumQubits } from '../src/sim/circuit';

// The reference computes in float32 (complex64), so parity is checked at float32 precision.
const ATOL = 2e-5;
const RTOL = 1e-4;

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string, atol = ATOL) {
  expect(actual.length, `${label}: length`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const tol = atol + RTOL * Math.abs(expected[i]);
    if (Math.abs(actual[i] - expected[i]) > tol) {
      throw new Error(`${label}[${i}]: got ${actual[i]}, expected ${expected[i]} (tol ${tol})`);
    }
  }
}

type Case = (typeof reference.cases)[number];

function setup(c: Case) {
  const model = new QuantumModel({
    task: c.task as TaskType,
    dataset: c.dataset as DatasetName,
    numQubits: c.numQubits as NumQubits,
    numLayers: c.layers,
  });
  model.weights.set(c.weights);
  model.biases.set(c.biases);
  const n = c.y.length;
  const set: DataSet = { n, features: c.x.length / n, x: Float64Array.from(c.x), y: Float64Array.from(c.y) };
  const all = Int32Array.from({ length: n }, (_, i) => i);
  return { model, set, all, n };
}

describe.each(reference.cases.map((c) => [`${c.numQubits}q ${c.task} ${c.dataset} L=${c.layers}`, c] as const))(
  'PyTorch parity: %s',
  (_name, c) => {
    it('per-layer states', () => {
      const { model, set, n } = setup(c);
      const states = model.forwardAll(set);
      const ss = model.stateSize;
      for (let l = 0; l < c.layers; l++) {
        for (let i = 0; i < n; i++) {
          const off = (i * c.layers + l) * ss;
          expectClose(states.subarray(off, off + ss), c.states[l][i], `layer ${l} sample ${i}`);
        }
      }
    });

    it.each(['none', 'l1', 'l2'] as RegType[])('loss and adjoint gradients (reg=%s)', (reg) => {
      const { model, set, all, n } = setup(c);
      const ref = c.grads[reg as keyof typeof c.grads];
      const loss = model.lossAndGrad(set, all, 0, n, {
        lr: 0,
        batchSize: n,
        regType: reg,
        regStrength: ref.strength,
      });
      expectClose([loss], [ref.loss], 'loss');
      expectClose(model.gradient.subarray(0, model.P), ref.weights, 'dL/dW');
      expectClose(model.gradient.subarray(model.P), ref.biases, 'dL/dB');
    });

    it('evaluation and predictions', () => {
      const { model, set } = setup(c);
      const ev = model.evaluate(set);
      expectClose([ev.loss], [c.evalLoss], 'eval loss');
      if (c.task === 'regression') {
        expectClose(ev.predictions, c.predictions, 'y_pred');
      } else {
        expect(Array.from(ev.predictions)).toEqual(c.predictions);
        const scores = new Float64Array(model.numClasses);
        const last = (c.layers - 1) * model.stateSize;
        c.scores!.forEach((row, i) => {
          model.scores(ev.states, i * c.layers * model.stateSize + last, scores);
          expectClose(scores, row, `scores ${i}`);
        });
      }
    });

    it('Adam steps with persistent optimizer state', () => {
      const { model, set, all, n } = setup(c);
      // Parameters with a structurally zero gradient (e.g. the first RZ(φ) acting on |0⟩, a global
      // phase) get float32 noise ~1e-9 in torch, which Adam's g/(|g|+ε) amplifies. Skip those.
      const live = [...c.grads.none.weights, ...c.grads.none.biases].map((g) => Math.abs(g) > 1e-6);
      const pick = (a: ArrayLike<number>) => Array.from(a).filter((_, i) => live[i]);
      c.adam.steps.forEach((step, s) => {
        model.lossAndGrad(set, all, 0, n);
        model.adam.step(model.theta, model.gradient, c.adam.lr);
        expectClose(pick(model.theta), pick([...step.weights, ...step.biases]), `adam step ${s}`, 1e-4);
      });
    });
  },
);

describe('PyTorch parity: data', () => {
  it.each(Object.keys(classificationDatasets))('labels for %s', (name) => {
    const pts = reference.datasets.points;
    const expected = reference.datasets.labels[name as keyof typeof reference.datasets.labels];
    const labels = expected.map((_, i) => labelPoint(name as ClassificationDataset, pts[2 * i], pts[2 * i + 1]));
    expect(labels).toEqual(expected);
  });

  it.each(Object.keys(reference.targets))('single-qubit targets for %s', (name) => {
    const targets = singleQubitTargets(name as ClassificationDataset);
    const expected = reference.targets[name as keyof typeof reference.targets];
    expect(targets.length).toBe(expected.length);
    targets.forEach((t, i) => expectClose(t, expected[i], `target ${i}`));
  });

  it.each(Object.keys(reference.fourier))('fourier curve %s', (name) => {
    const [x, y] = fourierCurve(name, 100);
    const ref = reference.fourier[name as keyof typeof reference.fourier];
    expectClose(x, ref.x, 'x');
    expectClose(y, ref.y, 'y');
  });
});
