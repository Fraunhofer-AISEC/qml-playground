import { describe, expect, it } from 'vitest';
import { Circuit } from '../src/sim/circuit';
import { DEFAULT_SETTINGS, type Settings } from '../src/state/settings';
import { sampleTrace } from '../src/viz/trajectory';
import { Trainer } from '../src/worker/trainer';

const STEPS = 8;

function check(settings: Settings) {
  const t = new Trainer(1, settings);
  for (let e = 0; e < 3; e++) t.stepEpoch();
  const snap = t.snapshot();
  const { numQubits: Q, numLayers: L } = settings;
  const test = t.data.test;
  const featIdx = new Circuit(Q, L, test.features).featIdx;
  for (const i of [0, 7, 42]) {
    const x = test.x.subarray(i * test.features, (i + 1) * test.features);
    const tr = sampleTrace(Q, L, featIdx, snap.theta, x, STEPS);

    // States after every layer match the worker's layerVis (float32) for that sample.
    for (let l = 0; l < L; l++) {
      for (let c = 0; c < 4; c++) expect(tr.layers[l][c]).toBeCloseTo(snap.layerVis[(i * L + l) * 4 + c], 4);
      expect(tr.path[tr.layerEnd[l]]).toEqual(tr.layers[l].slice(0, 3));
    }
    // |0⟩ start, 3·steps interpolated points per layer (+1 for the CNOT jump).
    expect(tr.path[0]).toEqual([0, 0, 1]);
    expect(tr.path.length).toBe(1 + L * (3 * STEPS + (Q === 2 ? 1 : 0)));
    // Angles are w·x + b.
    const P = L * Q * 3;
    expect(tr.angles[P - 1]).toBeCloseTo(snap.theta[P - 1] * x[featIdx[P - 1]] + snap.theta[2 * P - 1], 5);
  }
}

describe('sampleTrace', () => {
  it('1 qubit: follows the Bloch vector through all layers', () => {
    check({ ...DEFAULT_SETTINGS, numQubits: 1, numLayers: 4 });
  });

  it('2 qubits: follows the Q-simplex coordinates and tangle', () => {
    check({ ...DEFAULT_SETTINGS, numQubits: 2, numLayers: 3, dataset: '4_squares' });
  });

  it('1-qubit path stays on the Bloch sphere', () => {
    const tr = sampleTrace(1, 2, new Circuit(1, 2, 2).featIdx, [0.3, -1, 2, 0.5, 1.2, -0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.4, -0.2]);
    for (const p of tr.path) expect(Math.hypot(...p)).toBeCloseTo(1, 10);
  });
});

describe('snapshot gradient', () => {
  it('is the full-batch gradient of the training loss', () => {
    const settings = { ...DEFAULT_SETTINGS, numQubits: 1 as const, numLayers: 2, regType: 'none' as const };
    const t = new Trainer(1, settings);
    const snap = t.snapshot();
    expect(snap.grad.length).toBe(snap.theta.length);
    // Central finite difference of the mean training loss for one parameter.
    const k = 3;
    const h = 1e-4;
    const theta = Float64Array.from(snap.theta);
    const lossAt = (v: number) => {
      theta[k] = v;
      t.model.setParams(theta);
      return t.model.evaluate(t.data.train).loss;
    };
    const fd = (lossAt(snap.theta[k] + h) - lossAt(snap.theta[k] - h)) / (2 * h);
    expect(snap.grad[k]).toBeCloseTo(fd, 3);
  });
});
