import { describe, expect, it } from 'vitest';
import { QuantumModel, type ModelConfig } from '../src/model/qmodel';
import { createTaskData } from '../src/data/datasets';
import { Rng } from '../src/data/rng';
import { applyOneQubit, rot, rotDerivs } from '../src/sim/gates';
import type { RegType } from '../src/model/regularization';

const configs: ModelConfig[] = [
  { task: 'classification', dataset: 'tricrown', numQubits: 1, numLayers: 15 },
  { task: 'classification', dataset: 'circle', numQubits: 2, numLayers: 15 },
  { task: 'classification', dataset: '4_squares', numQubits: 2, numLayers: 7 },
  { task: 'regression', dataset: 'fourier_4', numQubits: 1, numLayers: 9 },
  { task: 'regression', dataset: 'fourier_5', numQubits: 2, numLayers: 15 },
];

describe('gates', () => {
  it('Rot is unitary and its derivatives match finite differences', () => {
    const rng = new Rng(3);
    for (let t = 0; t < 20; t++) {
      const a = [0, 1, 2].map(() => (rng.next() - 0.5) * 10);
      const u = new Float64Array(8);
      rot(a[0], a[1], a[2], u);
      // U†U = I applied to basis vectors
      for (const basis of [Float64Array.of(1, 0, 0, 0), Float64Array.of(0, 0, 1, 0)]) {
        const v = basis.slice();
        applyOneQubit(v, 0, 1, 0, u, 0);
        applyOneQubit(v, 0, 1, 0, u, 0, true);
        v.forEach((x, i) => expect(x).toBeCloseTo(basis[i], 12));
      }
      const d = new Float64Array(24);
      rotDerivs(a[0], a[1], a[2], d);
      const h = 1e-6;
      for (let j = 0; j < 3; j++) {
        const ap = [...a], am = [...a];
        ap[j] += h;
        am[j] -= h;
        const up = new Float64Array(8), um = new Float64Array(8);
        rot(ap[0], ap[1], ap[2], up);
        rot(am[0], am[1], am[2], um);
        for (let e = 0; e < 8; e++) expect(d[8 * j + e]).toBeCloseTo((up[e] - um[e]) / (2 * h), 7);
      }
    }
  });
});

describe.each(configs.map((c) => [`${c.numQubits}q ${c.task} ${c.dataset} L=${c.numLayers}`, c] as const))(
  'model %s',
  (_name, config) => {
    const data = createTaskData(config.task, config.dataset);
    const set = { ...data.train, n: 16 };

    it('states stay normalized', () => {
      const model = new QuantumModel(config);
      model.init(11);
      const states = model.forwardAll(set);
      const ss = model.stateSize;
      for (let off = 0; off < states.length; off += ss) {
        let norm = 0;
        for (let i = 0; i < ss; i++) norm += states[off + i] ** 2;
        expect(norm).toBeCloseTo(1, 12);
      }
    });

    it.each(['none', 'l1', 'l2'] as RegType[])('adjoint gradient matches finite differences (reg=%s)', (regType) => {
      const model = new QuantumModel(config);
      model.init(5);
      const idx = Int32Array.from({ length: set.n }, (_, i) => i);
      const opts = { lr: 0, batchSize: set.n, regType, regStrength: 0.03 };
      model.lossAndGrad(set, idx, 0, set.n, opts);
      const grad = model.gradient.slice();
      const h = 1e-6;
      for (let p = 0; p < model.theta.length; p++) {
        const orig = model.theta[p];
        model.theta[p] = orig + h;
        const lp = model.lossAndGrad(set, idx, 0, set.n, opts);
        model.theta[p] = orig - h;
        const lm = model.lossAndGrad(set, idx, 0, set.n, opts);
        model.theta[p] = orig;
        expect(grad[p]).toBeCloseTo((lp - lm) / (2 * h), 5);
      }
    });

    it('training reduces the loss', () => {
      const model = new QuantumModel(config);
      model.init(1);
      const rng = new Rng(2);
      const before = model.evaluate(data.train).loss;
      for (let e = 0; e < 10; e++) model.trainEpoch(data.train, { lr: 0.05, batchSize: 32, regType: 'none', regStrength: 0 }, rng);
      expect(model.evaluate(data.train).loss).toBeLessThan(before);
      expect(model.epoch).toBe(10);
    });
  },
);

describe('training edge cases', () => {
  it('a batch size larger than the training set still trains (clamped)', () => {
    const data = createTaskData('regression', 'fourier_1');
    const model = new QuantumModel({ task: 'regression', dataset: 'fourier_1', numQubits: 1, numLayers: 3 });
    model.init(0);
    const before = model.theta.slice();
    model.trainEpoch(data.train, { lr: 0.1, batchSize: 128, regType: 'none', regStrength: 0 }, new Rng(0));
    expect(Array.from(model.theta)).not.toEqual(Array.from(before));
  });

  it('a batch size of 1 works for a single qubit', () => {
    const data = createTaskData('classification', 'circle');
    const model = new QuantumModel({ task: 'classification', dataset: 'circle', numQubits: 1, numLayers: 2 });
    model.init(0);
    const loss = model.trainEpoch(data.train, { lr: 0.01, batchSize: 1, regType: 'none', regStrength: 0 }, new Rng(0));
    expect(Number.isFinite(loss)).toBe(true);
  });

  it('MC uncertainty: zero noise gives zero std, positive noise gives positive std', () => {
    const model = new QuantumModel({ task: 'regression', dataset: 'fourier_2', numQubits: 2, numLayers: 4 });
    model.init(0);
    const x = Float64Array.from({ length: 50 }, (_, i) => -1 + (2 * i) / 49);
    const exact = model.predictMc(x, 0, 10, new Rng(1));
    expect(Math.max(...exact.std)).toBe(0);
    const noisy = model.predictMc(x, 0.2, 20, new Rng(1));
    expect(Math.min(...noisy.std)).toBeGreaterThan(0);
    // parameters are restored afterwards
    expect(Array.from(model.predictRegression(x))).toEqual(Array.from(exact.mean));
  });
});

describe('performance', () => {
  it('2 qubits, 15 layers, 500 samples: an epoch takes well under 30 ms', () => {
    const data = createTaskData('classification', 'wavy_lines');
    const model = new QuantumModel({ task: 'classification', dataset: 'wavy_lines', numQubits: 2, numLayers: 15 });
    model.init(0);
    const rng = new Rng(0);
    const opts = { lr: 0.01, batchSize: 32, regType: 'l2' as const, regStrength: 0.01 };
    model.trainEpoch(data.train, opts, rng); // warm-up
    const runs = 20;
    const t0 = performance.now();
    for (let i = 0; i < runs; i++) model.trainEpoch(data.train, opts, rng);
    const ms = (performance.now() - t0) / runs;
    console.log(`epoch: ${ms.toFixed(2)} ms`);
    expect(ms).toBeLessThan(30);
  });
});
