import { describe, expect, it } from 'vitest';
import { createTaskData, type DatasetName, type TaskType } from '../src/data/datasets';
import { QuantumModel } from '../src/model/qmodel';
import { Circuit, FIXED_WEIGHT, type Architecture, type NumQubits } from '../src/sim/circuit';
import { applyOneQubit, rot } from '../src/sim/gates';
import { DEFAULT_SETTINGS } from '../src/state/settings';
import { decodeHash, encodeHash } from '../src/state/urlState';

const ARCHS: Architecture[] = [];
for (const entangler of ['cnot', 'cz', 'none'] as const) {
  for (const dataAngles of ['all', 'ry'] as const) {
    for (const weights of ['trainable', 'fixed'] as const) ARCHS.push({ entangler, dataAngles, weights });
  }
}

const CASES: { task: TaskType; dataset: DatasetName; numQubits: NumQubits }[] = [
  { task: 'classification', dataset: 'circle', numQubits: 1 },
  { task: 'classification', dataset: '3_circles', numQubits: 2 },
  { task: 'regression', dataset: 'fourier_2', numQubits: 2 },
];

describe('architecture options', () => {
  for (const c of CASES) {
    for (const arch of ARCHS) {
      if (c.numQubits === 1 && arch.entangler !== 'cnot') continue; // the entangler only exists for two qubits
      const name = `${c.task}, ${c.numQubits}q, ${arch.entangler}/${arch.dataAngles}/${arch.weights}`;
      it(`gradients match finite differences; frozen weights stay fixed (${name})`, () => {
        const model = new QuantumModel({ task: c.task, dataset: c.dataset, numQubits: c.numQubits, numLayers: 3, arch });
        model.init(3);
        const set = createTaskData(c.task, c.dataset, 1).train;
        const idx = Int32Array.from({ length: 24 }, (_, i) => i);
        model.lossAndGrad(set, idx, 0, idx.length);
        const grad = model.gradient.slice();
        const h = 1e-5;
        for (let i = 0; i < model.theta.length; i++) {
          if (!model.trainable[i]) {
            expect(grad[i]).toBe(0);
            const j = i % 3;
            const data = arch.dataAngles === 'all' || j === 1;
            expect(model.theta[i]).toBe(data ? FIXED_WEIGHT : 0);
            continue;
          }
          const v = model.theta[i];
          model.theta[i] = v + h;
          const up = model.lossAndGrad(set, idx, 0, idx.length);
          model.theta[i] = v - h;
          const down = model.lossAndGrad(set, idx, 0, idx.length);
          model.theta[i] = v;
          expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 6);
        }
        const frozen = model.trainable.reduce((n, t) => n + (t ? 0 : 1), 0);
        const P = model.P;
        expect(frozen).toBe(arch.weights === 'fixed' ? P : arch.dataAngles === 'ry' ? (2 * P) / 3 : 0);
      });
    }
  }

  it("entangler 'none' with two qubits is the product of two single-qubit evolutions", () => {
    const L = 4;
    const circuit = new Circuit(2, L, 2, 'none');
    const P = circuit.numAngles;
    const w = Float64Array.from({ length: P }, (_, k) => Math.sin(k + 1));
    const b = Float64Array.from({ length: P }, (_, k) => Math.cos(2 * k));
    const x = [0.3, -0.7];
    const states = new Float64Array(L * circuit.stateSize);
    circuit.forward(x, 0, w, b, states, 0);
    const psi = states.subarray((L - 1) * circuit.stateSize);

    const single = (q: number) => {
      const s = new Float64Array([1, 0, 0, 0]);
      const u = new Float64Array(8);
      for (let l = 0; l < L; l++) {
        const k = (l * 2 + q) * 3;
        const a = (j: number) => w[k + j] * x[circuit.featIdx[k + j]] + b[k + j];
        rot(a(0), a(1), a(2), u, 0);
        applyOneQubit(s, 0, 1, 0, u, 0);
      }
      return s;
    };
    const s0 = single(0);
    const s1 = single(1);
    for (let i = 0; i < 4; i++) {
      const [b0, b1] = [i >> 1, i & 1];
      const re = s0[2 * b0] * s1[2 * b1] - s0[2 * b0 + 1] * s1[2 * b1 + 1];
      const im = s0[2 * b0] * s1[2 * b1 + 1] + s0[2 * b0 + 1] * s1[2 * b1];
      expect(psi[2 * i]).toBeCloseTo(re, 12);
      expect(psi[2 * i + 1]).toBeCloseTo(im, 12);
    }
  });

  it('CZ negates |11⟩ relative to no entangler and leaves the other amplitudes unchanged', () => {
    const L = 2;
    const w = Float64Array.from({ length: 12 }, (_, k) => 0.3 * k - 1);
    const b = Float64Array.from({ length: 12 }, (_, k) => 0.2 * k);
    const firstLayer = (e: 'cz' | 'none') => {
      const c = new Circuit(2, L, 2, e);
      const s = new Float64Array(L * c.stateSize);
      c.forward([0.5, 0.1], 0, w, b, s, 0);
      return s.subarray(0, c.stateSize);
    };
    const cz = firstLayer('cz');
    const none = firstLayer('none');
    for (let i = 0; i < 6; i++) expect(cz[i]).toBeCloseTo(none[i], 12);
    expect(cz[6]).toBeCloseTo(-none[6], 12);
    expect(cz[7]).toBeCloseTo(-none[7], 12);
  });

  it('round-trips through the URL and keeps default links short', () => {
    expect(encodeHash(DEFAULT_SETTINGS)).not.toMatch(/ent=|da=|w=/);
    const s = { ...DEFAULT_SETTINGS, numQubits: 2 as const, entangler: 'cz' as const, dataAngles: 'ry' as const, weights: 'fixed' as const };
    expect(decodeHash(encodeHash(s)).settings).toEqual(s);
    expect(decodeHash('#v=1&ent=bogus&da=x&w=1').settings).toMatchObject({ entangler: 'cnot', dataAngles: 'all', weights: 'trainable' });
  });
});
