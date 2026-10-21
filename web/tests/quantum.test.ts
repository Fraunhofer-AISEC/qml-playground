import { describe, expect, it } from 'vitest';
import { blochVector, reducedBloch, tangle } from '../src/viz/quantum';
import { Trainer, UNIT_GRID, UNIT_POINTS_1D } from '../src/worker/trainer';
import { DEFAULT_SETTINGS } from '../src/state/settings';
import { blochValue, classAt } from '../src/viz/unitMaps';

const s = Math.SQRT1_2;

function reduced(state: number[], numQubits: 1 | 2, q: number) {
  const out = new Float32Array(4);
  reducedBloch(state, 0, numQubits, q, out, 0);
  return Array.from(out);
}

describe('reducedBloch', () => {
  it('matches the pure single-qubit Bloch vector', () => {
    const psi = [0.6, 0.0, 0.48, 0.64]; // normalized: 0.36 + 0.2304 + 0.4096
    const [x, y, z, purity] = reduced(psi, 1, 0);
    const [bx, by, bz] = blochVector(psi);
    expect([x, y, z].map((v) => +v.toFixed(5))).toEqual([bx, by, bz].map((v) => +v.toFixed(5)));
    expect(purity).toBeCloseTo(1, 6);
  });

  it('product state |0⟩⊗|+⟩: qubit 0 at +Z, qubit 1 at +X, both pure', () => {
    const psi = [s, 0, s, 0, 0, 0, 0, 0]; // |00⟩ + |01⟩
    expect(reduced(psi, 2, 0).map((v) => +v.toFixed(5))).toEqual([0, 0, 1, 1]);
    expect(reduced(psi, 2, 1).map((v) => +v.toFixed(5))).toEqual([1, 0, 0, 1]);
    expect(tangle(psi)).toBeCloseTo(0, 10);
  });

  it('product state |1⟩⊗|0⟩ distinguishes the qubit order (qubit 0 = most significant bit)', () => {
    const psi = [0, 0, 0, 0, 1, 0, 0, 0]; // |10⟩
    expect(reduced(psi, 2, 0)[2]).toBeCloseTo(-1, 6);
    expect(reduced(psi, 2, 1)[2]).toBeCloseTo(1, 6);
  });

  it('Bell state: both reduced states maximally mixed, tangle 1', () => {
    const psi = [s, 0, 0, 0, 0, 0, s, 0]; // (|00⟩ + |11⟩)/√2
    for (const q of [0, 1]) {
      const [x, y, z, purity] = reduced(psi, 2, q);
      expect(Math.hypot(x, y, z)).toBeCloseTo(0, 6);
      expect(purity).toBeCloseTo(0.5, 6);
    }
    expect(tangle(psi)).toBeCloseTo(1, 10);
  });
});

describe('unit maps in snapshots', () => {
  it('classification, 2 qubits: grid maps per layer and qubit', () => {
    const t = new Trainer(1, { ...DEFAULT_SETTINGS, numQubits: 2, numLayers: 4, dataset: 'tricrown' });
    const u = t.snapshot().units;
    const points = UNIT_GRID * UNIT_GRID;
    expect(u.points).toBe(points);
    expect(u.bloch).toHaveLength(4 * 2 * points * 4);
    expect(u.tangle).toHaveLength(4 * points);
    expect(u.scores).toHaveLength(4 * points * 3);
    for (let p = 0; p < points; p += 37) {
      const r = Math.hypot(...[0, 1, 2].map((c) => u.bloch[((3 * 2 + 0) * points + p) * 4 + c]));
      expect(r).toBeLessThanOrEqual(1 + 1e-5);
      // purity and tangle are consistent for pure two-qubit states: tangle = 4·det(ρ_A) = 2(1 − purity)
      const purity = blochValue(u, 2, 3, 0, p, 'purity');
      expect(u.tangle[3 * points + p]).toBeCloseTo(2 * (1 - purity), 4);
      const { cls, confidence } = classAt(u, 3, 3, p);
      expect(cls).toBeGreaterThanOrEqual(0);
      expect(confidence).toBeGreaterThanOrEqual(0);
      expect(confidence).toBeLessThanOrEqual(1);
    }
  });

  it('regression, 1 qubit: 1-D maps and ⟨Z⟩ output', () => {
    const t = new Trainer(1, { ...DEFAULT_SETTINGS, task: 'regression', dataset: 'fourier_2', numLayers: 3 });
    const u = t.snapshot().units;
    expect(u.points).toBe(UNIT_POINTS_1D);
    expect(u.tangle).toHaveLength(0);
    expect(u.scores).toHaveLength(3 * UNIT_POINTS_1D);
    // the output map after the last layer equals ⟨Z⟩ of the only qubit
    for (let p = 0; p < UNIT_POINTS_1D; p += 9) {
      expect(u.scores[2 * UNIT_POINTS_1D + p]).toBeCloseTo(blochValue(u, 1, 2, 0, p, 'z'), 5);
    }
  });
});
