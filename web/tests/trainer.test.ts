import { describe, expect, it } from 'vitest';
import { Trainer } from '../src/worker/trainer';
import { DEFAULT_SETTINGS, type Settings } from '../src/state/settings';

const settings: Settings = { ...DEFAULT_SETTINGS, numQubits: 2, numLayers: 3, speed: 0 };

describe('Trainer', () => {
  it('streams metrics incrementally, starting with epoch 0', () => {
    const t = new Trainer(1, settings);
    const s0 = t.snapshot();
    expect(s0.epoch).toBe(0);
    expect(s0.metrics).toMatchObject({ base: 0, from: 0 });
    expect(s0.metrics.loss).toHaveLength(1);
    t.stepEpoch();
    t.stepEpoch();
    const s1 = t.snapshot();
    expect(s1.metrics.from).toBe(1);
    expect(s1.metrics.loss).toHaveLength(2);
    expect(s1.latestEpoch).toBe(2);
    expect(s1.layerVis).toHaveLength(300 * 3 * 4);
    const C = t.dataInfo().numClasses;
    expect(s1.decision).toHaveLength(settings.gridSize ** 2 * C);
    // Two qubits: renormalized class probabilities sum to 1 at every grid point.
    for (let i = 0; i < 5; i++) {
      let sum = 0;
      for (let c = 0; c < C; c++) sum += s1.decision![i * C + c];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('seek shows historic parameters without changing the latest state', () => {
    const t = new Trainer(1, settings);
    const theta0 = t.snapshot().theta;
    for (let i = 0; i < 5; i++) t.stepEpoch();
    const latest = t.snapshot().theta;
    t.seek(0);
    const past = t.snapshot();
    expect(past.epoch).toBe(0);
    expect(Array.from(past.theta)).toEqual(Array.from(theta0));
    t.seek(5);
    expect(Array.from(t.snapshot().theta)).toEqual(Array.from(latest));
  });

  it('resumeFrom truncates history and reproduces the same continuation', () => {
    const a = new Trainer(1, settings);
    for (let i = 0; i < 4; i++) a.stepEpoch();
    a.snapshot();
    a.resumeFrom(2);
    const snap = a.snapshot();
    expect(snap.latestEpoch).toBe(2);
    expect(snap.metrics.from).toBe(3); // nothing new after truncation
    a.stepEpoch();
    const after = a.snapshot();
    expect(after.metrics.from).toBe(3);
    expect(after.latestEpoch).toBe(3);
  });

  it('restores shared parameters', () => {
    const a = new Trainer(1, settings);
    for (let i = 0; i < 3; i++) a.stepEpoch();
    const theta = a.snapshot().theta;
    const b = new Trainer(2, settings, { theta, epoch: 3 });
    const s = b.snapshot();
    expect(s.epoch).toBe(3);
    expect(s.metrics.base).toBe(3);
    expect(Array.from(s.theta)).toEqual(Array.from(theta));
  });

  it('regression snapshots include the MC curve', () => {
    const t = new Trainer(1, { ...settings, task: 'regression', dataset: 'fourier_2', noiseSigma: 0.1, mcRuns: 5 });
    const s = t.snapshot();
    expect(s.curve?.mean).toHaveLength(400);
    expect(Math.max(...s.curve!.std)).toBeGreaterThan(0);
    expect(s.decision).toBeUndefined();
  });
});

describe('per-layer diagnostics', () => {
  it('records gradient norms (NaN before training) and, for two qubits, the mean tangle per layer', () => {
    const settings: Settings = { ...DEFAULT_SETTINGS, numQubits: 2, numLayers: 3, dataset: '4_squares' };
    const t = new Trainer(1, settings);
    t.stepEpoch();
    t.stepEpoch();
    const m = t.snapshot().metrics;
    expect(m.gradNorm).toHaveLength(3);
    expect(m.gradNorm[0].every(Number.isNaN)).toBe(true);
    for (const row of m.gradNorm.slice(1)) {
      expect(row).toHaveLength(3);
      for (const v of row) expect(v).toBeGreaterThan(0);
    }
    expect(m.tangle).toHaveLength(3);
    for (const row of m.tangle) {
      expect(row).toHaveLength(3);
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1 + 1e-9);
      }
    }
  });

  it('layer gradient norms partition the full gradient', () => {
    const t = new Trainer(1, { ...DEFAULT_SETTINGS, numQubits: 1, numLayers: 4, regType: 'none' });
    const set = t.data.train;
    const idx = Int32Array.from({ length: set.n }, (_, i) => i);
    t.model.lossAndGrad(set, idx, 0, set.n);
    const total = Math.hypot(...t.model.gradient);
    expect(Math.hypot(...t.model.layerGradNorms)).toBeCloseTo(total, 10);
  });

  it('truncates the diagnostics on resume', () => {
    const t = new Trainer(1, { ...DEFAULT_SETTINGS, numQubits: 1, numLayers: 2 });
    for (let e = 0; e < 4; e++) t.stepEpoch();
    t.resumeFrom(2);
    t.resendMetrics();
    const m = t.snapshot().metrics;
    expect(m.gradNorm).toHaveLength(3);
    expect(m.tangle).toHaveLength(3);
    expect(m.tangle[0]).toEqual([]);
  });
});
