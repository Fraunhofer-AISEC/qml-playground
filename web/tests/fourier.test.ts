import { describe, expect, it } from 'vitest';
import { fourierMagnitudes, modelSpectrum, spectrumGrid, targetSpectrum, SPECTRUM_POINTS } from '../src/viz/fourier';
import { QuantumModel } from '../src/model/qmodel';

describe('fourierMagnitudes', () => {
  it('finds pure frequencies: 0.2 + cos(3t) − 0.4·sin(5t)', () => {
    const f = Array.from(spectrumGrid(), (x) => 0.2 + Math.cos(3 * Math.PI * x) - 0.4 * Math.sin(5 * Math.PI * x));
    const c = fourierMagnitudes(f, 6);
    expect(c[0]).toBeCloseTo(0.2, 10);
    expect(c[3]).toBeCloseTo(0.5, 10);
    expect(c[5]).toBeCloseTo(0.2, 10);
    for (const k of [1, 2, 4, 6]) expect(c[k]).toBeCloseTo(0, 10);
  });
});

describe('targetSpectrum', () => {
  it('has exactly the degrees of the Fourier data set', () => {
    const c = targetSpectrum('fourier_2');
    expect(c[1]).toBeCloseTo(c[2], 10);
    expect(c[1] / c[0]).toBeCloseTo((0.3 * Math.SQRT2) / 2 / 0.1, 6);
    for (let k = 3; k <= 10; k++) expect(c[k]).toBeCloseTo(0, 8);
  });
});

describe('modelSpectrum', () => {
  it('is the DFT of the model output on the uniform grid', () => {
    const model = new QuantumModel({ task: 'regression', dataset: 'fourier_1', numQubits: 1, numLayers: 3 });
    model.init(7);
    const direct = fourierMagnitudes(model.predictRegression(spectrumGrid()));
    expect(modelSpectrum(1, 3, 'fourier_1', model.theta)).toEqual(direct);
    expect(spectrumGrid().length).toBe(SPECTRUM_POINTS);
  });
});
