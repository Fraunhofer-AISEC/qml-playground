import type { Architecture, NumQubits } from '../sim/circuit';
import { fourierCurve, type RegressionDataset } from '../data/datasets';
import { QuantumModel } from '../model/qmodel';

/** Number of uniform samples of t = πx ∈ [−π, π) for the spectrum. */
export const SPECTRUM_POINTS = 512;
export const MAX_FREQUENCY = 10;

/** Uniform grid x_n = −1 + 2n/N, n = 0 … N−1 (one period of t = πx, without the duplicate endpoint). */
export function spectrumGrid(n = SPECTRUM_POINTS): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => -1 + (2 * i) / n);
}

/**
 * Magnitudes |c_k| of the complex Fourier coefficients c_k = (1/N) Σ_n f(t_n) e^{−ik t_n}, k = 0 … kMax,
 * of samples `f` on the grid t_n = −π + 2πn/N (direct DFT).
 */
export function fourierMagnitudes(f: ArrayLike<number>, kMax = MAX_FREQUENCY): number[] {
  const n = f.length;
  const out: number[] = [];
  for (let k = 0; k <= kMax; k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const t = -Math.PI + (2 * Math.PI * i) / n;
      re += f[i] * Math.cos(k * t);
      im -= f[i] * Math.sin(k * t);
    }
    out.push(Math.hypot(re, im) / n);
  }
  return out;
}

/** Spectrum of the target function of a regression data set. */
export function targetSpectrum(dataset: string): number[] {
  // fourierCurve(N + 1) samples x = −1 + 2i/N for i = 0 … N; drop the endpoint (x = 1 ≡ −1).
  const [, y] = fourierCurve(dataset, SPECTRUM_POINTS + 1);
  return fourierMagnitudes(y.subarray(0, SPECTRUM_POINTS));
}

/** Spectrum of the model output ⟨Z⟩ (without MC noise) for parameters θ. */
export function modelSpectrum(
  numQubits: NumQubits,
  numLayers: number,
  dataset: RegressionDataset,
  theta: ArrayLike<number>,
  arch?: Architecture,
): number[] {
  const model = new QuantumModel({ task: 'regression', dataset, numQubits, numLayers, arch });
  model.setParams(theta);
  return fourierMagnitudes(model.predictRegression(spectrumGrid()));
}
