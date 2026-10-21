/**
 * Per-sample losses on a final state ψ (interleaved complex, dimension D).
 *
 * Each function returns the sample loss and writes into `g` the gradient
 * ∂ℓ/∂Re(ψ) + i ∂ℓ/∂Im(ψ), scaled by `scale` (usually 1/batchSize for a mean reduction).
 */

const TINY = 1e-300;

/** |⟨ψ|t⟩|² for 2-dim states. */
export function fidelity(psi: Float64Array, po: number, t: Float64Array): number {
  const ar = psi[po] * t[0] + psi[po + 1] * t[1] + psi[po + 2] * t[2] + psi[po + 3] * t[3];
  const ai = psi[po] * t[1] - psi[po + 1] * t[0] + psi[po + 2] * t[3] - psi[po + 3] * t[2];
  return ar * ar + ai * ai;
}

/** 1 − |⟨ψ|t⟩|²  (`fidelity_loss`, single qubit). ∂F/∂ψ = 2·conj(a)·t with a = ⟨ψ|t⟩. */
export function fidelityLoss(
  psi: Float64Array,
  po: number,
  t: Float64Array,
  scale: number,
  g: Float64Array,
): number {
  const ar = psi[po] * t[0] + psi[po + 1] * t[1] + psi[po + 2] * t[2] + psi[po + 3] * t[3];
  const ai = psi[po] * t[1] - psi[po + 1] * t[0] + psi[po + 2] * t[3] - psi[po + 3] * t[2];
  const f = -2 * scale;
  for (let k = 0; k < 2; k++) {
    const tr = t[2 * k];
    const ti = t[2 * k + 1];
    g[2 * k] = f * (ar * tr + ai * ti);
    g[2 * k + 1] = f * (ar * ti - ai * tr);
  }
  return 1 - (ar * ar + ai * ai);
}

/**
 * Class probabilities from basis-state probabilities, renormalized over the first C basis states
 * (`renormalize_probabilities`). Writes C values into `out`.
 */
export function classProbabilities(
  psi: Float64Array,
  po: number,
  dim: number,
  numClasses: number,
  out: Float64Array,
): void {
  let nonClass = 0;
  for (let k = numClasses; k < dim; k++) nonClass += psi[po + 2 * k] ** 2 + psi[po + 2 * k + 1] ** 2;
  const denom = numClasses === dim ? 1 : 1 - nonClass;
  for (let c = 0; c < numClasses; c++) out[c] = (psi[po + 2 * c] ** 2 + psi[po + 2 * c + 1] ** 2) / denom;
}

/**
 * Negative log-likelihood of the renormalized class probability (`cross_entropy_loss`, two qubits):
 * ℓ = −log p_y + log(1 − Σ_{k≥C} p_k)   (the second term only when C < D).
 */
export function nllLoss(
  psi: Float64Array,
  po: number,
  dim: number,
  numClasses: number,
  label: number,
  scale: number,
  g: Float64Array,
): number {
  g.fill(0, 0, 2 * dim);
  const yr = psi[po + 2 * label];
  const yi = psi[po + 2 * label + 1];
  const py = Math.max(yr * yr + yi * yi, TINY);
  let loss = -Math.log(py);
  g[2 * label] = (-2 * scale * yr) / py;
  g[2 * label + 1] = (-2 * scale * yi) / py;
  if (numClasses < dim) {
    let nonClass = 0;
    for (let k = numClasses; k < dim; k++) nonClass += psi[po + 2 * k] ** 2 + psi[po + 2 * k + 1] ** 2;
    const denom = Math.max(1 - nonClass, TINY);
    loss += Math.log(denom);
    for (let k = numClasses; k < dim; k++) {
      g[2 * k] = (-2 * scale * psi[po + 2 * k]) / denom;
      g[2 * k + 1] = (-2 * scale * psi[po + 2 * k + 1]) / denom;
    }
  }
  return loss;
}

/** ⟨Z⟩ of qubit 0 (most significant bit): Σ_{k<D/2} |ψ_k|² − Σ_{k≥D/2} |ψ_k|². */
export function expvalZ0(psi: Float64Array, po: number, dim: number): number {
  const half = dim / 2;
  let z = 0;
  for (let k = 0; k < dim; k++) {
    const p = psi[po + 2 * k] ** 2 + psi[po + 2 * k + 1] ** 2;
    z += k < half ? p : -p;
  }
  return z;
}

/** Squared error (⟨Z⟩ − y)² (`mse_loss`). */
export function mseZLoss(
  psi: Float64Array,
  po: number,
  dim: number,
  target: number,
  scale: number,
  g: Float64Array,
): number {
  const z = expvalZ0(psi, po, dim);
  const diff = z - target;
  const half = dim / 2;
  for (let k = 0; k < dim; k++) {
    const f = 4 * scale * diff * (k < half ? 1 : -1);
    g[2 * k] = f * psi[po + 2 * k];
    g[2 * k + 1] = f * psi[po + 2 * k + 1];
  }
  return diff * diff;
}
