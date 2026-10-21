import { applyEntangler, type Entangler, type NumQubits } from '../sim/circuit';
import { applyOneQubit, rot } from '../sim/gates';
import { blochVector, tangle } from './quantum';

/** The path of one sample through the circuit, for "follow a sample". */
export interface SampleTrace {
  /** Rotation angle w·x + b of every sub-gate (flat parameter layout k). */
  angles: Float64Array;
  /** Visual coordinates after every layer, as in `Snapshot.layerVis`. */
  layers: number[][];
  /**
   * Smooth path from |0⟩ in plot coordinates: Bloch vector (1 qubit) or (p|01⟩, p|10⟩, p|00⟩) (2 qubits).
   * Every sub-rotation is interpolated; the entangler is a jump.
   */
  path: number[][];
  /** Index into `path` of the state after each layer. */
  layerEnd: number[];
}

function probs(s: Float64Array): number[] {
  const p = (k: number) => s[2 * k] ** 2 + s[2 * k + 1] ** 2;
  return [p(1), p(2), p(0)];
}

/**
 * Re-runs one sample through the circuit with parameters θ = [weights, biases], interpolating each RZ/RY/RZ
 * sub-rotation in `steps` steps. Cheap enough to run on the main thread for every hover.
 */
export function sampleTrace(
  numQubits: NumQubits,
  numLayers: number,
  featIdx: Int32Array,
  theta: ArrayLike<number>,
  x: ArrayLike<number>,
  steps = 8,
  entangler: Entangler = 'cnot',
): SampleTrace {
  const Q = numQubits;
  const P = numLayers * Q * 3;
  const angles = new Float64Array(P);
  for (let k = 0; k < P; k++) angles[k] = theta[k] * x[featIdx[k]] + theta[P + k];

  const point = (s: Float64Array) => (Q === 1 ? blochVector(s) : probs(s));
  const psi = new Float64Array(4 << (Q - 1));
  psi[0] = 1;
  const cur = new Float64Array(psi.length);
  const u = new Float64Array(8 * Q);
  const path = [point(psi)];
  const layerEnd: number[] = [];
  const layers: number[][] = [];

  for (let l = 0; l < numLayers; l++) {
    // Rot(φ, θ, ω) = RZ(ω)·RY(θ)·RZ(φ): apply φ, then θ, then ω, each on all qubits at once.
    for (let j = 0; j < 3; j++) {
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        cur.set(psi);
        for (let q = 0; q < Q; q++) {
          const a = angles[(l * Q + q) * 3 + j] * t;
          rot(j === 0 ? a : 0, j === 1 ? a : 0, j === 2 ? a : 0, u, 8 * q);
          applyOneQubit(cur, 0, Q, q, u, 8 * q);
        }
        path.push(point(cur));
      }
      psi.set(cur);
    }
    if (Q === 2) {
      applyEntangler(entangler, psi, 0);
      path.push(point(psi));
    }
    layerEnd.push(path.length - 1);
    layers.push(Q === 1 ? [...blochVector(psi), 0] : [...probs(psi), tangle(psi)]);
  }
  return { angles, layers, path, layerEnd };
}
