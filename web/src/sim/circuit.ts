import { applyCnot01, applyCz, applyOneQubit, rot, rotDerivs } from './gates';

export type NumQubits = 1 | 2;

/** Two-qubit gate at the end of every layer. */
export type Entangler = 'cnot' | 'cz' | 'none';
/** Which rotation angles upload the data: all three (w·x + b), or only RY's θ (φ and ω are pure biases). */
export type DataAngles = 'all' | 'ry';
/** Trainable data weights w, or fixed w = π (x ∈ [−1, 1] sweeps a full turn; integer frequencies in t = πx). */
export type WeightMode = 'trainable' | 'fixed';

export interface Architecture {
  entangler: Entangler;
  dataAngles: DataAngles;
  weights: WeightMode;
}

/** The architecture of the original model (and of the PyTorch reference). */
export const DEFAULT_ARCHITECTURE: Architecture = { entangler: 'cnot', dataAngles: 'all', weights: 'trainable' };
export const FIXED_WEIGHT = Math.PI;

/** Whether sub-gate j ∈ {φ, θ, ω} of a Rot uploads the data. */
export function isDataAngle(j: number, dataAngles: DataAngles): boolean {
  return dataAngles === 'all' || j === 1;
}

/** Applies the entangling gate of a layer (all choices are real and self-inverse). */
export function applyEntangler(entangler: Entangler, state: Float64Array, so: number): void {
  if (entangler === 'cnot') applyCnot01(state, so);
  else if (entangler === 'cz') applyCz(state, so);
}

/**
 * Data re-uploading circuit on 1 or 2 qubits.
 *
 * Parameters are flat arrays `weights` and `biases` of length P = layers * qubits * 3, indexed as
 * k = (layer * qubits + qubit) * 3 + j with j ∈ {φ, θ, ω}. Each angle is a_k = w_k * x[f_k] + b_k where the
 * feature index f_k reproduces how the PyTorch reference repeats the input features over all angles:
 *  - 2 qubits (`StateVecSimTorch.get_angles`): f_k = k mod F
 *  - 1 qubit  (`SingleQubitReuploadingTorch.forward`, reshaped as (3, layers)): f_k = (j * layers + layer) mod F
 *
 * Each layer applies Rot on every qubit, followed by the entangler (default CNOT(0→1)) for two qubits.
 */
export class Circuit {
  readonly numQubits: NumQubits;
  readonly numLayers: number;
  readonly numFeatures: number;
  readonly dim: number;
  /** Floats per state vector (interleaved re/im). */
  readonly stateSize: number;
  readonly numAngles: number;
  readonly featIdx: Int32Array;
  readonly entangler: Entangler;

  private readonly u: Float64Array; // one Rot matrix per qubit
  private readonly du: Float64Array; // 3 derivative matrices per qubit
  private readonly angles: Float64Array;
  private readonly tmp: Float64Array;
  private readonly base: Float64Array;
  private readonly psi: Float64Array;

  constructor(numQubits: NumQubits, numLayers: number, numFeatures: number, entangler: Entangler = 'cnot') {
    this.numQubits = numQubits;
    this.entangler = numQubits === 2 ? entangler : 'none';
    this.numLayers = numLayers;
    this.numFeatures = numFeatures;
    this.dim = 1 << numQubits;
    this.stateSize = 2 * this.dim;
    this.numAngles = numLayers * numQubits * 3;
    this.featIdx = new Int32Array(this.numAngles);
    for (let l = 0; l < numLayers; l++) {
      for (let q = 0; q < numQubits; q++) {
        for (let j = 0; j < 3; j++) {
          const k = (l * numQubits + q) * 3 + j;
          this.featIdx[k] = numQubits === 1 ? (j * numLayers + l) % numFeatures : k % numFeatures;
        }
      }
    }
    this.u = new Float64Array(8 * numQubits);
    this.du = new Float64Array(24 * numQubits);
    this.angles = new Float64Array(this.numAngles);
    this.tmp = new Float64Array(this.stateSize);
    this.base = new Float64Array(this.stateSize);
    this.psi = new Float64Array(this.stateSize);
  }

  private computeAngles(x: ArrayLike<number>, xo: number, w: ArrayLike<number>, b: ArrayLike<number>): void {
    const { angles, featIdx } = this;
    for (let k = 0; k < this.numAngles; k++) angles[k] = w[k] * x[xo + featIdx[k]] + b[k];
  }

  private layerUnitaries(l: number): void {
    const a = this.angles;
    for (let q = 0; q < this.numQubits; q++) {
      const k = (l * this.numQubits + q) * 3;
      rot(a[k], a[k + 1], a[k + 2], this.u, 8 * q);
    }
  }

  /**
   * Runs the circuit for one sample. Writes the state after every layer to `statesOut`
   * starting at `so`, i.e. layer l occupies [so + l*stateSize, so + (l+1)*stateSize).
   */
  forward(
    x: ArrayLike<number>,
    xo: number,
    w: ArrayLike<number>,
    b: ArrayLike<number>,
    statesOut: Float64Array,
    so: number,
  ): void {
    const { stateSize, numQubits, psi } = this;
    this.computeAngles(x, xo, w, b);
    psi.fill(0);
    psi[0] = 1;
    for (let l = 0; l < this.numLayers; l++) {
      this.layerUnitaries(l);
      for (let q = 0; q < numQubits; q++) applyOneQubit(psi, 0, numQubits, q, this.u, 8 * q);
      applyEntangler(this.entangler, psi, 0);
      statesOut.set(psi, so + l * stateSize);
    }
  }

  /**
   * Adjoint-method backward pass for one sample.
   *
   * `g` holds ∂L/∂Re(ψ_out) + i ∂L/∂Im(ψ_out) for the final state and is consumed (overwritten).
   * For every angle a: ∂L/∂a = Re⟨g_l | ∂U_l/∂a | ψ_{l-1}⟩, then g_{l-1} = U_l† g_l.
   * Gradients are accumulated into gw / gb.
   */
  backward(
    x: ArrayLike<number>,
    xo: number,
    w: ArrayLike<number>,
    b: ArrayLike<number>,
    states: Float64Array,
    so: number,
    g: Float64Array,
    gw: Float64Array,
    gb: Float64Array,
  ): void {
    const { stateSize, numQubits, tmp, base, psi, du, u, featIdx } = this;
    this.computeAngles(x, xo, w, b);
    const a = this.angles;

    for (let l = this.numLayers - 1; l >= 0; l--) {
      applyEntangler(this.entangler, g, 0);

      // ψ_{l-1}: state entering this layer
      if (l === 0) {
        psi.fill(0);
        psi[0] = 1;
      } else {
        psi.set(states.subarray(so + (l - 1) * stateSize, so + l * stateSize));
      }

      this.layerUnitaries(l);
      for (let q = 0; q < numQubits; q++) {
        const k0 = (l * numQubits + q) * 3;
        rotDerivs(a[k0], a[k0 + 1], a[k0 + 2], du.subarray(24 * q, 24 * q + 24));
      }

      for (let q = 0; q < numQubits; q++) {
        // base = (⊗_{r≠q} U_r) ψ_{l-1}
        base.set(psi);
        for (let r = 0; r < numQubits; r++) {
          if (r !== q) applyOneQubit(base, 0, numQubits, r, u, 8 * r);
        }
        for (let j = 0; j < 3; j++) {
          tmp.set(base);
          applyOneQubit(tmp, 0, numQubits, q, du, 24 * q + 8 * j);
          let dLda = 0;
          for (let i = 0; i < stateSize; i++) dLda += g[i] * tmp[i];
          const k = (l * numQubits + q) * 3 + j;
          gw[k] += dLda * x[xo + featIdx[k]];
          gb[k] += dLda;
        }
      }

      // g ← U_l† g (rotation part; the self-inverse entangler is handled above)
      for (let q = 0; q < numQubits; q++) applyOneQubit(g, 0, numQubits, q, u, 8 * q, true);
    }
  }
}
