/** Geometric representations of quantum states (port of `app/utils/qstate_representations.py`). */

/** Bloch vector (x, y, z) of a pure single-qubit state [re0, im0, re1, im1]. */
export function blochVector(s: ArrayLike<number>, o = 0): [number, number, number] {
  const ar = s[o], ai = s[o + 1], br = s[o + 2], bi = s[o + 3];
  return [2 * (ar * br + ai * bi), 2 * (ar * bi - ai * br), ar * ar + ai * ai - br * br - bi * bi];
}

/** Tangle (squared concurrence) of a pure two-qubit state: (2|a00·a11 − a01·a10|)². */
export function tangle(s: ArrayLike<number>, o = 0): number {
  const re = s[o] * s[o + 6] - s[o + 1] * s[o + 7] - (s[o + 2] * s[o + 4] - s[o + 3] * s[o + 5]);
  const im = s[o] * s[o + 7] + s[o + 1] * s[o + 6] - (s[o + 2] * s[o + 5] + s[o + 3] * s[o + 4]);
  return 4 * (re * re + im * im);
}

/**
 * Bloch vector of qubit `q` of a pure 1- or 2-qubit state, i.e. of its reduced density matrix
 * ρ_q = Tr_other |ψ⟩⟨ψ|. Writes (x, y, z, purity) to `out` at `oo`; purity = (1 + |r|²)/2 is 1 for a
 * product state and 1/2 when the qubit is maximally entangled with the other one.
 */
export function reducedBloch(
  s: ArrayLike<number>,
  so: number,
  numQubits: 1 | 2,
  q: number,
  out: Float32Array,
  oo: number,
): void {
  let r00 = 0, r11 = 0, re10 = 0, im10 = 0;
  if (numQubits === 1) {
    // ρ10 = b·conj(a)
    const ar = s[so], ai = s[so + 1], br = s[so + 2], bi = s[so + 3];
    r00 = ar * ar + ai * ai;
    r11 = br * br + bi * bi;
    re10 = br * ar + bi * ai;
    im10 = bi * ar - br * ai;
  } else {
    // Basis index = 2·b0 + b1 (qubit 0 is the most significant bit). For qubit q, pair the amplitudes that
    // differ only in bit q: ρ10 = Σ_other a(bit q = 1)·conj(a(bit q = 0)).
    const stride = q === 0 ? 2 : 1;
    for (const base of q === 0 ? [0, 1] : [0, 2]) {
      const i0 = so + 2 * base;
      const i1 = so + 2 * (base + stride);
      const ar = s[i0], ai = s[i0 + 1], br = s[i1], bi = s[i1 + 1];
      r00 += ar * ar + ai * ai;
      r11 += br * br + bi * bi;
      re10 += br * ar + bi * ai;
      im10 += bi * ar - br * ai;
    }
  }
  const x = 2 * re10, y = 2 * im10, z = r00 - r11;
  out[oo] = x;
  out[oo + 1] = y;
  out[oo + 2] = z;
  out[oo + 3] = (1 + x * x + y * y + z * z) / 2;
}

/**
 * Converts states (sample-major, `layers` states of `stateSize` floats per sample) into 4 visual floats each:
 * one qubit → Bloch (x, y, z, 0); two qubits → Q-simplex (p01, p10, p00, tangle).
 */
export function visualCoordinates(
  states: Float64Array,
  numSamples: number,
  layers: number,
  numQubits: 1 | 2,
): Float32Array {
  const ss = 2 << numQubits;
  const out = new Float32Array(numSamples * layers * 4);
  for (let i = 0; i < numSamples * layers; i++) {
    const so = i * ss;
    if (numQubits === 1) {
      const [x, y, z] = blochVector(states, so);
      out[4 * i] = x;
      out[4 * i + 1] = y;
      out[4 * i + 2] = z;
    } else {
      const p = (k: number) => states[so + 2 * k] ** 2 + states[so + 2 * k + 1] ** 2;
      out[4 * i] = p(1);
      out[4 * i + 1] = p(2);
      out[4 * i + 2] = p(0);
      out[4 * i + 3] = tangle(states, so);
    }
  }
  return out;
}
