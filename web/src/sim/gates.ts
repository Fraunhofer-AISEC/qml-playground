/**
 * Single-qubit gate primitives on interleaved complex buffers.
 *
 * A 2x2 complex matrix is stored as 8 floats: [re00, im00, re01, im01, re10, im10, re11, im11].
 * A state vector of dimension D is stored as 2*D floats: [re0, im0, re1, im1, ...].
 */

/**
 * Rot(φ, θ, ω) = RZ(ω) RY(θ) RZ(φ), identical to PennyLane's `Rot` and to
 * `StateVecSimTorch.get_single_qubits_unitary` in the Python reference.
 */
export function rot(phi: number, theta: number, omega: number, out: Float64Array, o = 0): void {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  const p = (phi + omega) / 2;
  const m = (phi - omega) / 2;
  const cp = Math.cos(p);
  const sp = Math.sin(p);
  const cm = Math.cos(m);
  const sm = Math.sin(m);
  out[o] = cp * c; // e^{-ip} c
  out[o + 1] = -sp * c;
  out[o + 2] = -cm * s; // -e^{im} s
  out[o + 3] = -sm * s;
  out[o + 4] = cm * s; // e^{-im} s
  out[o + 5] = -sm * s;
  out[o + 6] = cp * c; // e^{ip} c
  out[o + 7] = sp * c;
}

/**
 * Partial derivatives of Rot with respect to φ, θ and ω, written to `out` at offsets 0, 8, 16.
 */
export function rotDerivs(phi: number, theta: number, omega: number, out: Float64Array): void {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  const p = (phi + omega) / 2;
  const m = (phi - omega) / 2;
  const cp = Math.cos(p);
  const sp = Math.sin(p);
  const cm = Math.cos(m);
  const sm = Math.sin(m);

  // Entries of Rot (see rot()).
  const a00r = cp * c, a00i = -sp * c;
  const a01r = -cm * s, a01i = -sm * s;
  const a10r = cm * s, a10i = -sm * s;
  const a11r = cp * c, a11i = sp * c;

  // d/dφ: m00·(-i/2), m01·(+i/2), m10·(-i/2), m11·(+i/2).  (-i/2)(a+ib) = (b/2, -a/2)
  out[0] = a00i / 2; out[1] = -a00r / 2;
  out[2] = -a01i / 2; out[3] = a01r / 2;
  out[4] = a10i / 2; out[5] = -a10r / 2;
  out[6] = -a11i / 2; out[7] = a11r / 2;

  // d/dθ: e^{-ip}(-s/2), -e^{im}(c/2), e^{-im}(c/2), e^{ip}(-s/2)
  out[8] = -cp * s / 2; out[9] = sp * s / 2;
  out[10] = -cm * c / 2; out[11] = -sm * c / 2;
  out[12] = cm * c / 2; out[13] = -sm * c / 2;
  out[14] = -cp * s / 2; out[15] = -sp * s / 2;

  // d/dω: m00·(-i/2), m01·(-i/2), m10·(+i/2), m11·(+i/2)
  out[16] = a00i / 2; out[17] = -a00r / 2;
  out[18] = a01i / 2; out[19] = -a01r / 2;
  out[20] = -a10i / 2; out[21] = a10r / 2;
  out[22] = -a11i / 2; out[23] = a11r / 2;
}

/**
 * In-place application of a 2x2 matrix `u` (at offset `uo`) to qubit `q` of an n-qubit state.
 * Qubit 0 is the most significant bit, matching the Python reference (kron(U0, U1)).
 * With `dagger` the conjugate transpose is applied instead.
 */
export function applyOneQubit(
  state: Float64Array,
  so: number,
  numQubits: number,
  q: number,
  u: Float64Array,
  uo: number,
  dagger = false,
): void {
  const dim = 1 << numQubits;
  const stride = 1 << (numQubits - 1 - q);
  const u00r = u[uo];
  let u00i = u[uo + 1], u01r = u[uo + 2], u01i = u[uo + 3];
  let u10r = u[uo + 4], u10i = u[uo + 5];
  const u11r = u[uo + 6], u11i = u[uo + 7];
  let w11i = u11i;
  if (dagger) {
    // (U†)_{ij} = conj(U_{ji})
    const t01r = u10r, t01i = -u10i;
    u10r = u01r; u10i = -u01i;
    u01r = t01r; u01i = t01i;
    u00i = -u00i;
    w11i = -u11i;
  }
  for (let i = 0; i < dim; i++) {
    if (i & stride) continue;
    const j = i | stride;
    const ai = so + 2 * i;
    const aj = so + 2 * j;
    const xr = state[ai], xi = state[ai + 1];
    const yr = state[aj], yi = state[aj + 1];
    state[ai] = u00r * xr - u00i * xi + u01r * yr - u01i * yi;
    state[ai + 1] = u00r * xi + u00i * xr + u01r * yi + u01i * yr;
    state[aj] = u10r * xr - u10i * xi + u11r * yr - w11i * yi;
    state[aj + 1] = u10r * xi + u10i * xr + u11r * yi + w11i * yr;
  }
}

/** CZ on two qubits: negates the |11⟩ amplitude. Real, diagonal and self-inverse. */
export function applyCz(state: Float64Array, so: number): void {
  state[so + 6] = -state[so + 6];
  state[so + 7] = -state[so + 7];
}

/** CNOT(control=0, target=1) on two qubits: swaps amplitudes |10⟩ and |11⟩. Self-inverse. */
export function applyCnot01(state: Float64Array, so: number): void {
  const a = so + 4;
  const b = so + 6;
  const r = state[a], i = state[a + 1];
  state[a] = state[b];
  state[a + 1] = state[b + 1];
  state[b] = r;
  state[b + 1] = i;
}
