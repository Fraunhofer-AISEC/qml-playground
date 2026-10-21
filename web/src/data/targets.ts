import type { ClassificationDataset } from './datasets';

/** Number of classes per classification dataset. */
export function numClasses(name: ClassificationDataset): number {
  switch (name) {
    case 'circle':
    case 'square':
    case 'crown':
      return 2;
    case 'tricrown':
      return 3;
    case '4_squares':
    case 'wavy_lines':
    case '3_circles':
      return 4;
  }
}

/**
 * Single-qubit target states per class (port of `create_target(name, num_qubits=1)`), as interleaved
 * complex 2-vectors [re0, im0, re1, im1]. They are maximally separated on the Bloch sphere.
 * Two-qubit classification uses the computational basis states |00⟩..|11⟩ instead, so classes map to
 * basis indices 0..C-1.
 */
export function singleQubitTargets(name: ClassificationDataset): Float64Array[] {
  const c = numClasses(name);
  if (c === 2) return [Float64Array.of(1, 0, 0, 0), Float64Array.of(0, 0, 1, 0)];
  if (c === 3) {
    const a = Math.cos(Math.PI / 3);
    const b = Math.sin(Math.PI / 3);
    return [Float64Array.of(1, 0, 0, 0), Float64Array.of(a, 0, b, 0), Float64Array.of(a, 0, -b, 0)];
  }
  const a = 1 / Math.sqrt(3);
  const b = Math.sqrt(2 / 3);
  const ph = (2 * Math.PI) / 3;
  return [
    Float64Array.of(1, 0, 0, 0),
    Float64Array.of(a, 0, b, 0),
    Float64Array.of(a, 0, b * Math.cos(ph), b * Math.sin(ph)),
    Float64Array.of(a, 0, b * Math.cos(-ph), b * Math.sin(-ph)),
  ];
}
