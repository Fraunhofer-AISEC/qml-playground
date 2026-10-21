import { describe, expect, it } from 'vitest';
import { decisionBoundaries, isoSegments } from '../src/viz/contour';

const grid = (n: number, f: (x: number, y: number) => number) => {
  const z = new Float64Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) z[r * n + c] = f(-1 + (2 * c) / (n - 1), -1 + (2 * r) / (n - 1));
  return z;
};

describe('marching squares', () => {
  it('traces a circle at its radius', () => {
    const n = 41;
    const r = 0.6;
    const segs = isoSegments(grid(n, (x, y) => Math.hypot(x, y) - r), n);
    expect(segs.length).toBeGreaterThan(40);
    const cell = 2 / (n - 1);
    for (const [x0, y0, x1, y1] of segs) {
      expect(Math.abs(Math.hypot(x0, y0) - r)).toBeLessThan(cell);
      expect(Math.abs(Math.hypot(x1, y1) - r)).toBeLessThan(cell);
    }
  });

  it('traces a straight line exactly', () => {
    const n = 15;
    const segs = isoSegments(grid(n, (x, y) => x + 2 * y - 0.3), n);
    expect(segs.length).toBeGreaterThan(0);
    for (const [x0, y0, x1, y1] of segs) {
      expect(x0 + 2 * y0).toBeCloseTo(0.3, 9);
      expect(x1 + 2 * y1).toBeCloseTo(0.3, 9);
    }
  });

  it('finds nothing in a constant field', () => {
    expect(isoSegments(new Float64Array(25).fill(1), 5)).toHaveLength(0);
  });

  it('two-class boundary is the p = 0.5 line', () => {
    const n = 21;
    const p0 = grid(n, (x) => 0.5 + 0.4 * x);
    const scores = new Float64Array(n * n * 2);
    for (let i = 0; i < n * n; i++) {
      scores[2 * i] = p0[i];
      scores[2 * i + 1] = 1 - p0[i];
    }
    const segs = decisionBoundaries(scores, n, 2);
    expect(segs.length).toBe(n - 1);
    for (const s of segs) {
      expect(s[0]).toBeCloseTo(0, 9);
      expect(s[2]).toBeCloseTo(0, 9);
    }
  });
});
