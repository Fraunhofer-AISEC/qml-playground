/**
 * Marching squares: the `level` iso-line of a field sampled on an n×n grid over [-1, 1]² (row-major, y outer),
 * returned as line segments [x0, y0, x1, y1] in data coordinates. Ambiguous saddle cells are resolved with the
 * cell-center average.
 */
export function isoSegments(field: ArrayLike<number>, n: number, level = 0): number[][] {
  const segments: number[][] = [];
  const coord = (i: number) => -1 + (2 * i) / (n - 1);
  const at = (r: number, c: number) => field[r * n + c] - level;
  // Point on the edge between grid points (r0, c0) and (r1, c1), where the field crosses the level.
  const cross = (r0: number, c0: number, r1: number, c1: number): [number, number] => {
    const a = at(r0, c0);
    const b = at(r1, c1);
    const t = a === b ? 0.5 : a / (a - b);
    return [coord(c0 + (c1 - c0) * t), coord(r0 + (r1 - r0) * t)];
  };

  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      // Corners counter-clockwise from bottom-left: bl, br, tr, tl.
      const bl = at(r, c) > 0 ? 1 : 0;
      const br = at(r, c + 1) > 0 ? 2 : 0;
      const tr = at(r + 1, c + 1) > 0 ? 4 : 0;
      const tl = at(r + 1, c) > 0 ? 8 : 0;
      const code = bl | br | tr | tl;
      if (code === 0 || code === 15) continue;
      const bottom = () => cross(r, c, r, c + 1);
      const right = () => cross(r, c + 1, r + 1, c + 1);
      const top = () => cross(r + 1, c, r + 1, c + 1);
      const left = () => cross(r, c, r + 1, c);
      const seg = (p: [number, number], q: [number, number]) => segments.push([p[0], p[1], q[0], q[1]]);
      switch (code) {
        case 1:
        case 14:
          seg(left(), bottom());
          break;
        case 2:
        case 13:
          seg(bottom(), right());
          break;
        case 3:
        case 12:
          seg(left(), right());
          break;
        case 4:
        case 11:
          seg(right(), top());
          break;
        case 6:
        case 9:
          seg(bottom(), top());
          break;
        case 7:
        case 8:
          seg(left(), top());
          break;
        case 5:
        case 10: {
          // Saddle: bl and tr on one side, br and tl on the other.
          const center = (at(r, c) + at(r, c + 1) + at(r + 1, c) + at(r + 1, c + 1)) / 4;
          const blHigh = code === 5;
          if (center > 0 === blHigh) {
            seg(left(), top());
            seg(bottom(), right());
          } else {
            seg(left(), bottom());
            seg(right(), top());
          }
          break;
        }
      }
    }
  }
  return segments;
}

/**
 * Decision boundaries of class scores on an n×n grid ([point][class]): the zero line of each class's margin
 * s_c − max_{c'≠c} s_c'. With two classes this is the p = 0.5 line; with more it is the argmax boundary.
 */
export function decisionBoundaries(scores: ArrayLike<number>, n: number, numClasses: number): number[][] {
  const margin = new Float64Array(n * n);
  const segments: number[][] = [];
  // Two classes share one boundary; draw it once.
  const classes = numClasses === 2 ? 1 : numClasses;
  for (let k = 0; k < classes; k++) {
    for (let p = 0; p < n * n; p++) {
      let other = -Infinity;
      for (let c = 0; c < numClasses; c++) if (c !== k) other = Math.max(other, scores[p * numClasses + c]);
      margin[p] = scores[p * numClasses + k] - other;
    }
    segments.push(...isoSegments(margin, n, 0));
  }
  return segments;
}
