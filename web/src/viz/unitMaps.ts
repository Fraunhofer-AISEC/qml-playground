import type { UnitMaps } from '../worker/protocol';
import { CLASS_COLORS, type PlotTheme } from './theme';

/** What a qubit's unit map shows. */
export type UnitMode = 'z' | 'x' | 'y' | 'purity' | 'score';

export const UNIT_MODES: { value: UnitMode; label: string; description: string }[] = [
  { value: 'z', label: '⟨Z⟩', description: 'Expectation of Pauli Z on this qubit (+1 = |0⟩, −1 = |1⟩).' },
  { value: 'x', label: '⟨X⟩', description: 'Expectation of Pauli X on this qubit (+1 = |+⟩, −1 = |−⟩).' },
  { value: 'y', label: '⟨Y⟩', description: 'Expectation of Pauli Y on this qubit.' },
  {
    value: 'purity',
    label: 'Purity',
    description: 'Purity of the qubit’s reduced state: 1 = not entangled with the other qubit, 0.5 = maximally entangled.',
  },
  {
    value: 'score',
    label: 'Class',
    description: 'The classifier’s prediction if you measured right after this layer: class color, faded where it is unsure.',
  },
];

/** Value range per map kind: diverging maps live in [-1, 1], sequential ones in [0, 1]. */
export function isDiverging(mode: UnitMode | 'tangle'): boolean {
  return mode === 'z' || mode === 'x' || mode === 'y';
}

const COMPONENT: Record<'x' | 'y' | 'z' | 'purity', number> = { x: 0, y: 1, z: 2, purity: 3 };

/** Bloch component / purity of qubit q after layer l at grid point p. */
export function blochValue(u: UnitMaps, numQubits: number, l: number, q: number, p: number, mode: 'x' | 'y' | 'z' | 'purity') {
  return u.bloch[((l * numQubits + q) * u.points + p) * 4 + COMPONENT[mode]];
}

export function tangleValue(u: UnitMaps, l: number, p: number) {
  return u.tangle[l * u.points + p];
}

/** Predicted class and confidence ∈ [0, 1] from numClasses scores at `offset` (scores are normalized first). */
export function classFromScores(scores: ArrayLike<number>, offset: number, numClasses: number): { cls: number; confidence: number } {
  let sum = 0, best = 0;
  for (let c = 0; c < numClasses; c++) {
    sum += scores[offset + c];
    if (scores[offset + c] > scores[offset + best]) best = c;
  }
  const pmax = sum > 0 ? scores[offset + best] / sum : 1 / numClasses;
  const chance = 1 / numClasses;
  return { cls: best, confidence: Math.max(0, Math.min(1, (pmax - chance) / (1 - chance))) };
}

/** Classification: predicted class and confidence after layer l at point p. */
export function classAt(u: UnitMaps, numClasses: number, l: number, p: number): { cls: number; confidence: number } {
  return classFromScores(u.scores, (l * u.points + p) * numClasses, numClasses);
}

/** Text for the model-output tooltip: the prediction at one input point. */
export function describeClass(cls: number, confidence: number): string {
  return `class ${cls} (confidence ${(confidence * 100).toFixed(0)}%)`;
}

/** Regression: ⟨Z₀⟩ (the model output) if measured after layer l. */
export function outputValue(u: UnitMaps, l: number, p: number) {
  return u.scores[l * u.points + p];
}

// ---------- Colors ----------

export type RGB = [number, number, number];

function hex(h: string): RGB {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function mix(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** TF-Playground colors: orange for negative, blue for positive. */
export const NEGATIVE = '#f59322';
export const POSITIVE = '#0877bd';

export interface UnitPalette {
  neutral: RGB;
  negative: RGB;
  positive: RGB;
  sequential: RGB;
  classes: RGB[];
}

export function unitPalette(t: PlotTheme): UnitPalette {
  return {
    neutral: hex(t.name === 'dark' ? '#1b2422' : '#f4f6f5'),
    negative: hex(NEGATIVE),
    positive: hex(POSITIVE),
    sequential: hex(t.accent),
    classes: CLASS_COLORS.map(hex),
  };
}

export function divergingColor(pal: UnitPalette, v: number): RGB {
  const c = Math.max(-1, Math.min(1, v));
  return c >= 0 ? mix(pal.neutral, pal.positive, c) : mix(pal.neutral, pal.negative, -c);
}

export function sequentialColor(pal: UnitPalette, v: number): RGB {
  return mix(pal.neutral, pal.sequential, Math.max(0, Math.min(1, v)));
}

/** Class color faded towards the background where the prediction is unsure, as in TF Playground. */
/** One class's score v ∈ [0, 1] (probability or fidelity), from neutral to the class color. */
export function classScoreColor(pal: UnitPalette, cls: number, v: number): RGB {
  return mix(pal.neutral, pal.classes[cls], Math.max(0, Math.min(1, v)));
}

export function classColor(pal: UnitPalette, cls: number, confidence: number): RGB {
  return mix(pal.neutral, pal.classes[cls], 0.15 + 0.85 * confidence);
}

export function cssColor(c: RGB): string {
  return `rgb(${c.map(Math.round).join(',')})`;
}

// ---------- Rendering ----------

/** A value-to-color function for one map, or a function returning the color directly (class maps). */
export type PointColor = (p: number) => RGB;

/** Renders a size×size grid map (y up) to a PNG data URL. */
export function renderGrid(size: number, color: PointColor): string {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const [R, G, B] = color(r * size + c);
      const o = ((size - 1 - r) * size + c) * 4;
      img.data[o] = R;
      img.data[o + 1] = G;
      img.data[o + 2] = B;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

/**
 * Renders a 1-D map (regression) as a small line chart: value over x ∈ [-1, 1], with the background tinted by
 * the value so the tile reads like the 2-D maps.
 */
export function renderCurve(
  values: (p: number) => number,
  points: number,
  pal: UnitPalette,
  diverging: boolean,
  lineColor: string,
  width = 96,
  height = 96,
): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const lo = diverging ? -1 : 0;
  const toY = (v: number) => height - 6 - ((Math.max(lo, Math.min(1, v)) - lo) / (1 - lo)) * (height - 12);
  for (let p = 0; p < points; p++) {
    const v = values(p);
    ctx.fillStyle = cssColor(diverging ? divergingColor(pal, v) : sequentialColor(pal, v));
    const x0 = Math.floor((p / points) * width);
    ctx.fillRect(x0, 0, Math.ceil(width / points) + 1, height);
  }
  ctx.globalAlpha = 0.5;
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, toY(diverging ? 0 : 0.5));
  ctx.lineTo(width, toY(diverging ? 0 : 0.5));
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let p = 0; p < points; p++) {
    const x = (p / (points - 1)) * width;
    const y = toY(values(p));
    if (p === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  return canvas.toDataURL();
}
