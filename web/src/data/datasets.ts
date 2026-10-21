import { Rng } from './rng';

export type TaskType = 'classification' | 'regression';

export const classificationDatasets = {
  circle: 'Circle',
  '3_circles': '3 Circles',
  square: 'Square',
  '4_squares': '4 Squares',
  crown: 'Crown',
  tricrown: 'Tricrown',
  wavy_lines: 'Wavy Lines',
} as const;

export const regressionDatasets = {
  fourier_1: 'Fourier series degree 1',
  fourier_2: 'Fourier series degree 2',
  fourier_3: 'Fourier series degree 3',
  fourier_4: 'Fourier series degree 4',
  fourier_5: 'Fourier series degree 5',
} as const;

export type ClassificationDataset = keyof typeof classificationDatasets;
export type RegressionDataset = keyof typeof regressionDatasets;
export type DatasetName = ClassificationDataset | RegressionDataset;

export function datasetsFor(task: TaskType): Record<string, string> {
  return task === 'regression' ? regressionDatasets : classificationDatasets;
}

export function isValidDataset(task: TaskType, name: string): name is DatasetName {
  return name in datasetsFor(task);
}

/** Row-major samples: `x` has n*features entries, `y` has n entries (class index or target value). */
export interface DataSet {
  n: number;
  features: number;
  x: Float64Array;
  y: Float64Array;
}

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
const CROWN_R = [Math.sqrt(0.8), Math.sqrt(0.8 - 2 / Math.PI)];
const THREE_CIRCLES = {
  centers: [
    [-1, 1],
    [1, 0],
    [-0.5, -0.5],
  ],
  radii: [1, Math.sqrt(6 / Math.PI - 1), 0.5],
};

/** Label functions ported from `app/data/datasets_torch.py`. */
export function labelPoint(name: ClassificationDataset, px: number, py: number): number {
  const r = Math.hypot(px, py);
  switch (name) {
    case 'circle':
      return r > SQRT_2_PI ? 1 : 0;
    case '3_circles': {
      let label = 0;
      THREE_CIRCLES.centers.forEach(([cx, cy], j) => {
        if (Math.hypot(px - cx, py - cy) < THREE_CIRCLES.radii[j]) label = 1 + j;
      });
      return label;
    }
    case 'square':
      return Math.max(Math.abs(px), Math.abs(py)) > 0.5 * Math.SQRT2 ? 1 : 0;
    case '4_squares':
      if (px < 0 && py > 0) return 1;
      if (px > 0 && py < 0) return 2;
      if (px > 0 && py > 0) return 3;
      return 0;
    case 'crown':
      return r < CROWN_R[0] && r > CROWN_R[1] ? 1 : 0;
    case 'tricrown':
      if (r > CROWN_R[0]) return 2;
      if (r < CROWN_R[0] && r > CROWN_R[1]) return 1;
      return 0;
    case 'wavy_lines': {
      const f1 = wavyUpper(px);
      const f2 = wavyLower(px);
      if (py < f1 && py > f2) return 1;
      if (py > f1 && py < f2) return 2;
      if (py > f1 && py > f2) return 3;
      return 0;
    }
  }
}

export const wavyUpper = (s: number) => s + Math.sin(Math.PI * s);
export const wavyLower = (s: number) => -s + Math.sin(Math.PI * s);

export function createClassificationSet(name: ClassificationDataset, samples: number, seed: number): DataSet {
  const rng = new Rng(seed);
  const x = new Float64Array(2 * samples);
  const y = new Float64Array(samples);
  for (let i = 0; i < samples; i++) {
    const px = 1 - 2 * rng.next();
    const py = 1 - 2 * rng.next();
    x[2 * i] = px;
    x[2 * i + 1] = py;
    y[i] = labelPoint(name, px, py);
  }
  return { n: samples, features: 2, x, y };
}

export function fourierDegree(name: string): number {
  const degree = Number(name.split('_')[1]);
  if (!Number.isInteger(degree) || degree < 1 || degree > 5) throw new Error(`Unknown fourier dataset ${name}`);
  return degree;
}

/** Normalized real Fourier series on x ∈ [-1, 1] (port of `_fourier`). Returns [x, y]. */
export function fourierCurve(name: string, samples: number): [Float64Array, Float64Array] {
  const degree = fourierDegree(name);
  const x = new Float64Array(samples);
  const y = new Float64Array(samples);
  let maxAbs = 0;
  for (let i = 0; i < samples; i++) {
    x[i] = samples === 1 ? -1 : -1 + (2 * i) / (samples - 1);
    const t = x[i] * Math.PI;
    let v = 0.1;
    for (let k = 1; k <= degree; k++) v += 0.3 * Math.cos(k * t) - 0.3 * Math.sin(k * t);
    y[i] = v;
    maxAbs = Math.max(maxAbs, Math.abs(v));
  }
  if (maxAbs > 0) for (let i = 0; i < samples; i++) y[i] /= maxAbs;
  return [x, y];
}

export interface TaskData {
  train: DataSet;
  test: DataSet;
  /** Regression only: the full noisy sample grid and the training mask (for the data plot). */
  regression?: { x: Float64Array; yTrue: Float64Array; yNoisy: Float64Array; trainMask: Uint8Array };
}

export const REGRESSION_NOISE = 0.075;

/**
 * Mirrors `update_data` in `app/logic.py`:
 *  - classification: 500 training points (seed 42), 300 test points (seed 43)
 *  - regression: 100 grid points, Gaussian noise σ=0.075 on training targets, training gaps at
 *    (-0.1, 0.1) and x > 0.8; the test set is the clean full grid.
 * Unlike the reference, the regression noise is seeded so the data is reproducible.
 */
export function createTaskData(task: TaskType, name: DatasetName, seed = 42): TaskData {
  if (task === 'classification') {
    return {
      train: createClassificationSet(name as ClassificationDataset, 500, seed),
      test: createClassificationSet(name as ClassificationDataset, 300, seed + 1),
    };
  }
  const [x, yTrue] = fourierCurve(name, 100);
  const rng = new Rng(seed);
  const yNoisy = yTrue.map((v) => v + REGRESSION_NOISE * rng.normal());
  const trainMask = new Uint8Array(x.length);
  const trainX: number[] = [];
  const trainY: number[] = [];
  x.forEach((xi, i) => {
    const inGap = xi > -0.1 && xi < 0.1;
    const tailGap = xi > 0.8;
    if (!inGap && !tailGap) {
      trainMask[i] = 1;
      trainX.push(xi);
      trainY.push(yNoisy[i]);
    }
  });
  return {
    train: { n: trainX.length, features: 1, x: Float64Array.from(trainX), y: Float64Array.from(trainY) },
    test: { n: x.length, features: 1, x, y: yTrue },
    regression: { x, yTrue, yNoisy, trainMask },
  };
}
