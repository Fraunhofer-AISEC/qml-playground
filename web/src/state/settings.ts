import { isValidDataset, type DatasetName, type TaskType } from '../data/datasets';
import type { RegType } from '../model/regularization';
import { DEFAULT_ARCHITECTURE, type Architecture, type DataAngles, type Entangler, type NumQubits, type WeightMode } from '../sim/circuit';

/** Everything the user can configure. Mirrors the controls in `app/layout.py`. */
export interface Settings {
  task: TaskType;
  dataset: DatasetName;
  numQubits: NumQubits;
  numLayers: number;
  lr: number;
  batchSize: number;
  regType: RegType;
  regStrength: number;
  noiseSigma: number;
  mcRuns: number;
  /** Seed for parameter initialization, data noise and shuffling. */
  seed: number;
  /** Epochs per second while playing; 0 means "as fast as possible". */
  speed: number;
  /** Resolution of the decision-boundary grid (n × n). */
  gridSize: number;
  /** Architecture options (see `Architecture`); the defaults reproduce the original model. */
  entangler: Entangler;
  dataAngles: DataAngles;
  weights: WeightMode;
}

/** Changing one of these re-initializes the model, like the reset triggers of the Dash app. */
export const STRUCTURAL_KEYS = ['task', 'dataset', 'numQubits', 'numLayers', 'seed', 'entangler', 'dataAngles', 'weights'] as const;
export type StructuralSettings = Pick<Settings, (typeof STRUCTURAL_KEYS)[number]>;

export const LR_OPTIONS = [0.001, 0.01, 0.05, 0.1, 0.5];
export const REG_STRENGTH_OPTIONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];
export const REG_TYPES: { value: RegType; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'l1', label: 'L1' },
  { value: 'l2', label: 'L2' },
];
export const SPEED_OPTIONS = [1, 2, 5, 10, 20, 30, 60, 0];
export const LAYERS = { min: 1, max: 15 };
export const BATCH = { min: 1, max: 128 };
export const SIGMA = { min: 0, max: 1, step: 0.01 };
export const MC_RUNS = { min: 1, max: 30 };
export const GRID_SIZES = [15, 25, 40, 60];
export const ENTANGLERS: { value: Entangler; label: string }[] = [
  { value: 'cnot', label: 'CNOT' },
  { value: 'cz', label: 'CZ' },
  { value: 'none', label: 'None' },
];
export const DATA_ANGLES: { value: DataAngles; label: string }[] = [
  { value: 'all', label: 'All angles' },
  { value: 'ry', label: 'RY only' },
];
export const WEIGHT_MODES: { value: WeightMode; label: string }[] = [
  { value: 'trainable', label: 'Trainable' },
  { value: 'fixed', label: 'Fixed π' },
];

export function architecture(s: Settings): Architecture {
  return { entangler: s.entangler, dataAngles: s.dataAngles, weights: s.weights };
}

export const DEFAULT_SETTINGS: Settings = {
  task: 'classification',
  dataset: 'circle',
  numQubits: 1,
  numLayers: 5,
  lr: 0.01,
  batchSize: 32,
  regType: 'none',
  regStrength: 0.01,
  noiseSigma: 0,
  mcRuns: 1,
  seed: 42,
  speed: 2,
  gridSize: 40,
  ...DEFAULT_ARCHITECTURE,
};

export function defaultDataset(task: TaskType): DatasetName {
  return task === 'regression' ? 'fourier_1' : 'circle';
}

const clampInt = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(v)));

/** Coerces a partial/untrusted settings object into a valid one (used for URL state). */
export function sanitizeSettings(partial: Partial<Record<keyof Settings, unknown>>): Settings {
  const s: Settings = { ...DEFAULT_SETTINGS };
  const num = (v: unknown) => (typeof v === 'number' ? v : typeof v === 'string' && v !== '' ? Number(v) : NaN);

  if (partial.task === 'classification' || partial.task === 'regression') s.task = partial.task;
  s.dataset =
    typeof partial.dataset === 'string' && isValidDataset(s.task, partial.dataset)
      ? partial.dataset
      : defaultDataset(s.task);

  const q = num(partial.numQubits);
  if (q === 1 || q === 2) s.numQubits = q;
  const L = num(partial.numLayers);
  if (Number.isFinite(L)) s.numLayers = clampInt(L, LAYERS.min, LAYERS.max);
  const lr = num(partial.lr);
  if (LR_OPTIONS.includes(lr)) s.lr = lr;
  const bs = num(partial.batchSize);
  if (Number.isFinite(bs)) s.batchSize = clampInt(bs, BATCH.min, BATCH.max);
  if (REG_TYPES.some((r) => r.value === partial.regType)) s.regType = partial.regType as RegType;
  const rs = num(partial.regStrength);
  if (REG_STRENGTH_OPTIONS.includes(rs)) s.regStrength = rs;
  const sig = num(partial.noiseSigma);
  if (Number.isFinite(sig)) s.noiseSigma = Math.min(SIGMA.max, Math.max(SIGMA.min, Math.round(sig * 100) / 100));
  const mc = num(partial.mcRuns);
  if (Number.isFinite(mc)) s.mcRuns = clampInt(mc, MC_RUNS.min, MC_RUNS.max);
  const seed = num(partial.seed);
  if (Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32) s.seed = seed;
  const spd = num(partial.speed);
  if (SPEED_OPTIONS.includes(spd)) s.speed = spd;
  const grid = num(partial.gridSize);
  if (GRID_SIZES.includes(grid)) s.gridSize = grid;
  if (ENTANGLERS.some((o) => o.value === partial.entangler)) s.entangler = partial.entangler as Entangler;
  if (DATA_ANGLES.some((o) => o.value === partial.dataAngles)) s.dataAngles = partial.dataAngles as DataAngles;
  if (WEIGHT_MODES.some((o) => o.value === partial.weights)) s.weights = partial.weights as WeightMode;
  return s;
}

export function structuralChanged(a: Settings, b: Settings): boolean {
  return STRUCTURAL_KEYS.some((k) => a[k] !== b[k]);
}
