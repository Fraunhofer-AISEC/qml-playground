import { createTaskData, fourierCurve, type TaskData } from '../data/datasets';
import { Rng } from '../data/rng';
import { QuantumModel, type TrainOptions } from '../model/qmodel';
import { architecture, type Settings } from '../state/settings';
import { expvalZ0 } from '../model/losses';
import { blochVector, reducedBloch, tangle, visualCoordinates } from '../viz/quantum';
import type { DataInfo, MetricsChunk, RestoreState, Snapshot, UnitMaps } from './protocol';

/** Maximum number of epochs kept for replay (parameters + optimizer state per epoch). */
export const MAX_HISTORY = 5000;
const DENSE_POINTS = 400;
/** Unit-map resolution: grid side for classification, number of x points for regression. */
export const UNIT_GRID = 24;
export const UNIT_POINTS_1D = 64;

/** Mean tangle over all samples after each layer (states sample-major, `layers` states per sample). */
export function meanTangle(states: Float64Array, n: number, layers: number, stateSize: number): number[] {
  const out = new Array<number>(layers).fill(0);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < layers; l++) out[l] += tangle(states, (i * layers + l) * stateSize);
  }
  return out.map((v) => v / n);
}

interface HistoryEntry {
  epoch: number;
  theta: Float32Array;
  m: Float32Array;
  v: Float32Array;
  t: number;
}

/**
 * Owns the model, data and training history. Runs in the Web Worker but has no worker dependencies,
 * so it can be unit-tested directly.
 */
export class Trainer {
  readonly configId: number;
  readonly data: TaskData;
  readonly model: QuantumModel;
  settings: Settings;

  private rng: Rng;
  private mcRng: Rng;
  private history: HistoryEntry[] = [];
  private metrics = {
    loss: [] as number[],
    train: [] as number[],
    test: [] as number[],
    gradNorm: [] as number[][],
    tangle: [] as number[][],
  };
  /** First metrics index not yet sent to the main thread. */
  private unsentFrom = 0;
  /** Epoch currently displayed; null while following the latest epoch. */
  viewEpoch: number | null = null;
  private dense?: { x: Float64Array; y: Float64Array };
  private lastCurve?: { key: string; curve: { mean: Float32Array; std: Float32Array } };
  private lastUnits?: UnitMaps;
  private lastGrad?: { key: string; grad: Float32Array };
  private readonly trainIdx: Int32Array;

  constructor(configId: number, settings: Settings, restore?: RestoreState) {
    this.configId = configId;
    this.settings = settings;
    this.data = createTaskData(settings.task, settings.dataset, settings.seed);
    this.trainIdx = Int32Array.from({ length: this.data.train.n }, (_, i) => i);
    this.model = new QuantumModel({
      task: settings.task,
      dataset: settings.dataset,
      numQubits: settings.numQubits,
      numLayers: settings.numLayers,
      arch: architecture(settings),
    });
    this.model.init(settings.seed);
    this.rng = new Rng(settings.seed ^ 0x9e3779b9);
    this.mcRng = new Rng(settings.seed ^ 0x85ebca6b);
    if (restore && restore.theta.length === this.model.theta.length) {
      this.model.setParams(restore.theta);
      this.model.epoch = Math.max(0, Math.floor(restore.epoch));
    }
    this.metricsStart = this.model.epoch;
    if (settings.task === 'regression') {
      const [x, y] = fourierCurve(settings.dataset, DENSE_POINTS);
      this.dense = { x, y };
    }
    this.record(false);
  }

  get latestEpoch(): number {
    return this.model.epoch;
  }

  get firstEpoch(): number {
    return this.history.length ? this.history[0].epoch : this.model.epoch;
  }

  /** Metrics entry i belongs to epoch metricsStart + i (the epoch the trainer was created with). */
  private metricsStart = 0;

  get trainOptions(): TrainOptions {
    const { lr, batchSize, regType, regStrength } = this.settings;
    return { lr, batchSize, regType, regStrength };
  }

  dataInfo(): DataInfo {
    const { task, dataset, numQubits, numLayers } = this.settings;
    const reg = this.data.regression;
    return {
      configId: this.configId,
      task,
      dataset,
      numQubits,
      numLayers,
      arch: this.model.arch,
      numClasses: this.model.numClasses,
      trainX: this.data.train.x,
      trainY: this.data.train.y,
      testX: this.data.test.x,
      testY: this.data.test.y,
      targetsBloch: this.model.targets.map((t) => blochVector(t)),
      regression:
        reg && this.dense
          ? {
              gridX: reg.x,
              gridYTrue: reg.yTrue,
              gridYNoisy: reg.yNoisy,
              trainMask: reg.trainMask,
              denseX: this.dense.x,
              denseY: this.dense.y,
            }
          : undefined,
    };
  }

  /**
   * Evaluates the current parameters and appends metrics + history for the current epoch.
   * `trained`: the epoch was just trained, so the model's epoch gradient norms belong to it.
   */
  private record(trained = true): void {
    const train = this.model.evaluate(this.data.train);
    const test = this.model.evaluate(this.data.test);
    const isReg = this.settings.task === 'regression';
    const L = this.settings.numLayers;
    this.metrics.loss.push(train.loss);
    this.metrics.train.push(isReg ? train.loss : train.accuracy);
    this.metrics.test.push(isReg ? test.loss : test.accuracy);
    this.metrics.gradNorm.push(trained ? Array.from(this.model.epochGradNorms) : new Array<number>(L).fill(NaN));
    this.metrics.tangle.push(this.settings.numQubits === 2 ? meanTangle(test.states, this.data.test.n, L, this.model.stateSize) : []);
    const a = this.model.adam;
    this.history.push({
      epoch: this.model.epoch,
      theta: Float32Array.from(this.model.theta),
      m: Float32Array.from(a.m),
      v: Float32Array.from(a.v),
      t: a.t,
    });
    if (this.history.length > MAX_HISTORY) this.history.shift();
  }

  stepEpoch(): void {
    if (this.viewEpoch !== null) this.resumeFrom(this.viewEpoch);
    this.model.trainEpoch(this.data.train, this.trainOptions, this.rng);
    this.record();
  }

  private entry(epoch: number): HistoryEntry | undefined {
    const i = epoch - this.firstEpoch;
    return this.history[i];
  }

  seek(epoch: number): void {
    const clamped = Math.min(this.latestEpoch, Math.max(this.firstEpoch, Math.round(epoch)));
    this.viewEpoch = clamped === this.latestEpoch ? null : clamped;
  }

  /** Truncates history and metrics after `epoch` and restores the model + optimizer state at that epoch. */
  resumeFrom(epoch: number): void {
    const e = this.entry(Math.min(this.latestEpoch, Math.max(this.firstEpoch, Math.round(epoch))));
    this.viewEpoch = null;
    if (!e || e.epoch === this.latestEpoch) return;
    this.model.setParams(e.theta);
    this.model.adam.m.set(e.m);
    this.model.adam.v.set(e.v);
    this.model.adam.t = e.t;
    this.model.epoch = e.epoch;
    this.history.length = e.epoch - this.firstEpoch + 1;
    const keep = e.epoch - this.metricsStart + 1;
    this.metrics.loss.length = keep;
    this.metrics.train.length = keep;
    this.metrics.test.length = keep;
    this.metrics.gradNorm.length = keep;
    this.metrics.tangle.length = keep;
    this.unsentFrom = Math.min(this.unsentFrom, keep);
  }

  private takeMetrics(): MetricsChunk {
    const from = this.unsentFrom;
    this.unsentFrom = this.metrics.loss.length;
    return {
      base: this.metricsStart,
      from: this.metricsStart + from,
      loss: this.metrics.loss.slice(from),
      train: this.metrics.train.slice(from),
      test: this.metrics.test.slice(from),
      gradNorm: this.metrics.gradNorm.slice(from),
      tangle: this.metrics.tangle.slice(from),
    };
  }

  /** Re-send the complete metrics history with the next snapshot. */
  resendMetrics(): void {
    this.unsentFrom = 0;
  }

  /**
   * Builds a snapshot for the displayed epoch. While viewing history the model parameters are temporarily
   * swapped for the recorded ones.
   */
  /** Computes the per-layer unit maps for the current parameters (see `UnitMaps`). */
  private computeUnits(epoch: number): UnitMaps {
    const { model, settings } = this;
    const regression = settings.task === 'regression';
    const size = regression ? UNIT_POINTS_1D : UNIT_GRID;
    const points = regression ? size : size * size;
    const x = new Float64Array(points * (regression ? 1 : 2));
    if (regression) {
      for (let i = 0; i < size; i++) x[i] = -1 + (2 * i) / (size - 1);
    } else {
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const k = r * size + c;
          x[2 * k] = -1 + (2 * c) / (size - 1);
          x[2 * k + 1] = -1 + (2 * r) / (size - 1);
        }
      }
    }
    const L = settings.numLayers;
    const Q = settings.numQubits;
    const ss = model.stateSize;
    const states = model.forwardAll({ n: points, features: regression ? 1 : 2, x, y: new Float64Array(points) });
    const C = regression ? 1 : model.numClasses;
    const bloch = new Float32Array(L * Q * points * 4);
    const tangleOut = new Float32Array(Q === 2 ? L * points : 0);
    const scores = new Float32Array(L * points * C);
    const tmp = new Float64Array(Math.max(4, C));
    for (let l = 0; l < L; l++) {
      for (let p = 0; p < points; p++) {
        const so = (p * L + l) * ss;
        for (let q = 0; q < Q; q++) reducedBloch(states, so, Q, q, bloch, ((l * Q + q) * points + p) * 4);
        if (Q === 2) tangleOut[l * points + p] = tangle(states, so);
        if (regression) {
          scores[l * points + p] = expvalZ0(states, so, model.circuit.dim);
        } else {
          model.scores(states, so, tmp);
          for (let c = 0; c < C; c++) scores[(l * points + p) * C + c] = tmp[c];
        }
      }
    }
    return { epoch, size, points, bloch, tangle: tangleOut, scores };
  }

  snapshot(options: { mc: boolean; units?: boolean } = { mc: true }): Snapshot {
    const { model, settings } = this;
    const viewing = this.viewEpoch !== null ? this.entry(this.viewEpoch) : undefined;
    const saved = viewing ? model.theta.slice() : undefined;
    if (viewing) model.setParams(viewing.theta);

    const test = model.evaluate(this.data.test);
    const layerVis = visualCoordinates(test.states, this.data.test.n, settings.numLayers, settings.numQubits);
    let decision: Float32Array | undefined;
    let curve: Snapshot['curve'];
    if (settings.task === 'classification') {
      decision = Float32Array.from(model.decisionGrid(settings.gridSize));
    } else if (this.dense) {
      // MC sampling is the most expensive part of a snapshot; callers may skip it and reuse the last curve.
      const key = `${viewing?.epoch ?? model.epoch}|${settings.noiseSigma}|${settings.mcRuns}`;
      const cheap = settings.noiseSigma <= 0 || settings.mcRuns <= 1;
      if (this.lastCurve?.key !== key && (cheap || options.mc || !this.lastCurve)) {
        const { mean, std } = model.predictMc(this.dense.x, settings.noiseSigma, settings.mcRuns, this.mcRng);
        this.lastCurve = { key, curve: { mean: Float32Array.from(mean), std: Float32Array.from(std) } };
      }
      curve = this.lastCurve!.curve;
    }
    const shownEpoch = viewing?.epoch ?? model.epoch;
    // Unit maps are the most expensive part after MC; refresh them when asked or when they are missing.
    if (!this.lastUnits || (this.lastUnits.epoch !== shownEpoch && (options.units ?? true))) {
      this.lastUnits = this.computeUnits(shownEpoch);
    }
    // The full-batch gradient is refreshed at the same cadence as the unit maps.
    const { regType, regStrength } = settings;
    const gradKey = `${shownEpoch}|${regType}|${regStrength}`;
    if (!this.lastGrad || (this.lastGrad.key !== gradKey && (options.units ?? true))) {
      model.lossAndGrad(this.data.train, this.trainIdx, 0, this.data.train.n, this.trainOptions);
      this.lastGrad = { key: gradKey, grad: Float32Array.from(model.gradient) };
    }
    // Send a copy: the snapshot's buffers are transferred to the main thread.
    const u = this.lastUnits;
    const units: UnitMaps = { ...u, bloch: u.bloch.slice(), tangle: u.tangle.slice(), scores: u.scores.slice() };
    const theta = Float32Array.from(model.theta);
    if (saved) model.setParams(saved);

    return {
      configId: this.configId,
      epoch: viewing?.epoch ?? model.epoch,
      latestEpoch: this.latestEpoch,
      firstEpoch: this.firstEpoch,
      metrics: this.takeMetrics(),
      layerVis,
      predictions: Float32Array.from(test.predictions),
      decision,
      gridSize: settings.gridSize,
      curve,
      theta,
      grad: this.lastGrad.grad.slice(),
      units,
    };
  }
}
