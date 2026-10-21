import { Circuit, DEFAULT_ARCHITECTURE, FIXED_WEIGHT, isDataAngle, type Architecture, type NumQubits } from '../sim/circuit';
import type { ClassificationDataset, DataSet, DatasetName, TaskType } from '../data/datasets';
import { numClasses as datasetClasses, singleQubitTargets } from '../data/targets';
import { Rng } from '../data/rng';
import { Adam } from './adam';
import { classProbabilities, expvalZ0, fidelity, fidelityLoss, mseZLoss, nllLoss } from './losses';
import { applyRegularization, type RegType } from './regularization';

export interface ModelConfig {
  task: TaskType;
  dataset: DatasetName;
  numQubits: NumQubits;
  numLayers: number;
  /** Defaults to `DEFAULT_ARCHITECTURE`. */
  arch?: Architecture;
}

export interface TrainOptions {
  lr: number;
  batchSize: number;
  regType: RegType;
  regStrength: number;
}

export interface Evaluation {
  /** Mean loss without regularization (as reported by the reference `evaluate`). */
  loss: number;
  /** Classification accuracy, or NaN for regression. */
  accuracy: number;
  /** Predicted class index (classification) or ⟨Z⟩ (regression), per sample. */
  predictions: Float64Array;
  /** States after every layer: n * layers * stateSize floats, sample-major. */
  states: Float64Array;
}

/**
 * Data re-uploading classifier/regressor (ports of `QuantumReuploadingClassifier` and
 * `QuantumReuploadingRegressor`). Parameters live in one flat vector θ = [weights, biases].
 */
export class QuantumModel {
  readonly config: ModelConfig;
  readonly circuit: Circuit;
  /** Parameters per group (weights or biases). */
  readonly P: number;
  readonly theta: Float64Array;
  readonly weights: Float64Array;
  readonly biases: Float64Array;
  readonly numClasses: number;
  readonly targets: Float64Array[];
  readonly adam: Adam;
  readonly arch: Architecture;
  /** 1 for parameters that are trained; frozen ones (fixed or unused weights) keep their value and get zero gradient. */
  readonly trainable: Uint8Array;
  epoch = 0;
  /** ‖∂L/∂θ‖ per layer (weights and biases, without regularization) of the last `lossAndGrad`. */
  readonly layerGradNorms: Float64Array;
  /** Mean of `layerGradNorms` over the batches of the last `trainEpoch`. */
  readonly epochGradNorms: Float64Array;

  private readonly grad: Float64Array;
  private readonly gw: Float64Array;
  private readonly gb: Float64Array;
  private readonly g: Float64Array;
  private readonly probs = new Float64Array(4);
  private sampleStates: Float64Array;

  constructor(config: ModelConfig) {
    this.config = config;
    const features = config.task === 'regression' ? 1 : 2;
    this.arch = { ...DEFAULT_ARCHITECTURE, ...config.arch };
    this.circuit = new Circuit(config.numQubits, config.numLayers, features, this.arch.entangler);
    this.P = this.circuit.numAngles;
    this.trainable = new Uint8Array(2 * this.P).fill(1);
    for (let k = 0; k < this.P; k++) {
      if (!isDataAngle(k % 3, this.arch.dataAngles) || this.arch.weights === 'fixed') this.trainable[k] = 0;
    }
    this.theta = new Float64Array(2 * this.P);
    this.weights = this.theta.subarray(0, this.P);
    this.biases = this.theta.subarray(this.P);
    this.grad = new Float64Array(2 * this.P);
    this.gw = this.grad.subarray(0, this.P);
    this.gb = this.grad.subarray(this.P);
    this.g = new Float64Array(this.circuit.stateSize);
    this.sampleStates = new Float64Array(config.numLayers * this.circuit.stateSize);
    this.adam = new Adam(2 * this.P);
    this.layerGradNorms = new Float64Array(config.numLayers);
    this.epochGradNorms = new Float64Array(config.numLayers);
    if (config.task === 'classification') {
      const ds = config.dataset as ClassificationDataset;
      this.numClasses = datasetClasses(ds);
      this.targets = config.numQubits === 1 ? singleQubitTargets(ds) : [];
    } else {
      this.numClasses = 0;
      this.targets = [];
    }
  }

  /**
   * Random initialization mirroring the reference: standard normal for one qubit
   * (`torch.randn`), uniform [0, 1) for two qubits (`torch.rand`).
   */
  init(seed: number): void {
    const rng = new Rng(seed);
    const sample = this.config.numQubits === 1 ? () => rng.normal() : () => rng.next();
    for (let i = 0; i < this.theta.length; i++) this.theta[i] = sample();
    this.constrain();
    this.adam.reset();
    this.epoch = 0;
  }

  setParams(theta: ArrayLike<number>): void {
    this.theta.set(theta);
    this.constrain();
  }

  /** Sets the frozen weights: 0 for angles without data, π for fixed data weights. */
  private constrain(): void {
    for (let k = 0; k < this.P; k++) {
      if (!this.trainable[k]) this.weights[k] = isDataAngle(k % 3, this.arch.dataAngles) ? FIXED_WEIGHT : 0;
    }
  }

  private zeroFrozenGrad(): void {
    for (let i = 0; i < this.grad.length; i++) if (!this.trainable[i]) this.grad[i] = 0;
  }

  get stateSize(): number {
    return this.circuit.stateSize;
  }

  /** Sample loss and ∂ℓ/∂ψ (scaled) for the final state at `so`. */
  private sampleLoss(states: Float64Array, so: number, y: number, scale: number): number {
    const { numQubits, task } = this.config;
    const dim = this.circuit.dim;
    if (task === 'regression') return mseZLoss(states, so, dim, y, scale, this.g);
    if (numQubits === 1) return fidelityLoss(states, so, this.targets[y], scale, this.g);
    return nllLoss(states, so, dim, this.numClasses, y, scale, this.g);
  }

  /**
   * Mean loss over `idx[start, start+count)` and its gradient (incl. regularization) into `this.grad`.
   * Returns the loss including the regularization penalty.
   */
  lossAndGrad(set: DataSet, idx: ArrayLike<number>, start: number, count: number, opts?: TrainOptions): number {
    const { circuit, weights, biases, gw, gb, g } = this;
    const L = this.config.numLayers;
    const ss = circuit.stateSize;
    const lastOffset = (L - 1) * ss;
    this.grad.fill(0);
    const scale = 1 / count;
    let loss = 0;
    for (let s = 0; s < count; s++) {
      const i = idx[start + s];
      const xo = i * set.features;
      circuit.forward(set.x, xo, weights, biases, this.sampleStates, 0);
      loss += this.sampleLoss(this.sampleStates, lastOffset, set.y[i], scale) * scale;
      circuit.backward(set.x, xo, weights, biases, this.sampleStates, 0, g, gw, gb);
    }
    this.zeroFrozenGrad();
    const per = this.P / this.config.numLayers;
    for (let l = 0; l < this.config.numLayers; l++) {
      let sq = 0;
      for (let k = l * per; k < (l + 1) * per; k++) sq += gw[k] * gw[k] + gb[k] * gb[k];
      this.layerGradNorms[l] = Math.sqrt(sq);
    }
    if (opts && opts.regType !== 'none') {
      loss += applyRegularization(this.theta, this.grad, opts.regType, opts.regStrength);
      this.zeroFrozenGrad();
    }
    return loss;
  }

  get gradient(): Float64Array {
    return this.grad;
  }

  /**
   * One epoch of shuffled mini-batch Adam (`train_single_epoch`). The last partial batch is dropped
   * as in the reference; a batch size larger than the data set is clamped to the data set size.
   * Adam state persists across epochs (the reference re-created the optimizer every epoch).
   */
  trainEpoch(set: DataSet, opts: TrainOptions, rng: Rng): number {
    const order = new Int32Array(set.n);
    for (let i = 0; i < set.n; i++) order[i] = i;
    rng.shuffle(order);
    const bs = Math.max(1, Math.min(opts.batchSize, set.n));
    let total = 0;
    let batches = 0;
    this.epochGradNorms.fill(0);
    for (let start = 0; start + bs <= set.n; start += bs) {
      total += this.lossAndGrad(set, order, start, bs, opts);
      for (let l = 0; l < this.layerGradNorms.length; l++) this.epochGradNorms[l] += this.layerGradNorms[l];
      this.adam.step(this.theta, this.grad, opts.lr);
      batches++;
    }
    if (batches) for (let l = 0; l < this.epochGradNorms.length; l++) this.epochGradNorms[l] /= batches;
    this.epoch += 1;
    return batches ? total / batches : NaN;
  }

  /** Runs every sample, storing all layer states (n * L * stateSize floats). */
  forwardAll(set: DataSet, out?: Float64Array): Float64Array {
    const per = this.config.numLayers * this.circuit.stateSize;
    const states = out ?? new Float64Array(set.n * per);
    for (let i = 0; i < set.n; i++) {
      this.circuit.forward(set.x, i * set.features, this.weights, this.biases, states, i * per);
    }
    return states;
  }

  /**
   * Classification score of the final state at `so`:
   * single qubit → fidelity with each target; two qubits → renormalized class probabilities.
   */
  scores(states: Float64Array, so: number, out: Float64Array): void {
    if (this.config.numQubits === 1) {
      for (let c = 0; c < this.numClasses; c++) out[c] = fidelity(states, so, this.targets[c]);
    } else {
      classProbabilities(states, so, this.circuit.dim, this.numClasses, out);
    }
  }

  evaluate(set: DataSet): Evaluation {
    const states = this.forwardAll(set);
    const per = this.config.numLayers * this.circuit.stateSize;
    const lastOffset = (this.config.numLayers - 1) * this.circuit.stateSize;
    const predictions = new Float64Array(set.n);
    const scale = 1 / set.n;
    let loss = 0;
    let correct = 0;
    for (let i = 0; i < set.n; i++) {
      const so = i * per + lastOffset;
      loss += this.sampleLoss(states, so, set.y[i], scale) * scale;
      if (this.config.task === 'regression') {
        predictions[i] = expvalZ0(states, so, this.circuit.dim);
      } else {
        this.scores(states, so, this.probs);
        let best = 0;
        for (let c = 1; c < this.numClasses; c++) if (this.probs[c] > this.probs[best]) best = c;
        predictions[i] = best;
        if (best === set.y[i]) correct++;
      }
    }
    return {
      loss,
      accuracy: this.config.task === 'regression' ? NaN : correct / set.n,
      predictions,
      states,
    };
  }

  /**
   * Class scores on an n×n grid over [-1, 1]² (row-major, y outer; [point][class]): fidelities with the class
   * targets (1 qubit) or renormalized class probabilities (2 qubits). Class 0 matches `evaluate_decision_boundary`.
   */
  decisionGrid(n: number): Float64Array {
    const pts = new Float64Array(2 * n * n);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const k = r * n + c;
        pts[2 * k] = -1 + (2 * c) / (n - 1);
        pts[2 * k + 1] = -1 + (2 * r) / (n - 1);
      }
    }
    const set: DataSet = { n: n * n, features: 2, x: pts, y: new Float64Array(n * n) };
    const states = this.forwardAll(set);
    const per = this.config.numLayers * this.circuit.stateSize;
    const lastOffset = (this.config.numLayers - 1) * this.circuit.stateSize;
    const C = this.numClasses;
    const z = new Float64Array(n * n * C);
    for (let i = 0; i < n * n; i++) {
      this.scores(states, i * per + lastOffset, this.probs);
      for (let c = 0; c < C; c++) z[i * C + c] = this.probs[c];
    }
    return z;
  }

  /** ⟨Z⟩ predictions for 1-D inputs. */
  predictRegression(x: Float64Array): Float64Array {
    const ss = this.circuit.stateSize;
    const lastOffset = (this.config.numLayers - 1) * ss;
    const out = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) {
      this.circuit.forward(x, i, this.weights, this.biases, this.sampleStates, 0);
      out[i] = expvalZ0(this.sampleStates, lastOffset, this.circuit.dim);
    }
    return out;
  }

  /**
   * Monte Carlo uncertainty (`predict_mc`): M forward passes with Gaussian noise N(0, σ²) added to all
   * weights and biases. Returns mean and unbiased standard deviation per input.
   */
  predictMc(x: Float64Array, sigma: number, runs: number, rng: Rng): { mean: Float64Array; std: Float64Array } {
    if (sigma <= 0 || runs <= 1) {
      return { mean: this.predictRegression(x), std: new Float64Array(x.length) };
    }
    const original = this.theta.slice();
    const sum = new Float64Array(x.length);
    const sumSq = new Float64Array(x.length);
    for (let m = 0; m < runs; m++) {
      for (let i = 0; i < original.length; i++) this.theta[i] = original[i] + sigma * rng.normal();
      const y = this.predictRegression(x);
      for (let i = 0; i < x.length; i++) {
        sum[i] += y[i];
        sumSq[i] += y[i] * y[i];
      }
    }
    this.theta.set(original);
    const mean = sum.map((s) => s / runs);
    const std = sumSq.map((s2, i) => Math.sqrt(Math.max(0, (s2 - runs * mean[i] * mean[i]) / (runs - 1))));
    return { mean, std };
  }
}
