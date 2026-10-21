import type { DatasetName, TaskType } from '../data/datasets';
import type { Architecture, NumQubits } from '../sim/circuit';
import type { Settings } from '../state/settings';

export interface RestoreState {
  theta: ArrayLike<number>;
  epoch: number;
}

export type ToWorker =
  | { type: 'configure'; settings: Settings; restore?: RestoreState }
  | { type: 'hyper'; settings: Settings }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step' }
  | { type: 'fastForward'; epochs: number }
  /** View the model as it was after `epoch` (pauses training). */
  | { type: 'seek'; epoch: number }
  /** Discard history after `epoch` and continue training from there. */
  | { type: 'resumeFrom'; epoch: number }
  /** Return the view to the latest epoch. */
  | { type: 'live' };

/** Static information about the current configuration, sent once per `configure`. */
export interface DataInfo {
  configId: number;
  task: TaskType;
  dataset: DatasetName;
  numQubits: NumQubits;
  numLayers: number;
  arch: Architecture;
  numClasses: number;
  trainX: Float64Array;
  trainY: Float64Array;
  testX: Float64Array;
  testY: Float64Array;
  /** Bloch vectors of the single-qubit class targets (classification, 1 qubit). */
  targetsBloch: number[][];
  regression?: {
    gridX: Float64Array;
    gridYTrue: Float64Array;
    gridYNoisy: Float64Array;
    trainMask: Uint8Array;
    denseX: Float64Array;
    denseY: Float64Array;
  };
}

export interface MetricsChunk {
  /** Epoch of the first metrics entry of this configuration. */
  base: number;
  /** Epoch of the first entry in this chunk; the receiver drops everything from this epoch on first. */
  from: number;
  loss: number[];
  /** Train accuracy (classification) or train MSE (regression). */
  train: number[];
  /** Test accuracy (classification) or test MSE (regression). */
  test: number[];
  /** Mean ‖∂L/∂θ‖ per layer over the epoch's batches, [epoch][layer]; NaN for the starting epoch. */
  gradNorm: number[][];
  /** Two qubits: mean test-set tangle after each layer, [epoch][layer]; empty rows for one qubit. */
  tangle: number[][];
}

/**
 * "Activations" of every layer over the input space, for the circuit view (TF-Playground-style unit maps).
 * Classification: a size × size grid over [-1, 1]² (row-major, y outer). Regression: `size` points on x.
 */
export interface UnitMaps {
  epoch: number;
  /** Grid side length (classification) or number of x points (regression). */
  size: number;
  points: number;
  /** Reduced Bloch vector and purity per qubit: [layer][qubit][point][x, y, z, purity]. */
  bloch: Float32Array;
  /** Two qubits: tangle per [layer][point]. Empty for one qubit. */
  tangle: Float32Array;
  /** Classification: class scores if measured after each layer, [layer][point][class]. Regression: ⟨Z₀⟩ per [layer][point]. */
  scores: Float32Array;
}

export interface Snapshot {
  configId: number;
  /** Epoch being displayed. */
  epoch: number;
  /** Most recent trained epoch. */
  latestEpoch: number;
  /** Oldest epoch still kept in the replay history. */
  firstEpoch: number;
  metrics: MetricsChunk;
  /**
   * Visual coordinates of every test sample after every layer, 4 floats each, sample-major
   * ([sample][layer][4]): Bloch vector + 0 for one qubit; (p|01⟩, p|10⟩, p|00⟩, tangle) for two qubits.
   */
  layerVis: Float32Array;
  /** Test-set predictions: class index or ⟨Z⟩. */
  predictions: Float32Array;
  /** Classification: class scores on a gridSize × gridSize grid (row-major, y outer; [point][class]). */
  decision?: Float32Array;
  gridSize: number;
  /** Regression: MC prediction mean and std on DataInfo.regression.denseX. */
  curve?: { mean: Float32Array; std: Float32Array };
  /** Current parameters [weights, biases], each laid out as (layer, qubit, 3). */
  theta: Float32Array;
  /**
   * Full-batch gradient ∂L/∂θ over the training set (incl. regularization), same layout as `theta`.
   * Refreshed together with the unit maps, so it may lag a few epochs behind while training.
   */
  grad: Float32Array;
  /** Per-layer unit maps; may lag a few epochs behind `epoch` while training (refreshed at ~10 Hz). */
  units: UnitMaps;
}

export type FromWorker =
  | { type: 'data'; info: DataInfo }
  | { type: 'snapshot'; snap: Snapshot }
  | { type: 'status'; playing: boolean; busy: boolean }
  | { type: 'error'; message: string };
