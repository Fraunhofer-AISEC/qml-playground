import { useState, type ReactNode } from 'react';
import { useApp } from '../../state/store';
import {
  DATA_ANGLES,
  DEFAULT_SETTINGS,
  ENTANGLERS,
  LAYERS,
  LR_OPTIONS,
  MC_RUNS,
  REG_STRENGTH_OPTIONS,
  REG_TYPES,
  SIGMA,
  WEIGHT_MODES,
} from '../../state/settings';
import { Field, IconButton, InfoTip, Panel, Segmented, Select, Slider } from '../controls/Controls';

const LAYER_TICKS = Array.from({ length: LAYERS.max - LAYERS.min + 1 }, (_, i) => LAYERS.min + i);
const LAYER_LABELS = [1, 5, 10, 15].filter((v) => v >= LAYERS.min && v <= LAYERS.max);

/** One labeled row of settings. */
function SettingsRow({ label, info, disabled, children }: { label: string; info?: ReactNode; disabled?: boolean; children: ReactNode }) {
  return (
    <div className={`settings-row${disabled ? ' disabled' : ''}`}>
      <div className="settings-row-label">
        {label}
        {info && <InfoTip>{info}</InfoTip>}
      </div>
      <div className="settings-row-fields">{children}</div>
    </div>
  );
}

export function SettingsPanel() {
  const s = useApp((st) => st.settings);
  const update = useApp((st) => st.update);
  const regression = s.task === 'regression';
  const [archOpen, setArchOpen] = useState(false);
  const archLabel = (o: { value: string; label: string }[], v: string) => o.find((x) => x.value === v)?.label ?? v;
  const archSummary = [
    s.numQubits === 2 ? archLabel(ENTANGLERS, s.entangler) : null,
    archLabel(DATA_ANGLES, s.dataAngles),
    `w ${archLabel(WEIGHT_MODES, s.weights).toLowerCase()}`,
  ]
    .filter(Boolean)
    .join(' · ');
  const archDefault =
    s.entangler === DEFAULT_SETTINGS.entangler && s.dataAngles === DEFAULT_SETTINGS.dataAngles && s.weights === DEFAULT_SETTINGS.weights;

  return (
    <Panel title="Settings" area="settings">
      <SettingsRow label="Model">
        <Field
          label="Qubits"
          info="One qubit: states live on the Bloch sphere. Two qubits: the circuit can create entanglement, shown in the Q-simplex."
        >
          <Segmented
            label="Number of qubits"
            value={s.numQubits}
            options={[
              { value: 1, label: '1 qubit' },
              { value: 2, label: '2 qubits' },
            ]}
            onChange={(v) => update({ numQubits: v })}
          />
        </Field>
        <Field
          label="Layers"
          info="Each layer re-uploads the input data through trainable rotations Rot(w·x + b). More layers make the model more expressive, like depth in a neural network."
        >
          <Slider
            label="Number of layers"
            value={s.numLayers}
            min={LAYERS.min}
            max={LAYERS.max}
            onChange={(v) => update({ numLayers: v })}
            ticks={LAYER_TICKS}
            tickLabels={LAYER_LABELS}
          />
        </Field>
        <Field label="Seed" info="Controls parameter initialization, data noise and shuffling. The same seed reproduces the same training run.">
          <div className="seed">
            <input
              className="input"
              type="number"
              min={0}
              aria-label="Random seed"
              value={s.seed}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isInteger(v) && v >= 0 && v < 2 ** 32) update({ seed: v });
              }}
            />
            <IconButton icon="dice" label="New random seed" onClick={() => update({ seed: Math.floor(Math.random() * 100000) })} />
          </div>
        </Field>
      </SettingsRow>

      <SettingsRow
        label="Architecture"
        info="Variations of the circuit. Each change re-initializes the model. The defaults are the original model from the paper."
      >
        <button
          type="button"
          className={`arch-toggle${archDefault ? '' : ' changed'}`}
          aria-expanded={archOpen}
          onClick={() => setArchOpen(!archOpen)}
        >
          <span>{archSummary}</span>
          <span aria-hidden="true">{archOpen ? '▴' : '▾'}</span>
        </button>
        {archOpen && (
          <>
            {s.numQubits === 2 && (
              <Field
                label="Entangler"
                info="Two-qubit gate after every layer. CNOT flips Q2 when Q1 is |1⟩; CZ flips the sign of |11⟩; None keeps the qubits independent (no entanglement)."
              >
                <Segmented
                  label="Entangling gate"
                  value={s.entangler}
                  options={ENTANGLERS}
                  onChange={(v) => update({ entangler: v })}
                />
              </Field>
            )}
            <Field
              label="Data on"
              info="All angles: φ, θ and ω each upload one feature as w·x + b. RY only: only θ uploads the data; φ and ω are trainable biases."
            >
              <Segmented label="Angles that upload data" value={s.dataAngles} options={DATA_ANGLES} onChange={(v) => update({ dataAngles: v })} />
            </Field>
            <Field
              label="Weights w"
              info="Fixed π: every data weight is w = π, so x ∈ [−1, 1] sweeps a full turn and the model is a Fourier series with integer frequencies in t = πx (see the spectrum for regression). Only the biases are trained."
            >
              <Segmented label="Data weights" value={s.weights} options={WEIGHT_MODES} onChange={(v) => update({ weights: v })} />
            </Field>
          </>
        )}
      </SettingsRow>

      <SettingsRow label="Training">
        <Field label="Learning rate" info="Step size of the Adam optimizer.">
          <Select label="Learning rate" value={s.lr} options={LR_OPTIONS.map((v) => ({ value: v, label: String(v) }))} onChange={(v) => update({ lr: v })} />
        </Field>
        <Field
          label="Regularization"
          info="Adds a penalty on the circuit parameters to the loss: L1 = λ·Σ|θ| pushes parameters to zero, L2 = λ·Σθ² keeps them small and smooths the learned function."
        >
          <Select label="Regularization type" value={s.regType} options={REG_TYPES} onChange={(v) => update({ regType: v })} />
        </Field>
        <Field label="Strength λ">
          <Select
            label="Regularization strength"
            value={s.regStrength}
            disabled={s.regType === 'none'}
            options={REG_STRENGTH_OPTIONS.map((v) => ({ value: v, label: String(v) }))}
            onChange={(v) => update({ regStrength: v })}
          />
        </Field>
      </SettingsRow>

      <SettingsRow
        label="Uncertainty"
        disabled={!regression}
        info={
          <>
            Monte Carlo estimate of predictive uncertainty: every inference run adds N(0, σ²) noise to all weights and biases.
            The band in the regression fit shows ±2 standard deviations.{!regression && ' Applies to regression tasks only.'}
          </>
        }
      >
        <Field label="Noise σ">
          <Slider
            label="Gaussian noise sigma"
            value={s.noiseSigma}
            min={SIGMA.min}
            max={SIGMA.max}
            step={SIGMA.step}
            format={(v) => v.toFixed(2)}
            onChange={(v) => update({ noiseSigma: v })}
            disabled={!regression}
          />
        </Field>
        <Field label="MC runs">
          <Slider label="Monte Carlo runs" value={s.mcRuns} min={MC_RUNS.min} max={MC_RUNS.max} onChange={(v) => update({ mcRuns: v })} disabled={!regression} />
        </Field>
        {!regression && <span className="settings-row-note muted small">regression only</span>}
      </SettingsRow>
    </Panel>
  );
}
