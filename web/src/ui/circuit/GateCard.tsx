import { useMemo } from 'react';
import { NEGATIVE, POSITIVE } from '../../viz/unitMaps';
import { IconButton } from '../controls/Controls';

const TAU = 2 * Math.PI;
const BINS = 36;
const SIZE = 132;
const R_IN = 22;
const R_OUT = 58;

const GATE_NAMES = ['RZ', 'RY', 'RZ'] as const;
const ANGLE_NAMES = ['φ', 'θ', 'ω'] as const;

export interface GateRef {
  l: number;
  q: number;
  j: number;
}

/** Point at angle `a` (0 = up, clockwise) on a circle, as on the circuit dials. */
const polar = (r: number, a: number) => [SIZE / 2 + r * Math.sin(a), SIZE / 2 - r * Math.cos(a)] as const;

/** Annulus sector between radii r0 < r1 and angles a0 < a1. */
function sectorPath(r0: number, r1: number, a0: number, a1: number): string {
  const [x0, y0] = polar(r1, a0);
  const [x1, y1] = polar(r1, a1);
  const [x2, y2] = polar(r0, a1);
  const [x3, y3] = polar(r0, a0);
  return `M${x0},${y0} A${r1},${r1} 0 0 1 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 0 0 ${x3},${y3} Z`;
}

const wrap = (a: number) => ((((a + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2));
const fmtGrad = (v: number) => (v === 0 ? '0' : Math.abs(v) < 1e-3 ? v.toExponential(1) : v.toFixed(4));

function roleText(j: number, lastLayer: boolean, readoutIsZ: boolean): string {
  if (j === 1) {
    return 'Rotates the state about the Y axis of the Bloch sphere: moves amplitude between |0⟩ and |1⟩, i.e. tilts the state toward the poles or the equator.';
  }
  const base =
    j === 0
      ? 'Rotates the state about the Z axis: changes only the relative phase of |0⟩ and |1⟩, which the following RY turns into a change of probabilities.'
      : 'Rotates the state about the Z axis again: sets the phase with which the state enters the next layer.';
  return j === 2 && lastLayer && readoutIsZ
    ? `${base} In the last layer it cannot change the measured probabilities, so its gradient is zero.`
    : base;
}

/**
 * Details of one sub-gate: its parameters and their full-batch gradients, a circular histogram of the angles
 * w·x + b the training data produce, and what the rotation does.
 */
export function GateCard({
  gate,
  numQubits,
  numLayers,
  feature,
  values,
  dataAngle,
  fixedWeight,
  w,
  b,
  gw,
  gb,
  sampleAngle,
  readoutIsZ,
  left,
  top,
  onClose,
}: {
  gate: GateRef;
  numQubits: number;
  numLayers: number;
  /** Name of the uploaded feature (x₁, x₂ or x). */
  feature: string;
  /** That feature's value for every training sample. */
  values: Float64Array;
  /** The angle uploads the data (otherwise it is a pure bias). */
  dataAngle: boolean;
  /** The data weight is fixed to π (not trained). */
  fixedWeight: boolean;
  w: number;
  b: number;
  gw: number | null;
  gb: number | null;
  /** Angle of the followed sample, if any. */
  sampleAngle: number | null;
  /** The read-out depends only on computational-basis probabilities (⟨Z⟩ or 2-qubit class probabilities). */
  readoutIsZ: boolean;
  left: number;
  top: number;
  onClose: () => void;
}) {
  const { l, q, j } = gate;
  const angle = ANGLE_NAMES[j];
  const color = w >= 0 ? POSITIVE : NEGATIVE;

  const bins = useMemo(() => {
    const counts = new Array<number>(BINS).fill(0);
    for (let i = 0; i < values.length; i++) {
      const a = wrap(w * values[i] + b);
      counts[Math.min(BINS - 1, Math.floor(((a + Math.PI) / TAU) * BINS))]++;
    }
    return counts;
  }, [values, w, b]);
  const max = Math.max(1, ...bins);
  const [nx, ny] = polar(R_IN - 3, b);

  return (
    <div className="gate-card" style={{ left, top }} role="dialog" aria-label={`Gate details, layer ${l + 1}, qubit ${q + 1}`}>
      <div className="gate-card-head">
        <b>
          {GATE_NAMES[j]}({angle})
        </b>
        <span className="muted">
          Layer {l + 1} of {numLayers}
          {numQubits === 2 && ` · Q${q + 1}`}
        </span>
        <IconButton icon="close" label="Close gate details" variant="small" onClick={onClose} />
      </div>
      <div className="gate-card-body">
        <svg width={SIZE} height={SIZE} role="img" aria-label={`Histogram of ${angle} over the training data`}>
          <circle cx={SIZE / 2} cy={SIZE / 2} r={R_OUT} className="gate-card-ring" />
          <circle cx={SIZE / 2} cy={SIZE / 2} r={R_IN} className="gate-card-ring" />
          {bins.map((c, k) =>
            c ? (
              <path
                key={k}
                d={sectorPath(R_IN, R_IN + ((R_OUT - R_IN) * c) / max, -Math.PI + (k * TAU) / BINS, -Math.PI + ((k + 1) * TAU) / BINS)}
                fill={color}
                fillOpacity={0.7}
              />
            ) : null,
          )}
          <line x1={SIZE / 2} y1={SIZE / 2} x2={nx} y2={ny} stroke="var(--text)" strokeWidth={1.6} strokeLinecap="round" />
          <circle cx={SIZE / 2} cy={SIZE / 2} r={1.8} fill="var(--text)" />
          {sampleAngle !== null && (() => {
            const [sx, sy] = polar(R_OUT + 4, sampleAngle);
            return <circle cx={sx} cy={sy} r={3.5} className="dial-sample" />;
          })()}
          {[
            [0, '0'],
            [Math.PI / 2, 'π/2'],
            [Math.PI, '±π'],
            [-Math.PI / 2, '−π/2'],
          ].map(([a, t]) => {
            const [x, y] = polar(R_OUT + 9, a as number);
            return (
              <text key={t} x={x} y={y + 3} textAnchor="middle" className="gate-card-tick">
                {t}
              </text>
            );
          })}
        </svg>
        <dl className="gate-card-values">
          <dt>angle</dt>
          <dd>
            {dataAngle ? `${angle} = ${fixedWeight ? 'π' : 'w'}·${feature} + b` : `${angle} = b (no data)`}
          </dd>
          {dataAngle && (
            <>
              <dt>w</dt>
              <dd>{fixedWeight ? 'π (fixed)' : fmt(w)}</dd>
            </>
          )}
          <dt>b</dt>
          <dd>{fmt(b)} rad</dd>
          {dataAngle && !fixedWeight && (
            <>
              <dt>∂L/∂w</dt>
              <dd>{gw === null ? '–' : fmtGrad(gw)}</dd>
            </>
          )}
          <dt>∂L/∂b</dt>
          <dd>{gb === null ? '–' : fmtGrad(gb)}</dd>
          {sampleAngle !== null && (
            <>
              <dt>sample</dt>
              <dd>
                {angle} = {fmt(wrap(sampleAngle))} rad
              </dd>
            </>
          )}
        </dl>
      </div>
      <p className="gate-card-note">{roleText(j, l === numLayers - 1, readoutIsZ)}</p>
      <p className="gate-card-note muted">
        Bars: angles of all training samples (needle = b{sampleAngle !== null ? ', dot = followed sample' : ''}). Gradients are
        over the full training set, incl. regularization; Adam moves w and b against them.
      </p>
    </div>
  );
}
