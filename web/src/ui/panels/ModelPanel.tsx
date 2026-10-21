import { useMemo } from 'react';
import { useApp } from '../../state/store';
import { layerGridColumns, layerGridFigure, LAYER_ROW_HEIGHT } from '../../viz/stateSpace';
import { usePlot } from '../../viz/plotly';
import { CircuitView } from '../circuit/CircuitView';
import { Icon, IconButton, Panel } from '../controls/Controls';
import { useFollowedSample, useResolvedTheme, useSampleEvents, useStateHighlight, useStateSpaceSpec, useThrottled, useWidth } from '../hooks';

/** Full-width circuit pane: the interpretable model view. */
export function CircuitPanel() {
  const theme = useResolvedTheme();
  const info = useApp((s) => s.info);
  const snap = useThrottled(
    useApp((s) => s.snap),
    120,
  );
  const qubits = info?.numQubits ?? 1;
  const arch = info?.arch;
  const entangling =
    qubits === 2 && arch && arch.entangler !== 'none'
      ? `, followed by a ${arch.entangler === 'cz' ? 'CZ' : 'CNOT'} that can entangle the qubits`
      : '';
  const uploading =
    arch?.dataAngles === 'ry'
      ? ', where only the RY angle θ = w·x + b uploads the data and φ, ω are trainable biases'
      : ', with each angle set to w·x + b for one input feature';

  return (
    <Panel
      title="Circuit"
      area="circuit"
      actions={<FollowBadge />}
      info={
        <>
          Data re-uploading circuit: every layer applies Rot(φ, θ, ω) = RZ(ω)·RY(θ)·RZ(φ) on each qubit{uploading}
          {entangling}.{arch?.weights === 'fixed' && ' The data weights are fixed to w = π.'} Re-uploading the data in every
          layer makes the model more expressive, like depth in a neural network.
        </>
      }
    >
      {info ? <CircuitView info={info} snap={snap} theme={theme} /> : <div className="circuit-placeholder" />}
    </Panel>
  );
}

/** Which test sample is followed through the circuit, with a button to stop following it. */
function FollowBadge() {
  const info = useApp((s) => s.info);
  const sample = useApp((s) => s.sample);
  const prediction = useApp((s) => (s.sample && s.snap ? s.snap.predictions[s.sample.index] : NaN));
  const pinSample = useApp((s) => s.pinSample);
  if (!info) return null;
  if (!sample) return <span className="follow-hint muted">Hover a test point in Output or Results to follow it</span>;
  const i = sample.index;
  const y = info.testY[i];
  let text: string;
  if (info.task === 'regression') {
    text = `x = ${info.testX[i].toFixed(2)} · y = ${y.toFixed(2)}${Number.isNaN(prediction) ? '' : ` · ⟨Z⟩ = ${prediction.toFixed(2)}`}`;
  } else {
    const verdict = Number.isNaN(prediction) ? '' : ` → ${prediction} ${prediction === y ? '✓' : '✗'}`;
    text = `class ${y}${verdict}`;
  }
  return (
    <span className={`follow-badge${sample.pinned ? ' pinned' : ''}`} title={sample.pinned ? 'Pinned (click the point again or press Esc to release)' : 'Click the point to pin it'}>
      {sample.pinned && <Icon name="pin" size={12} />}
      <b>Sample #{i}</b>
      <span>{text}</span>
      {sample.pinned && <IconButton icon="close" label="Stop following this sample" variant="small" onClick={() => pinSample(null)} />}
    </span>
  );
}

/** States of the test samples after every layer, on the Bloch sphere or in the Q-simplex. */
export function LayerStatesPanel() {
  const theme = useResolvedTheme();
  const info = useApp((s) => s.info);
  const snap = useThrottled(
    useApp((s) => s.snap),
    120,
  );
  const spec = useStateSpaceSpec();
  const [wrapRef, width] = useWidth<HTMLDivElement>();

  const cols = spec ? layerGridColumns(spec.numLayers, width) : 2;
  const hl = useStateHighlight(useFollowedSample(snap));
  const layers = useMemo(() => (spec ? layerGridFigure(theme, spec, snap?.layerVis ?? null, cols, hl) : null), [spec, theme, snap, cols, hl]);
  const layersRef = usePlot(layers, useSampleEvents());
  const rows = spec ? Math.ceil((spec.numLayers - 1) / cols) : 0;
  const qubits = info?.numQubits ?? 1;

  return (
    <Panel
      title="Layer states"
      area="layers"
      info={
        qubits === 1
          ? 'Bloch sphere: each dot is one test sample after the given layer; colors are the true classes. The last layer is shown under Results.'
          : 'Q-simplex: position = basis-state probabilities, marker size = entanglement (tangle). Bell states lie on the bold edges. The last layer is shown under Results.'
      }
    >
      <div ref={wrapRef} className="layer-grid-wrap">
        {spec && spec.numLayers > 1 ? (
          <div ref={layersRef} className="chart" style={{ height: rows * LAYER_ROW_HEIGHT }} />
        ) : (
          <p className="muted center">With a single layer, the final state is shown under Results.</p>
        )}
      </div>
    </Panel>
  );
}
