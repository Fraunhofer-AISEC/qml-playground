import { useMemo } from 'react';
import { useApp } from '../../state/store';
import type { ClassificationDataset } from '../../data/datasets';
import { residualsFigure, resultsFigure } from '../../viz/charts';
import { finalStateFigure } from '../../viz/stateSpace';
import { usePlot } from '../../viz/plotly';
import { Panel } from '../controls/Controls';
import { useFollowedSample, useResolvedTheme, useSampleEvents, useStateSpaceSpec, useStateHighlight, useThrottled } from '../hooks';

export function ResultsPanel() {
  const theme = useResolvedTheme();
  const info = useApp((s) => s.info);
  const snap = useThrottled(
    useApp((s) => s.snap),
    50,
  );
  const spec = useStateSpaceSpec();
  const events = useSampleEvents();
  const followed = useFollowedSample(snap);
  const hl = useStateHighlight(followed);
  const index = followed?.index ?? null;

  const finalFig = useMemo(() => (spec ? finalStateFigure(theme, spec, snap?.layerVis ?? null, hl) : null), [spec, theme, snap, hl]);
  const finalRef = usePlot(finalFig, events);

  const testFig = useMemo(() => {
    if (!info) return null;
    if (info.regression) return residualsFigure(theme, info.testX, info.testY, snap?.predictions ?? null, index);
    return resultsFigure(theme, info.dataset as ClassificationDataset, info.testX, info.testY, snap?.predictions ?? null, info.numClasses, index);
  }, [info, theme, snap, index]);
  const testRef = usePlot(testFig, events);

  const regression = info?.task === 'regression';
  const qubits = info?.numQubits ?? 1;

  return (
    <Panel title="Results" area="results">
      <h3 className="subheading">
        Final state
        <span className="muted"> · {qubits === 1 ? 'Bloch sphere' : 'Q-simplex'}</span>
      </h3>
      <div ref={finalRef} className="chart chart-final" />
      <h3 className="subheading">{regression ? 'Test residuals' : 'Test predictions'}</h3>
      <div ref={testRef} className="chart chart-results" />
    </Panel>
  );
}
