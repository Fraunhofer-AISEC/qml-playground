import { useMemo, useState } from 'react';
import { useApp } from '../../state/store';
import { layerHeatmapFigure, metricsFigure } from '../../viz/charts';
import { usePlot } from '../../viz/plotly';
import { Panel, Segmented } from '../controls/Controls';
import { useResolvedTheme, useThrottled } from '../hooks';

const fmt = (v: number | undefined, digits = 3) => (v === undefined || Number.isNaN(v) ? '–' : v.toFixed(digits));

type Tab = 'curves' | 'grad' | 'tangle';

export function MetricsPanel() {
  const theme = useResolvedTheme();
  const metrics = useThrottled(
    useApp((s) => s.metrics),
    100,
  );
  const task = useApp((s) => s.settings.task);
  const snap = useApp((s) => s.snap);
  const viewEpoch = snap && snap.epoch !== snap.latestEpoch ? snap.epoch : null;
  const regression = task === 'regression';
  const twoQubits = useApp((s) => s.settings.numQubits === 2);
  const [tabChoice, setTab] = useState<Tab>('curves');
  const tab = tabChoice === 'tangle' && !twoQubits ? 'curves' : tabChoice;
  const figure = useMemo(
    () => (tab === 'curves' ? metricsFigure(theme, metrics, regression, viewEpoch) : layerHeatmapFigure(theme, metrics, tab, viewEpoch)),
    [theme, metrics, regression, viewEpoch, tab],
  );
  const ref = usePlot(figure);

  const i = snap ? snap.epoch - metrics.base : -1;
  const at = (a: number[]) => (i >= 0 && i < a.length ? a[i] : undefined);

  return (
    <Panel
      title="Metrics"
      area="metrics"
      info={
        <>
          {regression ? 'Curves: mean squared error between ⟨Z⟩ and the targets.' : 'Curves: training loss and classification accuracy on the training and test sets.'}{' '}
          Gradients: mean gradient norm ‖∂L/∂θ‖ of each layer per epoch (log scale, without regularization). Dark rows that
          stay dark are layers that barely learn, e.g. vanishing gradients.
          {twoQubits && ' Entanglement: mean tangle of the test samples after each layer, showing when and where the model starts to use entanglement.'}
        </>
      }
      actions={
        <Segmented
          label="Metrics view"
          value={tab}
          options={[
            { value: 'curves', label: 'Curves' },
            { value: 'grad', label: 'Gradients' },
            ...(twoQubits ? [{ value: 'tangle' as const, label: 'Entangl.' }] : []),
          ]}
          onChange={setTab}
        />
      }
    >
      <div ref={ref} className="chart chart-metrics" />
      <dl className="stats">
        {regression ? (
          <>
            <div>
              <dt>Train MSE</dt>
              <dd>{fmt(at(metrics.train), 4)}</dd>
            </div>
            <div>
              <dt>Test MSE</dt>
              <dd>{fmt(at(metrics.test), 4)}</dd>
            </div>
          </>
        ) : (
          <>
            <div>
              <dt>Loss</dt>
              <dd>{fmt(at(metrics.loss))}</dd>
            </div>
            <div>
              <dt>Train acc</dt>
              <dd>{fmt(at(metrics.train))}</dd>
            </div>
            <div>
              <dt>Test acc</dt>
              <dd>{fmt(at(metrics.test))}</dd>
            </div>
          </>
        )}
      </dl>
    </Panel>
  );
}
