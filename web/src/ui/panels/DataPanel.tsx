import { useApp } from '../../state/store';
import { BATCH } from '../../state/settings';
import { Field, Panel, Segmented, Slider } from '../controls/Controls';
import { useResolvedTheme } from '../hooks';
import { DatasetThumbs } from './DatasetThumbs';

export function DataPanel() {
  const s = useApp((st) => st.settings);
  const update = useApp((st) => st.update);
  const theme = useResolvedTheme();

  return (
    <Panel title="Data" area="data">
      <Field
        label="Task"
        info="Classification: 2-D points with class labels. Regression: fit a 1-D Fourier series with the expectation value ⟨Z⟩ of the first qubit."
      >
        <Segmented
          label="Task type"
          value={s.task}
          options={[
            { value: 'classification', label: 'Classification' },
            { value: 'regression', label: 'Regression' },
          ]}
          onChange={(v) => update({ task: v })}
        />
      </Field>
      <Field label="Data set">
        <DatasetThumbs task={s.task} value={s.dataset as string} onChange={(v) => update({ dataset: v as typeof s.dataset })} theme={theme} />
      </Field>
      <Field label="Batch size" info="Samples per Adam step. An epoch is one pass over the shuffled training set.">
        <Slider label="Batch size" value={s.batchSize} min={BATCH.min} max={BATCH.max} onChange={(v) => update({ batchSize: v })} />
      </Field>
    </Panel>
  );
}
