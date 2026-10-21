import { useMemo, useState } from 'react';
import { useApp } from '../../state/store';
import { GRID_SIZES } from '../../state/settings';
import type { RegressionDataset } from '../../data/datasets';
import { regressionFitFigure, spectrumFigure } from '../../viz/charts';
import { modelSpectrum, targetSpectrum } from '../../viz/fourier';
import { usePlot } from '../../viz/plotly';
import { CLASS_COLORS } from '../../viz/theme';
import { classScoreColor, cssColor, unitPalette } from '../../viz/unitMaps';
import { InfoTip, Panel, Select } from '../controls/Controls';
import { useFollowedSample, useResolvedTheme, useThrottled } from '../hooks';
import { DecisionMap } from '../output/DecisionMap';

/** The model's output over the input space: the end of the circuit, shown large as in TF Playground. */
export function OutputPanel() {
  const theme = useResolvedTheme();
  const info = useApp((s) => s.info);
  const snap = useThrottled(
    useApp((s) => s.snap),
    50,
  );
  const gridSize = useApp((s) => s.settings.gridSize);
  const update = useApp((s) => s.update);
  const regression = info?.task === 'regression';
  const followed = useFollowedSample(snap);
  const fx = followed && regression ? followed.x[0] : null;
  const fy = followed && regression ? followed.prediction : null;

  const fit = useMemo(() => {
    if (!info?.regression) return null;
    const r = info.regression;
    const hl = fx !== null && fy !== null && !Number.isNaN(fy) ? { x: fx, y: fy } : null;
    return regressionFitFigure(theme, r.denseX, r.denseY, snap?.curve ?? null, info.trainX, info.trainY, hl);
  }, [info, theme, snap, fx, fy]);
  const fitRef = usePlot(fit);

  // Fourier spectrum of the model output vs. the target (regression).
  const target = useMemo(() => (info?.regression ? targetSpectrum(info.dataset) : null), [info]);
  const theta = snap?.theta ?? null;
  const spectrum = useMemo(() => {
    if (!info?.regression || !target) return null;
    const model = theta ? modelSpectrum(info.numQubits, info.numLayers, info.dataset as RegressionDataset, theta, info.arch) : null;
    return spectrumFigure(theme, target, model);
  }, [info, target, theta, theme]);
  const spectrumRef = usePlot(spectrum);

  // Classification: predicted class (null) or the score of one class.
  const [view, setView] = useState<number | null>(null);
  const C = info?.numClasses ?? 0;
  const shownView = view !== null && view < C ? view : null;
  const oneQubit = info?.numQubits === 1;
  const scoreLabel = (c: number) => (oneQubit ? `fidelity with Ψ${c}` : `p(class ${c})`);
  const pal = useMemo(() => unitPalette(theme), [theme]);

  return (
    <Panel
      title="Output"
      area="output"
      info={
        regression
          ? 'Model output ⟨Z⟩ of qubit 1 over x, with a ±2σ Monte Carlo uncertainty band (see Settings → Uncertainty), the ground truth and the training samples.'
          : 'Model output over the whole input space: the predicted class color, faded where the model is unsure. The line is the decision boundary (p = 0.5 for two classes). Dots are the test samples in their true class colors: hover one to follow it through the circuit, click to pin it (Esc clears). The selector shows a single class score instead: the class probability (2 qubits) or the fidelity with the class target state Ψc (1 qubit).'
      }
      actions={
        !regression && (
          <div className="panel-actions">
            <Select
              label="Shown quantity"
              value={shownView === null ? -1 : shownView}
              options={[
                { value: -1, label: 'Predicted class' },
                ...Array.from({ length: C }, (_, c) => ({ value: c, label: oneQubit ? `Fidelity Ψ${c}` : `p(class ${c})` })),
              ]}
              onChange={(v) => setView(v < 0 ? null : v)}
            />
            <Select
              label="Decision grid resolution"
              value={gridSize}
              options={GRID_SIZES.map((g) => ({ value: g, label: `${g}×${g}` }))}
              onChange={(v) => update({ gridSize: v })}
            />
          </div>
        )
      }
    >
      {info &&
        (regression ? (
          <>
            <div ref={fitRef} className="chart chart-output" />
            <h3 className="subheading">
              Fourier spectrum
              <InfoTip>
                A data re-uploading model is a Fourier series in its inputs: each layer that uploads x with weight w can add
                frequencies ±w. More layers and larger |w| give access to higher frequencies. Bars show |c_k| of the model
                output (without MC noise) and of the target for t = πx; non-integer w spreads the model's weight over
                neighbouring k.
              </InfoTip>
            </h3>
            <div ref={spectrumRef} className="chart chart-spectrum" />
          </>
        ) : (
          <>
            <DecisionMap
              info={info}
              scores={snap?.decision ?? null}
              n={snap?.gridSize ?? gridSize}
              theme={theme}
              followed={followed?.index ?? null}
              predictions={snap?.predictions ?? null}
              view={shownView}
              scoreLabel={scoreLabel}
            />
            <div className="unit-legend">
              {shownView === null ? (
                CLASS_COLORS.slice(0, info.numClasses).map((c, i) => (
                  <span key={i} className="unit-legend-chip">
                    <i style={{ background: c }} /> class {i}
                  </span>
                ))
              ) : (
                <span className="unit-legend-chip">
                  {scoreLabel(shownView)}: 0
                  <i
                    className="unit-legend-bar"
                    style={{
                      background: `linear-gradient(to right, ${cssColor(classScoreColor(pal, shownView, 0))}, ${cssColor(classScoreColor(pal, shownView, 1))})`,
                    }}
                  />
                  1
                </span>
              )}
              <span className="unit-legend-chip">
                <i className="legend-line" /> decision boundary
              </span>
            </div>
          </>
        ))}
    </Panel>
  );
}
