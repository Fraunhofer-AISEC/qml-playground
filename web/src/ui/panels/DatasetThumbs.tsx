import { useMemo } from 'react';
import { datasetsFor, fourierCurve, labelPoint, type ClassificationDataset, type TaskType } from '../../data/datasets';
import type { PlotTheme } from '../../viz/theme';
import { classColor, renderGrid, unitPalette } from '../../viz/unitMaps';

const THUMB_GRID = 40;

/** Classification thumbnail: the true class regions, like the dataset icons in TF Playground. */
function classThumb(name: ClassificationDataset, theme: PlotTheme): string {
  const pal = unitPalette(theme);
  const n = THUMB_GRID;
  return renderGrid(n, (p) => {
    const x = -1 + (2 * (p % n) + 1) / n;
    const y = -1 + (2 * Math.floor(p / n) + 1) / n;
    return classColor(pal, labelPoint(name, x, y), 0.75);
  });
}

/** Regression thumbnail: the target Fourier series as an SVG path in a 100×100 box. */
function curvePath(name: string): string {
  const [x, y] = fourierCurve(name, 64);
  return Array.from(x, (xv, i) => `${i ? 'L' : 'M'}${((xv + 1) * 50).toFixed(1)},${(50 - y[i] * 38).toFixed(1)}`).join(' ');
}

/** Dataset picker: one clickable preview per dataset of the current task. */
export function DatasetThumbs({
  task,
  value,
  onChange,
  theme,
}: {
  task: TaskType;
  value: string;
  onChange: (name: string) => void;
  theme: PlotTheme;
}) {
  const entries = Object.entries(datasetsFor(task));
  const images = useMemo(
    () =>
      Object.fromEntries(
        Object.keys(datasetsFor(task)).map((name) => [
          name,
          task === 'regression' ? curvePath(name) : classThumb(name as ClassificationDataset, theme),
        ]),
      ),
    [task, theme],
  );
  const hint = task === 'regression' ? ' — training samples leave gaps at x ≈ 0 and x > 0.8 to probe generalization' : '';

  return (
    <div className="thumbs" role="radiogroup" aria-label="Data set">
      {entries.map(([name, label]) => (
        <button
          key={name}
          type="button"
          role="radio"
          aria-checked={name === value}
          aria-label={label}
          title={label + hint}
          className={`thumb${name === value ? ' active' : ''}`}
          onClick={() => onChange(name)}
        >
          {task === 'regression' ? (
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <line x1="0" x2="100" y1="50" y2="50" className="thumb-axis" />
              <path d={images[name]} className="thumb-curve" />
            </svg>
          ) : (
            <img src={images[name]} alt="" />
          )}
        </button>
      ))}
    </div>
  );
}
