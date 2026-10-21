import { useMemo, useState, type MouseEvent } from 'react';
import { useApp } from '../../state/store';
import type { DataInfo } from '../../worker/protocol';
import { decisionBoundaries } from '../../viz/contour';
import type { PlotTheme } from '../../viz/theme';
import { classColor, classFromScores, classScoreColor, cssColor, describeClass, renderGrid, unitPalette } from '../../viz/unitMaps';
import { useWidth } from '../hooks';

const M = { left: 24, right: 6, top: 6, bottom: 22 };
const TICKS = [-1, 0, 1];
/** Hover radius (px) for picking a test sample. */
const PICK_R = 7;

interface Hover {
  x: number;
  y: number;
  left: number;
  top: number;
  title: string;
  text: string;
}

/**
 * The model output over the whole input space, like TF Playground's output map: the predicted class color,
 * faded where the model is unsure, the decision boundary as a line, and the test samples in their true colors.
 */
export function DecisionMap({
  info,
  scores,
  n,
  theme,
  followed,
  predictions,
  view,
  scoreLabel,
}: {
  info: DataInfo;
  scores: Float32Array | null;
  n: number;
  theme: PlotTheme;
  /** Index of the followed test sample. */
  followed: number | null;
  /** Predicted class per test sample. */
  predictions: Float32Array | null;
  /** null: predicted class faded by confidence; c: the score of class c. */
  view: number | null;
  /** Name of a class score, e.g. "p(class 1)". */
  scoreLabel: (c: number) => string;
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<Hover | null>(null);
  const hoverSample = useApp((s) => s.hoverSample);
  const pinSample = useApp((s) => s.pinSample);
  const C = info.numClasses;
  const pal = useMemo(() => unitPalette(theme), [theme]);
  const ready = scores !== null && scores.length === n * n * C;

  const image = useMemo(
    () =>
      ready
        ? renderGrid(n, (p) => {
            if (view !== null) return classScoreColor(pal, view, scores[p * C + view]);
            const { cls, confidence } = classFromScores(scores, p * C, C);
            return classColor(pal, cls, confidence);
          })
        : null,
    [ready, scores, n, C, pal, view],
  );
  const segments = useMemo(() => (ready ? decisionBoundaries(scores, n, C) : []), [ready, scores, n, C]);

  const size = Math.max(0, Math.min(width, 520) - M.left - M.right);
  const sx = (v: number) => M.left + ((v + 1) / 2) * size;
  const sy = (v: number) => M.top + ((1 - v) / 2) * size;
  // Grid values sit on the cell centers of the image, so it extends half a cell beyond [-1, 1].
  const half = size / (n - 1) / 2;
  const path = segments.map(([x0, y0, x1, y1]) => `M${sx(x0).toFixed(1)},${sy(y0).toFixed(1)}L${sx(x1).toFixed(1)},${sy(y1).toFixed(1)}`).join('');

  const tx = info.testX;
  const ty = info.testY;

  /** The test sample within PICK_R pixels of the cursor, if any. */
  const pick = (px: number, py: number): number | null => {
    let best: number | null = null;
    let bestD = PICK_R * PICK_R;
    for (let i = 0; i < ty.length; i++) {
      const d = (sx(tx[2 * i]) - px) ** 2 + (sy(tx[2 * i + 1]) - py) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };

  const cursor = (e: MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const fy = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    return { fx, fy, i: pick(M.left + fx * size, M.top + fy * size) };
  };

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    if (!ready) return;
    const { fx, fy, i } = cursor(e);
    // Keep following the last sample while the cursor stays on the map.
    if (i !== null) hoverSample(i);
    const left = M.left + fx * size;
    const place = { left: left + 180 > width ? left - 170 : left + 12, top: M.top + fy * size + 12 };
    if (i !== null) {
      const pred = predictions ? predictions[i] : NaN;
      setHover({
        x: tx[2 * i],
        y: tx[2 * i + 1],
        ...place,
        title: `Test sample #${i}`,
        text: `class ${ty[i]}${Number.isNaN(pred) ? '' : ` → predicted ${pred}${pred === ty[i] ? ' ✓' : ' ✗'}`}`,
      });
      return;
    }
    const c = Math.round(fx * (n - 1));
    const r = Math.round((1 - fy) * (n - 1));
    const p = r * n + c;
    const { cls, confidence } = classFromScores(scores, p * C, C);
    const text =
      view === null ? describeClass(cls, confidence) : `${scoreLabel(view)} = ${scores[p * C + view].toFixed(3)} · predicted class ${cls}`;
    setHover({ x: -1 + 2 * fx, y: 1 - 2 * fy, ...place, title: 'Model output', text });
  };

  return (
    <div ref={wrapRef} className="decision-map">
      {width > 0 && (
        <svg width={size + M.left + M.right} height={size + M.top + M.bottom} role="img" aria-label="Model output over the input space">
          <defs>
            <clipPath id="decision-clip">
              <rect x={M.left} y={M.top} width={size} height={size} />
            </clipPath>
          </defs>
          <rect x={M.left} y={M.top} width={size} height={size} fill={cssColor(pal.neutral)} />
          {image && (
            <image
              href={image}
              x={M.left - half}
              y={M.top - half}
              width={size + 2 * half}
              height={size + 2 * half}
              preserveAspectRatio="none"
              clipPath="url(#decision-clip)"
            />
          )}
          <g clipPath="url(#decision-clip)">
            <path d={path} className="decision-boundary" />
          </g>
          <g>
            {Array.from({ length: ty.length }, (_, i) => (
              <circle key={i} cx={sx(tx[2 * i])} cy={sy(tx[2 * i + 1])} r={2.4} fill={cssColor(pal.classes[ty[i]])} className="decision-point" />
            ))}
          </g>
          {followed !== null && followed < ty.length && (
            <circle cx={sx(tx[2 * followed])} cy={sy(tx[2 * followed + 1])} r={6.5} className="sample-ring" />
          )}
          <rect x={M.left} y={M.top} width={size} height={size} className="decision-frame" />
          {TICKS.map((v) => (
            <g key={v} className="decision-tick">
              <text x={sx(v)} y={M.top + size + 12} textAnchor="middle">
                {v}
              </text>
              <text x={M.left - 5} y={sy(v) + 3.5} textAnchor="end">
                {v}
              </text>
            </g>
          ))}
          <text x={M.left + size} y={M.top + size + 20} textAnchor="end" className="decision-tick axis">
            x₁
          </text>
          <text x={4} y={M.top + 10} className="decision-tick axis">
            x₂
          </text>
          <rect
            x={M.left}
            y={M.top}
            width={size}
            height={size}
            fill="transparent"
            className="unit-tile"
            onMouseMove={onMove}
            onMouseLeave={() => {
              setHover(null);
              hoverSample(null);
            }}
            onClick={(e) => pinSample(cursor(e).i)}
          />
        </svg>
      )}
      {hover && (
        <div className="unit-hover" style={{ left: hover.left, top: hover.top }} role="tooltip">
          <div className="unit-hover-title">{hover.title}</div>
          <div>{hover.text}</div>
          <div className="muted">
            x₁ = {hover.x.toFixed(2)}, x₂ = {hover.y.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}
