import type { Figure, Layout, Trace } from './plotly';
import { CLASS_COLORS, baseLayout, type PlotTheme } from './theme';

/** Static description of what the state-space plots show for one configuration. */
export interface StateSpaceSpec {
  numQubits: 1 | 2;
  numLayers: number;
  /** Sample indices of the test set grouped by color (class, or a single group for regression). */
  groups: Int32Array[];
  /** Color index per group. */
  groupColors: number[];
  /** Bloch vectors of single-qubit targets (classification only). */
  targetsBloch: number[][];
  /** Two-qubit classification: number of classes (basis states |00⟩.. used as targets); 0 for regression. */
  numClasses: number;
}

const AXIS = {
  showgrid: false,
  zeroline: false,
  showline: false,
  showticklabels: false,
  showbackground: false,
  showspikes: false,
  ticks: '',
  title: { text: '' },
};

const SPHERE = (() => {
  const n = 25;
  const x: number[][] = [], y: number[][] = [], z: number[][] = [];
  for (let i = 0; i < n; i++) {
    const u = (2 * Math.PI * i) / (n - 1);
    x.push([]);
    y.push([]);
    z.push([]);
    for (let j = 0; j < n; j++) {
      const v = (Math.PI * j) / (n - 1);
      x[i].push(Math.cos(u) * Math.sin(v));
      y[i].push(Math.sin(u) * Math.sin(v));
      z[i].push(Math.cos(v));
    }
  }
  return { x, y, z, ones: z.map((r) => r.map(() => 1)) };
})();

/** Bloch sphere background (port of `make_bloch_sphere`). */
function blochBackground(t: PlotTheme, scene: string, spec: StateSpaceSpec, withArrows: boolean) {
  const contour = (start: number, end: number, size: number) => ({ start, end, size, show: true, color: t.accentStrong });
  const traces: Trace[] = [
    {
      type: 'surface',
      scene,
      ...SPHERE,
      surfacecolor: SPHERE.ones,
      opacity: 0.18,
      colorscale: [
        [0, t.accent],
        [1, t.accent],
      ],
      showscale: false,
      hoverinfo: 'skip',
      contours: { x: contour(0, 0.001, 0.001), y: contour(0, 0.001, 0.001), z: contour(-1, 1, 0.25) },
      lighting: { ambient: 1, diffuse: 0, specular: 0 },
    },
  ];
  const ends = [
    [1.2, 0, 0],
    [0, 1.2, 0],
    [0, 0, 1.2],
  ];
  for (const d of ends) {
    traces.push({
      type: 'scatter3d',
      scene,
      mode: 'lines',
      x: [-d[0], d[0]],
      y: [-d[1], d[1]],
      z: [-d[2], d[2]],
      line: { color: t.muted, width: 1.5 },
      hoverinfo: 'skip',
    });
    if (withArrows) {
      traces.push({
        type: 'cone',
        scene,
        x: [d[0]],
        y: [d[1]],
        z: [d[2]],
        u: [d[0] / 6],
        v: [d[1] / 6],
        w: [d[2] / 6],
        anchor: 'tail',
        sizemode: 'absolute',
        sizeref: 0.12,
        colorscale: [
          [0, t.muted],
          [1, t.muted],
        ],
        showscale: false,
        hoverinfo: 'skip',
      });
    }
  }
  const annotations = [
    { x: 0, y: 0, z: 1.4, text: '|0⟩' },
    { x: 0, y: 0, z: -1.4, text: '|1⟩' },
    { x: 1.4, y: 0, z: 0, text: '|+⟩' },
    { x: -1.4, y: 0, z: 0, text: '|−⟩' },
  ].map((a) => ({ ...a, showarrow: false, font: { color: t.text, size: 12 } }));
  spec.targetsBloch.forEach((v, n) => {
    annotations.push({
      x: 1.1 * v[0],
      y: 1.1 * v[1],
      z: 1.1 * v[2],
      text: `Ψ${n}`,
      showarrow: false,
      font: { color: CLASS_COLORS[n], size: 14 },
    });
  });
  return { traces, annotations };
}

// Vertices of the Q-simplex (port of `make_tetrahedron`): coordinates are (p|01⟩, p|10⟩, p|00⟩).
const TETRA = [
  [0, 0, 1], // |00⟩
  [1, 0, 0], // |01⟩
  [0, 1, 0], // |10⟩
  [0, 0, 0], // |11⟩
];

function tetraBackground(t: PlotTheme, scene: string, spec: StateSpaceSpec) {
  const traces: Trace[] = [
    {
      type: 'mesh3d',
      scene,
      x: TETRA.map((v) => v[0]),
      y: TETRA.map((v) => v[1]),
      z: TETRA.map((v) => v[2]),
      i: [0, 0, 0, 1],
      j: [1, 2, 3, 2],
      k: [2, 3, 1, 3],
      opacity: 0.15,
      color: t.accent,
      hoverinfo: 'skip',
    },
  ];
  const edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
  ];
  for (const [a, b] of edges) {
    const bell = (a === 0 && b === 3) || (a === 1 && b === 2);
    traces.push({
      type: 'scatter3d',
      scene,
      mode: 'lines',
      x: [TETRA[a][0], TETRA[b][0]],
      y: [TETRA[a][1], TETRA[b][1]],
      z: [TETRA[a][2], TETRA[b][2]],
      opacity: 0.7,
      line: { color: t.accent, width: bell ? 4 : 2 },
      hoverinfo: 'skip',
    });
  }
  const isTarget = (i: number) => spec.numClasses === 0 || i < spec.numClasses;
  const annotations: Record<string, unknown>[] = TETRA.map((v, i) => ({
    x: v[0],
    y: v[1],
    z: v[2],
    text: `|${i.toString(2).padStart(2, '0')}⟩`,
    showarrow: false,
    font: { color: isTarget(i) && spec.numClasses ? CLASS_COLORS[i] : t.text, size: isTarget(i) ? 14 : 12 },
  }));
  const mid = (a: number, b: number) => TETRA[a].map((c, k) => (c + TETRA[b][k]) / 2);
  for (const [label, p] of [
    ['|Φ±⟩', mid(0, 3)],
    ['|Ψ±⟩', mid(1, 2)],
  ] as const) {
    annotations.push({ x: p[0], y: p[1], z: p[2], text: label, showarrow: false, font: { color: t.accent, size: 12 } });
  }
  return { traces, annotations };
}

function background(t: PlotTheme, scene: string, spec: StateSpaceSpec, large: boolean) {
  return spec.numQubits === 1 ? blochBackground(t, scene, spec, large) : tetraBackground(t, scene, spec);
}

/** The followed sample's path through the state space (see `sampleTrace`). */
export interface StateHighlight {
  path: number[][];
  layerEnd: number[];
  /** Color of the sample (its true class). */
  color: string;
}

/**
 * Two traces for the followed sample up to the end of `layer`: its path from |0⟩ and its current state.
 * Always emitted (empty without a sample) so trace indices stay stable across updates.
 */
function highlightTraces(t: PlotTheme, scene: string, hl: StateHighlight | undefined, layer: number, markerScale: number): Trace[] {
  const pts = hl ? hl.path.slice(0, hl.layerEnd[layer] + 1) : [];
  const end = pts.length ? [pts[pts.length - 1]] : [];
  const col = (p: number[][], c: number) => p.map((v) => v[c]);
  return [
    {
      type: 'scatter3d',
      scene,
      mode: 'lines',
      x: col(pts, 0),
      y: col(pts, 1),
      z: col(pts, 2),
      line: { color: t.text, width: 4 * markerScale },
      opacity: 0.85,
      hoverinfo: 'skip',
    },
    {
      type: 'scatter3d',
      scene,
      mode: 'markers',
      x: col(end, 0),
      y: col(end, 1),
      z: col(end, 2),
      marker: { size: 8 * markerScale, color: hl?.color ?? t.text, line: { color: t.text, width: 2 } },
      hoverinfo: 'skip',
    },
  ];
}

/** One scatter3d trace per color group for the given layer (port of the clientside state updates). */
function stateTraces(
  scene: string,
  spec: StateSpaceSpec,
  vis: Float32Array | null,
  layer: number,
  markerScale: number,
): Trace[] {
  const L = spec.numLayers;
  return spec.groups.map((idx, g) => {
    const n = vis ? idx.length : 0;
    const x = new Array<number>(n), y = new Array<number>(n), z = new Array<number>(n);
    const size = spec.numQubits === 2 ? new Array<number>(n) : 2.5 * markerScale;
    for (let s = 0; s < n; s++) {
      const o = (idx[s] * L + layer) * 4;
      x[s] = vis![o];
      y[s] = vis![o + 1];
      z[s] = vis![o + 2];
      // Marker size encodes the tangle (entanglement), as in `make_tetrahedron_state_traces`.
      if (Array.isArray(size)) size[s] = (3 + 7 * vis![o + 3]) * markerScale;
    }
    return {
      type: 'scatter3d',
      scene,
      mode: 'markers',
      x,
      y,
      z,
      marker: { size, color: CLASS_COLORS[spec.groupColors[g]], opacity: spec.numQubits === 2 ? 0.7 : 0.6, line: { width: 0 } },
      // Test-sample index, for following a sample on hover.
      customdata: vis ? Array.from(idx) : [],
      hovertemplate:
        spec.numQubits === 2 ? 'p|01⟩=%{x:.2f}<br>p|10⟩=%{y:.2f}<br>p|00⟩=%{z:.2f}<extra></extra>' : 'x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
    };
  });
}

function sceneLayout(spec: StateSpaceSpec, annotations: unknown[], small: boolean, domain?: { x: number[]; y: number[] }) {
  const camera =
    spec.numQubits === 1
      ? { up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0 }, eye: small ? { x: 1.15, y: 1.15, z: 0.8 } : { x: 1.25, y: 1.25, z: 0.85 } }
      : { eye: small ? { x: 1.5, y: 1.15, z: 0.45 } : { x: 1.6, y: 1.25, z: 0.35 } };
  return {
    xaxis: AXIS,
    yaxis: AXIS,
    zaxis: AXIS,
    annotations,
    camera,
    aspectmode: 'cube',
    bgcolor: 'rgba(0,0,0,0)',
    ...(domain ? { domain } : {}),
  };
}

/** Final-layer state plot (`make_state_space_plot`). */
export function finalStateFigure(t: PlotTheme, spec: StateSpaceSpec, vis: Float32Array | null, hl?: StateHighlight): Figure {
  const bg = background(t, 'scene', spec, true);
  return {
    data: [
      ...bg.traces,
      ...stateTraces('scene', spec, vis, spec.numLayers - 1, 1),
      ...highlightTraces(t, 'scene', hl, spec.numLayers - 1, 1),
    ],
    layout: { ...baseLayout(t), uirevision: 'final', scene: sceneLayout(spec, bg.annotations, false) },
  };
}

export const LAYER_ROW_HEIGHT = 250;

export function layerGridColumns(numLayers: number, width: number): number {
  const panels = numLayers - 1;
  if (panels <= 1) return 1;
  return width >= 760 ? 3 : 2;
}

/**
 * States after every layer except the last (which is the final-state plot), one scene per layer
 * (`make_state_space_model_plot`).
 */
export function layerGridFigure(
  t: PlotTheme,
  spec: StateSpaceSpec,
  vis: Float32Array | null,
  cols: number,
  hl?: StateHighlight,
): Figure | null {
  const panels = spec.numLayers - 1;
  if (panels < 1) return null;
  const rows = Math.ceil(panels / cols);
  const data: Trace[] = [];
  const layout: Layout = { ...baseLayout(t), uirevision: `layers-${spec.numLayers}-${cols}`, height: rows * LAYER_ROW_HEIGHT };
  const titles: Record<string, unknown>[] = [];
  for (let l = 0; l < panels; l++) {
    const scene = l === 0 ? 'scene' : `scene${l + 1}`;
    const r = Math.floor(l / cols);
    const c = l % cols;
    const domain = {
      x: [c / cols + 0.005, (c + 1) / cols - 0.005],
      y: [1 - (r + 1) / rows + 0.02, 1 - r / rows - 0.06 / rows],
    };
    const bg = background(t, scene, spec, false);
    data.push(...bg.traces, ...stateTraces(scene, spec, vis, l, 0.8), ...highlightTraces(t, scene, hl, l, 0.8));
    layout[scene] = sceneLayout(spec, bg.annotations, true, domain);
    titles.push({
      text: `Layer ${l + 1}`,
      x: (c + 0.5) / cols,
      y: 1 - r / rows,
      xref: 'paper',
      yref: 'paper',
      xanchor: 'center',
      yanchor: 'top',
      showarrow: false,
      font: { size: 12, color: t.muted },
    });
  }
  layout.annotations = titles;
  return { data, layout };
}
