import { useEffect, useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { Circuit, isDataAngle } from '../../sim/circuit';
import type { DataInfo, Snapshot, UnitMaps } from '../../worker/protocol';
import type { PlotTheme } from '../../viz/theme';
import {
  NEGATIVE,
  POSITIVE,
  UNIT_MODES,
  blochValue,
  classAt,
  classColor,
  cssColor,
  describeClass,
  divergingColor,
  isDiverging,
  outputValue,
  renderCurve,
  renderGrid,
  sequentialColor,
  tangleValue,
  unitPalette,
  type UnitMode,
} from '../../viz/unitMaps';
import { InfoTip, Segmented } from '../controls/Controls';
import { useFollowedSample, useWidth, type FollowedSample } from '../hooks';
import { GateCard, type GateRef } from './GateCard';

const SUB_GATES = ['RZ', 'RY', 'RZ'] as const;
const ANGLE_NAMES = ['φ', 'θ', 'ω'] as const;
const FEATURE_NAMES = ['x₁', 'x₂'];
const TAU = 2 * Math.PI;

/** Left margin of every row (qubit and bus labels). */
const PREFIX = 52;
/** Gap between the input tiles above the first row. */
const INPUT_GAP = 36;

interface Geometry {
  gateW: number;
  gateH: number;
  pitch: number;
  dialR: number;
  labels: boolean;
  tile: number;
  /** Width of one layer column. */
  colW: number;
  /** Width reserved for the measurement and output label at the end of the last row. */
  outW: number;
  /** Height of the input band above the first row. */
  inputBand: number;
  busY: (bottom: boolean, f: number) => number;
  wireY: (q: number) => number;
  height: number;
  /** x positions inside a row, for the i-th layer of that row. */
  gateX: (i: number, j: number) => number;
  entX: (i: number) => number;
  tileX: (i: number) => number;
  rowContentW: (layers: number) => number;
}

function geometry(numQubits: 1 | 2, numFeatures: number, compact: boolean): Geometry {
  const gateW = compact ? 30 : 42;
  const gateH = compact ? 30 : 50;
  const pitch = gateW + (compact ? 4 : 6);
  const tile = compact ? 40 : 48;
  const entW = numQubits === 2 ? 30 : 10;
  const colW = 3 * pitch + entW + tile + 20;
  const busGap = 12;
  const busBlock = 12 + numFeatures * busGap;
  const wire0 = busBlock + gateH / 2 + 14;
  const wireGap = Math.max(110, gateH + tile + 30);
  const wireY = (q: number) => wire0 + q * wireGap;
  const lastWire = wireY(numQubits - 1);
  // Bottom buses are stacked in reverse so their left-margin feeds nest without crossing.
  const busY = (bottom: boolean, f: number) =>
    bottom ? lastWire + gateH / 2 + 14 + (numFeatures - 1 - f) * busGap : 12 + f * busGap;
  const height = numQubits === 2 ? busY(true, 0) + 12 : lastWire + gateH / 2 + 22;
  return {
    gateW,
    gateH,
    pitch,
    dialR: compact ? 9 : 11,
    labels: !compact,
    tile,
    colW,
    outW: 96,
    inputBand: tile + 14,
    busY,
    wireY,
    height,
    gateX: (i, j) => 10 + i * colW + j * pitch,
    entX: (i) => 10 + i * colW + 3 * pitch + entW / 2 - 4,
    tileX: (i) => 10 + i * colW + 3 * pitch + entW + 6,
    rowContentW: (layers) => 10 + layers * colW,
  };
}

/** How many layers fit on one row of the given width (the circuit folds onto further rows). */
function layersPerRow(width: number, colW: number, outW: number): number {
  return Math.max(1, Math.floor((width - PREFIX - outW - 4) / colW));
}

/** Classical (double-line) wire carrying an input feature. */
function Bus({ d }: { d: string }) {
  return (
    <g fill="none">
      <path d={d} className="bus-outer" />
      <path d={d} className="bus-inner" />
    </g>
  );
}

/** Point at angle `a` (0 = up, clockwise) on a circle. */
const polar = (cx: number, cy: number, r: number, a: number) => [cx + r * Math.sin(a), cy - r * Math.cos(a)] as const;

function wedgePath(cx: number, cy: number, r: number, a0: number, a1: number): string {
  const [x0, y0] = polar(cx, cy, r, a0);
  const [x1, y1] = polar(cx, cy, r, a1);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M${cx},${cy} L${x0},${y0} A${r},${r} 0 ${large} 1 ${x1},${y1} Z`;
}

const fmt = (v: number, d = 2) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(d));
const wrapAngle = (a: number) => ((((a + Math.PI) % TAU) + TAU) % TAU) - Math.PI;

/**
 * The angle dial of one sub-gate: the needle is the bias b, the wedge is the range of angles w·x + b that
 * the data sweeps for x ∈ [-1, 1]. A full disc plus a badge means more than one full turn.
 */
function Dial({ cx, cy, r, w, b, sample }: { cx: number; cy: number; r: number; w: number; b: number; sample?: number }) {
  const marker = sample === undefined ? null : polar(cx, cy, r, sample);
  const span = Math.abs(w);
  const turns = span / Math.PI; // 2|w| / 2π
  const color = w >= 0 ? POSITIVE : NEGATIVE;
  const [nx, ny] = polar(cx, cy, r, b);
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="var(--surface)" stroke="var(--border-strong)" strokeWidth={1} />
      {turns >= 1 ? (
        <circle cx={cx} cy={cy} r={r} fill={color} fillOpacity={0.35} />
      ) : (
        span > 1e-3 && <path d={wedgePath(cx, cy, r, b - span, b + span)} fill={color} fillOpacity={0.4} />
      )}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="var(--text)" strokeWidth={1.6} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={1.6} fill="var(--text)" />
      {marker && <circle cx={marker[0]} cy={marker[1]} r={2.6} className="dial-sample" />}
      {turns >= 1 && (
        <text x={cx + r - 1} y={cy - r + 3} className="dial-badge" textAnchor="middle">
          ×{turns.toFixed(turns < 10 ? 1 : 0)}
        </text>
      )}
    </g>
  );
}

interface HoverState {
  kind: 'unit' | 'tangle' | 'input';
  l: number;
  q: number;
  /** Grid point index under the cursor. */
  p: number;
  /** Position relative to the wrapper. */
  left: number;
  top: number;
  image: string;
}

export function CircuitView({ info, snap, theme }: { info: DataInfo; snap: Snapshot | null; theme: PlotTheme }) {
  const [mode, setMode] = useState<UnitMode>('z');
  // Compact by default for deep circuits until the user chooses.
  const [compactChoice, setCompact] = useState<boolean | null>(null);
  const compact = compactChoice ?? info.numLayers > 8;
  const [hover, setHover] = useState<HoverState | null>(null);
  const [gate, setGate] = useState<(GateRef & { left: number; top: number }) | null>(null);
  const [rowsRef, width] = useWidth<HTMLDivElement>();
  const followed = useFollowedSample(snap);
  const [gradFlow, setGradFlow] = useState(false);

  useEffect(() => {
    if (!gate) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') setGate(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [gate]);
  // A new configuration may not have the selected gate any more.
  const [gateInfo, setGateInfo] = useState(info);
  if (gateInfo !== info) {
    setGateInfo(info);
    setGate(null);
  }

  const regression = info.task === 'regression';
  const Q = info.numQubits;
  const L = info.numLayers;
  const F = regression ? 1 : 2;
  const arch = info.arch;
  const featIdx = useMemo(() => new Circuit(Q, L, F).featIdx, [Q, L, F]);
  const g = geometry(Q, F, compact);
  const units = snap?.units ?? null;
  const theta = snap?.theta ?? null;
  const P = L * Q * 3;
  const pal = useMemo(() => unitPalette(theme), [theme]);
  const effectiveMode: UnitMode = mode === 'score' && regression ? 'z' : mode;
  const modeInfo = UNIT_MODES.find((m) => m.value === effectiveMode)!;

  const images = useMemo(() => renderUnitImages(units, info, effectiveMode, pal, theme), [units, info, effectiveMode, pal, theme]);
  const inputImages = useMemo(() => renderInputImages(regression, pal, theme), [regression, pal, theme]);

  // Fold the circuit onto as many rows as needed so it never scrolls horizontally.
  const perRow = layersPerRow(width, g.colW, g.outW);
  const rows: number[][] = [];
  for (let start = 0; start < L; start += perRow) {
    rows.push(Array.from({ length: Math.min(perRow, L - start) }, (_, i) => start + i));
  }

  const onTileMove = (e: MouseEvent<SVGImageElement>, kind: HoverState['kind'], l: number, q: number, image: string) => {
    if (!units) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = Math.min(0.999, Math.max(0, (e.clientX - rect.left) / rect.width));
    const fy = Math.min(0.999, Math.max(0, (e.clientY - rect.top) / rect.height));
    const n = units.size;
    const p = regression ? Math.floor(fx * n) : Math.floor((1 - fy) * n) * n + Math.floor(fx * n);
    const wrap = (e.currentTarget.closest('.circuit') as HTMLElement).getBoundingClientRect();
    const cardW = 176;
    // Open to the right of the tile, or to the left when there is no room.
    const right = rect.right - wrap.left + 8;
    const left = right + cardW > wrap.width ? rect.left - wrap.left - cardW - 8 : right;
    setHover({ kind, l, q, p, left: Math.max(0, left), top: rect.top - wrap.top, image });
  };

  const tileProps = (kind: HoverState['kind'], l: number, q: number, image: string) => ({
    onMouseMove: (e: MouseEvent<SVGImageElement>) => onTileMove(e, kind, l, q, image),
    onMouseLeave: () => setHover(null),
  });

  const featureName = (f: number) => (regression ? 'x' : FEATURE_NAMES[f]);

  /** Opens (or toggles) the detail card of a sub-gate below it. */
  const openGate = (e: MouseEvent<SVGGElement> | KeyboardEvent<SVGGElement>, ref: GateRef) => {
    if (gate && gate.l === ref.l && gate.q === ref.q && gate.j === ref.j) {
      setGate(null);
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const wrap = (e.currentTarget.closest('.circuit') as HTMLElement).getBoundingClientRect();
    const cardW = Math.min(320, wrap.width);
    const left = Math.min(wrap.width - cardW, Math.max(0, rect.left - wrap.left + rect.width / 2 - cardW / 2));
    setGate({ ...ref, left, top: rect.bottom - wrap.top + 6 });
  };

  /** Crosshair at the followed sample's position on a unit-map tile with `n` grid points per side. */
  const crosshair = (x: number, y: number, size: number, n: number) => {
    if (!followed) return null;
    // Grid values sit on the pixel centers of the rendered tile.
    const frac = (v: number) => (((v + 1) / 2) * (n - 1) + 0.5) / n;
    const px = x + frac(followed.x[0]) * size;
    if (regression) return <line x1={px} x2={px} y1={y} y2={y + size} className="sample-cross" />;
    const py = y + (1 - frac(followed.x[1])) * size;
    return (
      <g pointerEvents="none">
        <line x1={x} x2={x + size} y1={py} y2={py} className="sample-cross" />
        <line x1={px} x2={px} y1={y} y2={y + size} className="sample-cross" />
        <circle cx={px} cy={py} r={3} className="sample-ring" />
      </g>
    );
  };
  const unitN = units?.size ?? (regression ? 64 : 24);

  // Gradient flow: taps scale with |∂L/∂w|, gates with ‖(∂L/∂w, ∂L/∂b)‖, both relative to the largest one.
  const gradScale = useMemo(() => {
    const grad = snap?.grad;
    if (!grad) return null;
    let maxW = 0;
    let maxG = 0;
    for (let k = 0; k < P; k++) {
      maxW = Math.max(maxW, Math.abs(grad[k]));
      maxG = Math.max(maxG, Math.hypot(grad[k], grad[P + k]));
    }
    return { grad, maxW: maxW || 1, maxG: maxG || 1 };
  }, [snap, P]);
  const showGrad = gradFlow && gradScale !== null;

  /** Everything belonging to layer `l`, drawn as the i-th column of its row. */
  const layerColumn = (l: number, i: number) => (
    <g key={l}>
      {Array.from({ length: Q }, (_, q) =>
        SUB_GATES.map((name, j) => {
          const k = (l * Q + q) * 3 + j;
          const w = theta ? theta[k] : 0;
          const b = theta ? theta[P + k] : 0;
          const f = featIdx[k];
          const data = isDataAngle(j, arch.dataAngles);
          const fixed = arch.weights === 'fixed';
          const x = g.gateX(i, j);
          const cx = x + g.gateW / 2;
          const cy = g.wireY(q);
          const top = cy - g.gateH / 2;
          const bottom = top + g.gateH;
          const fromBottom = Q === 2 && q === 1;
          const busY = g.busY(fromBottom, f);
          const gw = gradScale ? gradScale.grad[k] : 0;
          const gb = gradScale ? gradScale.grad[P + k] : 0;
          const gateGrad = gradScale ? Math.hypot(gw, gb) / gradScale.maxG : 1;
          const strength = showGrad ? Math.sqrt(Math.abs(gw) / gradScale.maxW) : Math.min(1, Math.abs(w) / Math.PI);
          const color = showGrad ? 'var(--accent)' : w >= 0 ? POSITIVE : NEGATIVE;
          const vanishing = showGrad && gateGrad < 0.01;
          const title = !data
            ? `Layer ${l + 1}, Q${q + 1}: ${name}(${ANGLE_NAMES[j]}), ${ANGLE_NAMES[j]} = b (no data, trainable bias)\nb = ${fmt(b)} rad` +
              (gradScale ? `\n∂L/∂b = ${gb.toExponential(1)}` : '')
            : `Layer ${l + 1}, Q${q + 1}: ${name}(${ANGLE_NAMES[j]}), ${ANGLE_NAMES[j]} = ${fixed ? 'π' : 'w'}·${featureName(f)} + b\n` +
            `w = ${fmt(w)}, b = ${fmt(b)} rad\n` +
            (Math.abs(w) >= Math.PI
              ? `The data sweeps ${(Math.abs(w) / Math.PI).toFixed(1)} full turns (high-frequency encoding).`
              : `The data sweeps the angle from ${fmt(wrapAngle(b - Math.abs(w)))} to ${fmt(wrapAngle(b + Math.abs(w)))} rad.`) +
            (gradScale ? `\n∂L/∂w = ${gw.toExponential(1)}, ∂L/∂b = ${gb.toExponential(1)}` : '');
          return (
            <g key={`${q}-${j}`}>
              {data && (
                <>
                  <line
                    x1={cx}
                    x2={cx}
                    y1={busY}
                    y2={fromBottom ? bottom : top}
                    stroke={color}
                    strokeWidth={1 + 3.5 * strength}
                    strokeOpacity={0.25 + 0.75 * strength}
                  />
                  <circle cx={cx} cy={busY} r={1.5 + strength * 1.5} fill={color} />
                </>
              )}
              <g
                className={`gate${gate && gate.l === l && gate.q === q && gate.j === j ? ' selected' : ''}${vanishing ? ' vanishing' : ''}`}
                role="button"
                tabIndex={0}
                aria-label={`Layer ${l + 1}, Q${q + 1}, ${name}(${ANGLE_NAMES[j]}): details`}
                onClick={(e) => openGate(e, { l, q, j })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openGate(e, { l, q, j });
                  }
                }}
              >
                <title>{`${title}\nClick for details.`}</title>
                <rect x={x} y={top} width={g.gateW} height={g.gateH} rx={6} className="gate-box" />
                {g.labels && (
                  <text x={cx} y={top + 11} textAnchor="middle" className="gate-name">
                    {name}
                  </text>
                )}
                <Dial cx={cx} cy={g.labels ? cy + 1 : cy} r={g.dialR} w={w} b={b} sample={followed?.angles[k]} />
                {g.labels && data && (
                  <text x={cx} y={bottom - 4} textAnchor="middle" className="gate-feature">
                    {featureName(f)}
                  </text>
                )}
              </g>
            </g>
          );
        }),
      )}

      {Q === 2 && arch.entangler === 'cnot' && (
        <g className="cnot">
          <title>CNOT: flips Q2 when Q1 is |1⟩. This is the only gate that can entangle the qubits.</title>
          <line x1={g.entX(i)} x2={g.entX(i)} y1={g.wireY(0)} y2={g.wireY(1) + 9} className="wire" />
          <circle cx={g.entX(i)} cy={g.wireY(0)} r={4.5} fill="var(--text)" />
          <circle cx={g.entX(i)} cy={g.wireY(1)} r={9} fill="var(--surface)" className="wire" />
          <line x1={g.entX(i) - 9} x2={g.entX(i) + 9} y1={g.wireY(1)} y2={g.wireY(1)} className="wire" />
        </g>
      )}
      {Q === 2 && arch.entangler === 'cz' && (
        <g className="cnot">
          <title>CZ: flips the sign of |11⟩. Symmetric in both qubits; the only gate that can entangle them.</title>
          <line x1={g.entX(i)} x2={g.entX(i)} y1={g.wireY(0)} y2={g.wireY(1)} className="wire" />
          <circle cx={g.entX(i)} cy={g.wireY(0)} r={4.5} fill="var(--text)" />
          <circle cx={g.entX(i)} cy={g.wireY(1)} r={4.5} fill="var(--text)" />
        </g>
      )}

      {Array.from({ length: Q }, (_, q) => {
        const img = images?.unit[l * Q + q];
        const y = g.wireY(q) - g.tile / 2;
        return img ? (
          <g key={`u${q}`}>
            <image
              href={img}
              x={g.tileX(i)}
              y={y}
              width={g.tile}
              height={g.tile}
              preserveAspectRatio="none"
              className="unit-tile"
              {...tileProps('unit', l, q, img)}
            />
            <rect x={g.tileX(i)} y={y} width={g.tile} height={g.tile} className="unit-frame" pointerEvents="none" />
            {crosshair(g.tileX(i), y, g.tile, unitN)}
          </g>
        ) : null;
      })}
      {Q === 2 && images?.tangle[l] && (
        <g>
          <image
            href={images.tangle[l]}
            x={g.tileX(i) + 6}
            y={(g.wireY(0) + g.wireY(1)) / 2 - (g.tile - 12) / 2}
            width={g.tile - 12}
            height={g.tile - 12}
            preserveAspectRatio="none"
            className="unit-tile"
            {...tileProps('tangle', l, 0, images.tangle[l])}
          />
          <rect
            x={g.tileX(i) + 6}
            y={(g.wireY(0) + g.wireY(1)) / 2 - (g.tile - 12) / 2}
            width={g.tile - 12}
            height={g.tile - 12}
            className="unit-frame dashed"
            pointerEvents="none"
          />
          {crosshair(g.tileX(i) + 6, (g.wireY(0) + g.wireY(1)) / 2 - (g.tile - 12) / 2, g.tile - 12, unitN)}
        </g>
      )}
      <text
        x={g.gateX(i, 0) + (3 * g.pitch) / 2 - 3}
        y={Q === 2 ? (g.wireY(0) + g.wireY(1)) / 2 + 4 : g.height - 4}
        textAnchor="middle"
        className="circuit-label"
      >
        Layer {l + 1}
      </text>
    </g>
  );

  /** x of the vertical drop from input f down to its bus, in row coordinates. */
  const inputX = (f: number) => width / 2 - (F * g.tile + (F - 1) * INPUT_GAP) / 2 + f * (g.tile + INPUT_GAP);
  const dropX = (f: number) => inputX(f) + g.tile / 2;

  /** Input tiles centered above the first row, each dropping a bus onto its feature line. */
  const inputs = Array.from({ length: F }, (_, f) => {
    const x = inputX(f);
    const y = 4;
    return (
      <g key={`in${f}`}>
        <Bus d={`M${dropX(f)},${y + g.tile} V${g.inputBand + g.busY(false, f)}`} />
        <image
          href={inputImages[f]}
          x={x}
          y={y}
          width={g.tile}
          height={g.tile}
          className="unit-tile"
          preserveAspectRatio="none"
          {...tileProps('input', 0, f, inputImages[f])}
        />
        <rect x={x} y={y} width={g.tile} height={g.tile} className="unit-frame" pointerEvents="none" />
        {crosshair(x, y, g.tile, regression ? 64 : 24)}
        <text x={x - 6} y={y + g.tile / 2 + 4} textAnchor="end" className="circuit-label strong">
          {featureName(f)}
        </text>
      </g>
    );
  });

  /** A horizontal bus from x0 to x1, with gaps where a drop to a lower bus crosses it (first row). */
  const busLine = (y: number, x0: number, x1: number, f: number, first: boolean) => {
    const cuts = first ? Array.from({ length: F - f - 1 }, (_, k) => dropX(f + 1 + k)).filter((x) => x > x0 && x < x1) : [];
    const stops = [x0, ...cuts.flatMap((x) => [x - 5, x + 5]), x1];
    const d = stops.reduce((acc, x, k) => (k % 2 === 0 ? `${acc}M${x},${y}` : `${acc}H${x}`), '');
    return <Bus d={d} />;
  };

  /** Measurement meter at the end of a qubit wire. */
  const meter = (x: number, y: number, key: number) => (
    <g key={key} className="meter">
      <rect x={x} y={y - 10} width={24} height={20} rx={3} className="gate-box" />
      <path d={`M${x + 5},${y + 5} A7,7 0 0 1 ${x + 19},${y + 5}`} className="wire" fill="none" />
      <line x1={x + 12} y1={y + 5} x2={x + 17} y2={y - 4} className="wire" />
    </g>
  );

  const renderRow = (layers: number[], r: number) => {
    const first = r === 0;
    const last = r === rows.length - 1;
    const prefix = PREFIX;
    const contentEnd = prefix + g.rowContentW(layers.length);
    const top = first ? g.inputBand : 0;
    const wireEnd = last ? contentEnd + 4 : width - 18;
    const midY = Q === 2 ? (g.wireY(0) + g.wireY(1)) / 2 : g.wireY(0);
    return (
      <svg
        key={r}
        width={width}
        height={g.height + top}
        className="circuit-row"
        role="img"
        aria-label={`Circuit layers ${layers[0] + 1} to ${layers[layers.length - 1] + 1}`}
      >
        {first && inputs}
        <g transform={`translate(0,${top})`}>
          {/* Classical data buses */}
          {Array.from({ length: F }, (_, f) => {
            // The first row feeds the bottom buses (2 qubits) through nested connectors in the left margin.
            const xv = 8 + f * 7;
            return (
              <g key={`bus${f}`}>
                {busLine(g.busY(false, f), first && Q === 2 ? xv : prefix, contentEnd - 10, f, first)}
                {Q === 2 && <Bus d={`M${first ? xv : prefix},${first ? g.busY(false, f) : g.busY(true, f)} ${first ? `V${g.busY(true, f)}` : ''} H${contentEnd - 10}`} />}
                {!first && (
                  <>
                    <text x={prefix - 4} y={g.busY(false, f) + 3.5} textAnchor="end" className="circuit-label small">
                      {featureName(f)}
                    </text>
                    {Q === 2 && (
                      <text x={prefix - 4} y={g.busY(true, f) + 3.5} textAnchor="end" className="circuit-label small">
                        {featureName(f)}
                      </text>
                    )}
                  </>
                )}
              </g>
            );
          })}
          {/* Qubit wires, with continuation markers between rows */}
          {Array.from({ length: Q }, (_, q) => (
            <g key={`w${q}`}>
              <line x1={prefix - 18} x2={wireEnd} y1={g.wireY(q)} y2={g.wireY(q)} className="wire" />
              {first ? (
                <>
                  <text x={prefix - 2} y={g.wireY(q) - 7} textAnchor="end" className="circuit-label strong">
                    Q{q + 1}
                  </text>
                  <text x={prefix - 2} y={g.wireY(q) + 15} textAnchor="end" className="circuit-label">
                    |0⟩
                  </text>
                </>
              ) : (
                <text x={prefix - 22} y={g.wireY(q) + 4} textAnchor="end" className="circuit-label strong">
                  ↪ Q{q + 1}
                </text>
              )}
              {!last && (
                <text x={width - 14} y={g.wireY(q) + 4} className="circuit-label strong">
                  ↩
                </text>
              )}
            </g>
          ))}
          <g transform={`translate(${prefix},0)`}>{layers.map((l, i) => layerColumn(l, i))}</g>
          {last && (
            <g>
              <title>
                {regression
                  ? 'Measurement: ⟨Z⟩ of Q1 is the model output, shown large in the Output panel.'
                  : 'Measurement: the class scores are read out here; the prediction is shown large in the Output panel.'}
              </title>
              {Array.from({ length: Q }, (_, q) => meter(wireEnd, g.wireY(q), q))}
              <text x={wireEnd + 32} y={midY + 4} className="circuit-label strong">
                Output →
              </text>
            </g>
          )}
        </g>
      </svg>
    );
  };

  return (
    <div className="circuit">
      <div className="circuit-toolbar">
        <div className="field-label">
          Unit maps
          <InfoTip>
            Like the neurons in TensorFlow Playground, each tile shows one quantity over the whole input space after a
            layer. {modeInfo.description} {Q === 2 && 'The dashed tile between the wires shows the entanglement (tangle) of the two qubits.'}
          </InfoTip>
        </div>
        <Segmented
          label="Unit map quantity"
          value={effectiveMode}
          options={UNIT_MODES.filter((m) => !(regression && m.value === 'score')).map((m) => ({ value: m.value, label: m.label }))}
          onChange={setMode}
        />
        <Legend mode={effectiveMode} pal={pal} numClasses={info.numClasses} />
        <label className="toggle" title="Tap width = |∂L/∂w|; gates with a near-zero gradient are dimmed">
          <input type="checkbox" checked={gradFlow} onChange={(e) => setGradFlow(e.target.checked)} /> Gradient flow
        </label>
        <label className="toggle">
          <input type="checkbox" checked={compact} onChange={(e) => setCompact(e.target.checked)} /> Compact
        </label>
      </div>

      <div ref={rowsRef} className="circuit-rows">
        {width > 0 && rows.map(renderRow)}
      </div>
      {hover && units && <HoverCard hover={hover} units={units} info={info} mode={effectiveMode} />}
      {gate && <GateDetails gate={gate} info={info} snap={snap} featIdx={featIdx} followed={followed} onClose={() => setGate(null)} />}

      {showGrad ? (
        <p className="legend-note muted">
          <b>Gradient flow</b>: <span style={{ color: 'var(--accent)' }}>tap width</span> = |∂L/∂w| relative to the largest one; dimmed
          gates have less than 1% of the largest gradient (full training set).
        </p>
      ) : (
        <p className="legend-note muted">
          <b>Dial</b>: needle = bias b, wedge = angles w·x + b swept by the data (<span style={{ color: POSITIVE }}>w &gt; 0</span>,{' '}
          <span style={{ color: NEGATIVE }}>w &lt; 0</span>, ×n = full turns). <b>Tap width</b> = |w|.
          <InfoTip>
            Each dial shows one rotation angle of a gate. The needle points at the bias b, and the shaded wedge is the range of
            angles w·x + b the data sweeps for x ∈ [−1, 1]. A full disc with ×n means the data sweeps n full turns, a
            high-frequency encoding. The width of a data tap shows |w|, i.e. how strongly that feature is uploaded.
          </InfoTip>
        </p>
      )}
    </div>
  );
}

/** Gate card for the selected sub-gate, fed from the snapshot and the training data. */
function GateDetails({
  gate,
  info,
  snap,
  featIdx,
  followed,
  onClose,
}: {
  gate: GateRef & { left: number; top: number };
  info: DataInfo;
  snap: Snapshot | null;
  featIdx: Int32Array;
  followed: FollowedSample | null;
  onClose: () => void;
}) {
  const regression = info.task === 'regression';
  const F = regression ? 1 : 2;
  const P = info.numLayers * info.numQubits * 3;
  const k = (gate.l * info.numQubits + gate.q) * 3 + gate.j;
  const f = featIdx[k];
  const values = useMemo(() => info.trainX.filter((_, i) => i % F === f), [info, F, f]);
  return (
    <GateCard
      gate={gate}
      numQubits={info.numQubits}
      numLayers={info.numLayers}
      feature={regression ? 'x' : FEATURE_NAMES[f]}
      values={values}
      w={snap ? snap.theta[k] : 0}
      b={snap ? snap.theta[P + k] : 0}
      dataAngle={isDataAngle(gate.j, info.arch.dataAngles)}
      fixedWeight={info.arch.weights === 'fixed'}
      gw={snap ? snap.grad[k] : null}
      gb={snap ? snap.grad[P + k] : null}
      sampleAngle={followed ? followed.angles[k] : null}
      readoutIsZ={regression || info.numQubits === 2}
      left={gate.left}
      top={gate.top}
      onClose={onClose}
    />
  );
}

function Legend({ mode, pal, numClasses }: { mode: UnitMode; pal: ReturnType<typeof unitPalette>; numClasses: number }) {
  if (mode === 'score') {
    return (
      <div className="unit-legend" aria-label="Class colors">
        {pal.classes.slice(0, numClasses).map((c, i) => (
          <span key={i} className="unit-legend-chip">
            <i style={{ background: cssColor(c) }} /> {i}
          </span>
        ))}
      </div>
    );
  }
  const div = isDiverging(mode);
  const stops = div ? [-1, -0.5, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];
  const gradient = stops.map((v) => cssColor(div ? divergingColor(pal, v) : sequentialColor(pal, v)));
  return (
    <div className="unit-legend">
      <span>{div ? '−1' : mode === 'purity' ? '0.5' : '0'}</span>
      <i className="unit-legend-bar" style={{ background: `linear-gradient(to right, ${gradient.join(',')})` }} />
      <span>1</span>
    </div>
  );
}

function HoverCard({ hover, units, info, mode }: { hover: HoverState; units: UnitMaps; info: DataInfo; mode: UnitMode }) {
  const regression = info.task === 'regression';
  const n = units.size;
  const px = regression ? -1 + (2 * hover.p) / (n - 1) : -1 + (2 * (hover.p % n)) / (n - 1);
  const py = regression ? 0 : -1 + (2 * Math.floor(hover.p / n)) / (n - 1);
  const where = regression ? `x = ${px.toFixed(2)}` : `x₁ = ${px.toFixed(2)}, x₂ = ${py.toFixed(2)}`;
  let title = '';
  let value = '';
  if (hover.kind === 'input') {
    title = regression ? 'Input x' : `Input ${FEATURE_NAMES[hover.q]}`;
    value = regression ? where : `${FEATURE_NAMES[hover.q]} = ${(hover.q === 0 ? px : py).toFixed(2)}`;
  } else if (hover.kind === 'tangle') {
    title = `Entanglement after layer ${hover.l + 1}`;
    value = `tangle = ${tangleValue(units, hover.l, hover.p).toFixed(3)}`;
  } else if (mode === 'score') {
    title = `Q${hover.q + 1} after layer ${hover.l + 1}: prediction if measured here`;
    if (regression) {
      value = `⟨Z⟩ = ${outputValue(units, hover.l, hover.p).toFixed(3)}`;
    } else {
      const { cls, confidence } = classAt(units, info.numClasses, hover.l, hover.p);
      value = describeClass(cls, confidence);
    }
  } else {
    const label = UNIT_MODES.find((m) => m.value === mode)!.label;
    title = `Q${hover.q + 1} after layer ${hover.l + 1}`;
    value = `${label} = ${blochValue(units, info.numQubits, hover.l, hover.q, hover.p, mode as 'x' | 'y' | 'z' | 'purity').toFixed(3)}`;
  }
  return (
    <div className="unit-hover" style={{ left: hover.left, top: hover.top }} role="tooltip">
      <div className="unit-hover-title">{title}</div>
      <img src={hover.image} alt="" width={150} height={150} />
      <div>{value}</div>
      {hover.kind !== 'input' && <div className="muted">{where}</div>}
    </div>
  );
}

interface UnitImages {
  unit: string[];
  tangle: string[];
}

function renderUnitImages(
  units: UnitMaps | null,
  info: DataInfo,
  mode: UnitMode,
  pal: ReturnType<typeof unitPalette>,
  theme: PlotTheme,
): UnitImages | null {
  if (!units) return null;
  const Q = info.numQubits;
  const L = info.numLayers;
  const regression = info.task === 'regression';
  const C = info.numClasses;
  const render = (value: (p: number) => number, diverging: boolean) =>
    regression
      ? renderCurve(value, units.points, pal, diverging, theme.text)
      : renderGrid(units.size, (p) => (diverging ? divergingColor(pal, value(p)) : sequentialColor(pal, value(p))));
  const classMap = (l: number) =>
    renderGrid(units.size, (p) => {
      const { cls, confidence } = classAt(units, C, l, p);
      return classColor(pal, cls, confidence);
    });

  const unit: string[] = [];
  for (let l = 0; l < L; l++) {
    for (let q = 0; q < Q; q++) {
      if (mode === 'score') unit.push(classMap(l));
      else if (mode === 'purity') unit.push(render((p) => 2 * blochValue(units, Q, l, q, p, 'purity') - 1, false));
      else unit.push(render((p) => blochValue(units, Q, l, q, p, mode), true));
    }
  }
  const tangle = Q === 2 ? Array.from({ length: L }, (_, l) => render((p) => tangleValue(units, l, p), false)) : [];
  return { unit, tangle };
}

function renderInputImages(regression: boolean, pal: ReturnType<typeof unitPalette>, theme: PlotTheme): string[] {
  const n = 24;
  if (regression) return [renderCurve((p) => -1 + (2 * p) / 63, 64, pal, true, theme.text)];
  return [
    renderGrid(n, (p) => divergingColor(pal, -1 + (2 * (p % n)) / (n - 1))),
    renderGrid(n, (p) => divergingColor(pal, -1 + (2 * Math.floor(p / n)) / (n - 1))),
  ];
}
