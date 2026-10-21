import { useEffect, useMemo, useRef, useState } from 'react';
import { Circuit } from '../sim/circuit';
import { useApp } from '../state/store';
import { CLASS_COLORS, PLOT_THEMES, type PlotTheme, type ResolvedTheme } from '../viz/theme';
import type { StateHighlight, StateSpaceSpec } from '../viz/stateSpace';
import { sampleTrace, type SampleTrace } from '../viz/trajectory';
import type { Snapshot } from '../worker/protocol';
import type { PlotEvents } from '../viz/plotly';

/** Returns `value`, updated at most every `ms` milliseconds (trailing edge included). */
export function useThrottled<T>(value: T, ms: number): T {
  const [out, setOut] = useState(value);
  const last = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    const now = performance.now();
    const wait = ms - (now - last.current);
    clearTimeout(timer.current);
    if (wait <= 0) {
      last.current = now;
      setOut(value);
    } else {
      timer.current = setTimeout(() => {
        last.current = performance.now();
        setOut(value);
      }, wait);
    }
    return () => clearTimeout(timer.current);
  }, [value, ms]);
  return out;
}

function systemTheme(): ResolvedTheme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** Resolves the theme mode, applies it to <html data-theme>, and returns the matching plot colors. */
export function useResolvedTheme(): PlotTheme {
  const mode = useApp((s) => s.themeMode);
  const [system, setSystem] = useState<ResolvedTheme>(systemTheme);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!mq) return;
    const onChange = () => setSystem(mq.matches ? 'dark' : 'light');
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  const resolved = mode === 'system' ? system : mode;
  useEffect(() => {
    document.documentElement.dataset.theme = resolved;
  }, [resolved]);
  return PLOT_THEMES[resolved];
}

/** Test-sample grouping and targets for the state-space plots of the current configuration. */
export function useStateSpaceSpec(): StateSpaceSpec | null {
  const info = useApp((s) => s.info);
  return useMemo(() => {
    if (!info) return null;
    const n = info.testY.length;
    let groups: Int32Array[];
    let groupColors: number[];
    if (info.task === 'regression') {
      groups = [Int32Array.from({ length: n }, (_, i) => i)];
      groupColors = [1];
    } else {
      const buckets: number[][] = Array.from({ length: info.numClasses }, () => []);
      info.testY.forEach((c, i) => buckets[c].push(i));
      groups = buckets.map((b) => Int32Array.from(b));
      groupColors = buckets.map((_, c) => c);
    }
    return {
      numQubits: info.numQubits,
      numLayers: info.numLayers,
      groups,
      groupColors,
      targetsBloch: info.task === 'classification' && info.numQubits === 1 ? info.targetsBloch : [],
      numClasses: info.task === 'classification' ? info.numClasses : 0,
    };
  }, [info]);
}

/** Width of an element, tracked with a ResizeObserver. */
export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export interface FollowedSample extends SampleTrace {
  index: number;
  pinned: boolean;
  /** Input features of the sample. */
  x: number[];
  /** True label (class index or regression target). */
  label: number;
  /** Model prediction in `snap` (class index or ⟨Z⟩), or NaN without a snapshot. */
  prediction: number;
}

/** The followed test sample, re-run through the circuit with the parameters of `snap` (see `sampleTrace`). */
export function useFollowedSample(snap: Snapshot | null): FollowedSample | null {
  const info = useApp((s) => s.info);
  const sample = useApp((s) => s.sample);
  const featIdx = useMemo(
    () => (info ? new Circuit(info.numQubits, info.numLayers, info.task === 'regression' ? 1 : 2).featIdx : null),
    [info],
  );
  const theta = snap?.theta ?? null;
  const prediction = sample && snap ? snap.predictions[sample.index] : NaN;
  return useMemo(() => {
    if (!info || !featIdx || !theta || !sample || sample.index >= info.testY.length) return null;
    const F = info.task === 'regression' ? 1 : 2;
    const x = Array.from(info.testX.subarray(sample.index * F, (sample.index + 1) * F));
    return {
      ...sampleTrace(info.numQubits, info.numLayers, featIdx, theta, x, 8, info.arch.entangler),
      index: sample.index,
      pinned: sample.pinned,
      x,
      label: info.testY[sample.index],
      prediction,
    };
  }, [info, featIdx, theta, sample, prediction]);
}

/** Plot events that follow (hover) or pin (click) the test sample whose index is the point's customdata. */
export function useSampleEvents(): PlotEvents {
  const hoverSample = useApp((s) => s.hoverSample);
  const pinSample = useApp((s) => s.pinSample);
  return useMemo(
    () => ({
      onHover: (d) => hoverSample(Number(d)),
      onUnhover: () => hoverSample(null),
      onClick: (d) => pinSample(Number(d)),
    }),
    [hoverSample, pinSample],
  );
}

/** The followed sample as a state-space highlight, colored by its true class. */
export function useStateHighlight(followed: FollowedSample | null): StateHighlight | undefined {
  const regression = useApp((s) => s.info?.task === 'regression');
  return useMemo(
    () =>
      followed
        ? { path: followed.path, layerEnd: followed.layerEnd, color: CLASS_COLORS[regression ? 1 : followed.label] }
        : undefined,
    [followed, regression],
  );
}
