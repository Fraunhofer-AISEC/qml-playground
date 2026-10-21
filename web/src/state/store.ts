import { create } from 'zustand';
import type { DataInfo, FromWorker, RestoreState, Snapshot, ToWorker } from '../worker/protocol';
import { defaultDataset, structuralChanged, type Settings } from './settings';
import { decodeHash, encodeHash } from './urlState';
import type { MetricsSeries } from '../viz/charts';

export type ThemeMode = 'system' | 'light' | 'dark';

/** The test sample followed through the circuit ("follow a sample"). */
export interface SampleFocus {
  index: number;
  /** Pinned by a click: hovering other samples does not change it. */
  pinned: boolean;
}

interface AppState {
  settings: Settings;
  playing: boolean;
  busy: boolean;
  info: DataInfo | null;
  snap: Snapshot | null;
  metrics: MetricsSeries;
  error: string | null;
  themeMode: ThemeMode;
  sample: SampleFocus | null;

  update(patch: Partial<Settings>): void;
  togglePlay(): void;
  step(): void;
  reset(): void;
  fastForward(epochs: number): void;
  seek(epoch: number): void;
  resumeFrom(epoch: number): void;
  goLive(): void;
  setThemeMode(mode: ThemeMode): void;
  shareUrl(withParams: boolean): string;
  /** Follows test sample `index` while hovering (ignored while a sample is pinned); null stops following. */
  hoverSample(index: number | null): void;
  /** Pins test sample `index` (toggles if it is already pinned); null clears the selection. */
  pinSample(index: number | null): void;
}

const worker = new Worker(new URL('../worker/trainer.worker.ts', import.meta.url), { type: 'module' });
const post = (msg: ToWorker) => worker.postMessage(msg);

const initial = decodeHash(window.location.hash);
const emptyMetrics = (base = 0): MetricsSeries => ({ base, loss: [], train: [], test: [], gradNorm: [], tangle: [] });

function readThemeMode(): ThemeMode {
  try {
    const v = localStorage.getItem('qmlpg-theme');
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {
    /* storage unavailable */
  }
  return 'light';
}

export const useApp = create<AppState>((set, get) => ({
  settings: initial.settings,
  playing: false,
  busy: false,
  info: null,
  snap: null,
  metrics: emptyMetrics(),
  error: null,
  themeMode: readThemeMode(),
  sample: null,

  update(patch) {
    const prev = get().settings;
    const next = { ...prev, ...patch };
    if (patch.task && patch.task !== prev.task && !patch.dataset) next.dataset = defaultDataset(patch.task);
    set({ settings: next });
    if (structuralChanged(prev, next)) configure(next);
    else post({ type: 'hyper', settings: next });
  },
  togglePlay() {
    post({ type: get().playing ? 'pause' : 'play' });
  },
  step() {
    post({ type: 'step' });
  },
  reset() {
    configure(get().settings);
  },
  fastForward(epochs) {
    post({ type: 'fastForward', epochs });
  },
  seek(epoch) {
    post({ type: 'seek', epoch });
  },
  resumeFrom(epoch) {
    post({ type: 'resumeFrom', epoch });
  },
  goLive() {
    post({ type: 'live' });
  },
  setThemeMode(mode) {
    try {
      localStorage.setItem('qmlpg-theme', mode);
    } catch {
      /* storage unavailable */
    }
    set({ themeMode: mode });
  },
  shareUrl(withParams) {
    const { settings, snap } = get();
    const restore = withParams && snap ? { theta: snap.theta, epoch: snap.epoch } : undefined;
    const url = new URL(window.location.href);
    url.hash = encodeHash(settings, restore);
    return url.toString();
  },
  hoverSample(index) {
    const cur = get().sample;
    if (cur?.pinned || (cur?.index ?? null) === index) return;
    set({ sample: index === null ? null : { index, pinned: false } });
  },
  pinSample(index) {
    const cur = get().sample;
    const unpin = index === null || (cur?.pinned && cur.index === index);
    set({ sample: unpin ? null : { index, pinned: true } });
  },
}));

function configure(settings: Settings, restore?: RestoreState) {
  useApp.setState({ info: null, snap: null, sample: null, metrics: emptyMetrics(restore?.epoch ?? 0), error: null });
  post({ type: 'configure', settings, restore });
}

/** Applies an incremental metrics chunk (append, or truncate-then-append after a resume). */
function mergeMetrics(current: MetricsSeries, chunk: Snapshot['metrics']): MetricsSeries {
  const keep = Math.max(0, chunk.from - chunk.base);
  const sameBase = current.base === chunk.base;
  const cut = <T,>(a: T[]): T[] => (sameBase ? a.slice(0, keep) : []);
  return {
    base: chunk.base,
    loss: cut(current.loss).concat(chunk.loss),
    train: cut(current.train).concat(chunk.train),
    test: cut(current.test).concat(chunk.test),
    gradNorm: cut(current.gradNorm).concat(chunk.gradNorm),
    tangle: cut(current.tangle).concat(chunk.tangle),
  };
}

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'data':
      useApp.setState({ info: msg.info });
      return;
    case 'snapshot': {
      const { info, metrics } = useApp.getState();
      if (!info || msg.snap.configId !== info.configId) return; // stale snapshot from a previous configuration
      useApp.setState({ snap: msg.snap, metrics: mergeMetrics(metrics, msg.snap.metrics) });
      return;
    }
    case 'status':
      useApp.setState({ playing: msg.playing, busy: msg.busy });
      return;
    case 'error':
      useApp.setState({ error: msg.message });
      return;
  }
};

// Keep the URL hash in sync with the settings (without trained parameters), debounced.
let urlTimer: ReturnType<typeof setTimeout> | undefined;
useApp.subscribe((s, prev) => {
  if (s.settings === prev.settings) return;
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => history.replaceState(null, '', encodeHash(s.settings)), 300);
});

configure(initial.settings, initial.restore);
if (initial.restore) history.replaceState(null, '', encodeHash(initial.settings));
