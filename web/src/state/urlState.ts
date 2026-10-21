import { DEFAULT_SETTINGS, sanitizeSettings, type Settings } from './settings';
import type { RestoreState } from '../worker/protocol';

/** Short URL keys for every setting (format version 1). */
const KEYS: Record<keyof Settings, string> = {
  task: 'task',
  dataset: 'ds',
  numQubits: 'q',
  numLayers: 'L',
  lr: 'lr',
  batchSize: 'bs',
  regType: 'reg',
  regStrength: 'rs',
  noiseSigma: 'sig',
  mcRuns: 'mc',
  seed: 'seed',
  speed: 'spd',
  gridSize: 'grid',
  entangler: 'ent',
  dataAngles: 'da',
  weights: 'w',
};

const VERSION = '1';

function toBase64Url(bytes: Uint8Array): string {
  let bin = '';
  bytes.forEach((b) => (bin += String.fromCharCode(b)));
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(s: string): Uint8Array {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (s.length % 4)) % 4);
  const bin = atob(b64);
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

export function encodeParams(theta: ArrayLike<number>): string {
  const f = Float32Array.from(theta);
  return toBase64Url(new Uint8Array(f.buffer));
}

export function decodeParams(s: string): Float32Array | null {
  try {
    const bytes = fromBase64Url(s);
    if (bytes.length % 4 !== 0) return null;
    const f = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4).slice();
    return f.every(Number.isFinite) ? f : null;
  } catch {
    return null;
  }
}

/** Serializes settings (only non-defaults, to keep links short) and optional trained parameters. */
export function encodeHash(settings: Settings, restore?: RestoreState): string {
  const p = new URLSearchParams();
  p.set('v', VERSION);
  (Object.keys(KEYS) as (keyof Settings)[]).forEach((k) => {
    if (settings[k] !== DEFAULT_SETTINGS[k] || k === 'task' || k === 'dataset' || k === 'numQubits') {
      p.set(KEYS[k], String(settings[k]));
    }
  });
  if (restore) {
    p.set('ep', String(restore.epoch));
    p.set('p', encodeParams(restore.theta));
  }
  return `#${p.toString()}`;
}

/** Parses a location hash; unknown or invalid values fall back to defaults. */
export function decodeHash(hash: string): { settings: Settings; restore?: RestoreState } {
  const p = new URLSearchParams(hash.replace(/^#/, ''));
  const raw: Partial<Record<keyof Settings, unknown>> = {};
  (Object.keys(KEYS) as (keyof Settings)[]).forEach((k) => {
    const v = p.get(KEYS[k]);
    if (v !== null) raw[k] = v;
  });
  const settings = sanitizeSettings(raw);
  const params = p.get('p');
  const theta = params ? decodeParams(params) : null;
  const epoch = Number(p.get('ep') ?? 0);
  const restore = theta ? { theta, epoch: Number.isFinite(epoch) && epoch >= 0 ? Math.floor(epoch) : 0 } : undefined;
  return { settings, restore };
}
