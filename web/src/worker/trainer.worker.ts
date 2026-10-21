/// <reference lib="webworker" />
import type { FromWorker, Snapshot, ToWorker } from './protocol';
import { Trainer } from './trainer';
import { structuralChanged } from '../state/settings';

declare const self: DedicatedWorkerGlobalScope;

/** Minimum time between snapshots posted to the main thread (≈30 Hz). */
const SNAPSHOT_INTERVAL_MS = 33;
/** Minimum time between Monte Carlo uncertainty refreshes while training. */
const MC_INTERVAL_MS = 150;
/** Minimum time between unit-map (circuit view) refreshes while training. */
const UNITS_INTERVAL_MS = 100;
/** Time budget per loop slice when training at maximum speed or fast-forwarding. */
const SLICE_MS = 25;

let trainer: Trainer | null = null;
let configId = 0;
let playing = false;
let fastForwardLeft = 0;
let timer: ReturnType<typeof setTimeout> | null = null;
let lastPost = 0;
let lastMc = 0;
let lastUnits = 0;

function send(msg: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

function postSnapshot(force = false) {
  if (!trainer) return;
  const now = performance.now();
  if (!force && now - lastPost < SNAPSHOT_INTERVAL_MS) return;
  const mc = force || now - lastMc >= MC_INTERVAL_MS;
  const units = force || now - lastUnits >= UNITS_INTERVAL_MS;
  const snap: Snapshot = trainer.snapshot({ mc, units });
  if (mc) lastMc = now;
  if (units) lastUnits = now;
  lastPost = now;
  const transfer: Transferable[] = [
    snap.layerVis.buffer,
    snap.predictions.buffer,
    snap.theta.buffer,
    snap.grad.buffer,
    snap.units.bloch.buffer,
    snap.units.tangle.buffer,
    snap.units.scores.buffer,
  ];
  if (snap.decision) transfer.push(snap.decision.buffer);
  send({ type: 'snapshot', snap }, transfer);
}

function sendStatus() {
  send({ type: 'status', playing, busy: fastForwardLeft > 0 });
}

function stopTimer() {
  if (timer !== null) clearTimeout(timer);
  timer = null;
}

function schedule(delay: number) {
  stopTimer();
  timer = setTimeout(loop, delay);
}

function loop() {
  timer = null;
  if (!trainer) return;
  const t0 = performance.now();

  if (fastForwardLeft > 0) {
    while (fastForwardLeft > 0 && performance.now() - t0 < SLICE_MS) {
      trainer.stepEpoch();
      fastForwardLeft--;
    }
    postSnapshot(fastForwardLeft === 0);
    if (fastForwardLeft === 0) sendStatus();
    schedule(0);
    return;
  }

  if (!playing) return;
  const speed = trainer.settings.speed;
  if (speed > 0) {
    trainer.stepEpoch();
    postSnapshot();
    schedule(Math.max(0, 1000 / speed - (performance.now() - t0)));
  } else {
    do trainer.stepEpoch();
    while (performance.now() - t0 < SLICE_MS);
    postSnapshot();
    schedule(0);
  }
}

function setPlaying(value: boolean) {
  playing = value;
  sendStatus();
  if (playing) schedule(0);
  else postSnapshot(true);
}

function handle(msg: ToWorker) {
  switch (msg.type) {
    case 'configure': {
      stopTimer();
      fastForwardLeft = 0;
      trainer = new Trainer(++configId, msg.settings, msg.restore);
      send({ type: 'data', info: trainer.dataInfo() });
      postSnapshot(true);
      if (playing) schedule(0);
      sendStatus();
      return;
    }
    case 'hyper':
      if (!trainer) return;
      if (structuralChanged(trainer.settings, msg.settings)) {
        handle({ type: 'configure', settings: msg.settings });
        return;
      }
      trainer.settings = msg.settings;
      if (!playing) postSnapshot(true);
      else if (timer !== null && msg.settings.speed > 0) schedule(0);
      return;
    case 'play':
      setPlaying(true);
      return;
    case 'pause':
      setPlaying(false);
      return;
    case 'step':
      if (!trainer) return;
      if (playing) setPlaying(false);
      trainer.stepEpoch();
      postSnapshot(true);
      return;
    case 'fastForward':
      if (!trainer) return;
      fastForwardLeft += Math.max(0, Math.floor(msg.epochs));
      sendStatus();
      schedule(0);
      return;
    case 'seek':
      if (!trainer) return;
      if (playing) setPlaying(false);
      trainer.seek(msg.epoch);
      postSnapshot(true);
      return;
    case 'resumeFrom':
      if (!trainer) return;
      trainer.resumeFrom(msg.epoch);
      postSnapshot(true);
      return;
    case 'live':
      if (!trainer) return;
      trainer.viewEpoch = null;
      postSnapshot(true);
      return;
  }
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  try {
    handle(e.data);
  } catch (err) {
    send({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
