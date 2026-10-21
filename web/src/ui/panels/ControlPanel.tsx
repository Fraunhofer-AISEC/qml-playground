import { useState } from 'react';
import { useApp } from '../../state/store';
import { SPEED_OPTIONS } from '../../state/settings';
import { IconButton, Panel, Select } from '../controls/Controls';

const speedLabel = (s: number) => (s === 0 ? 'Max' : `${s} / s`);

export function ControlPanel() {
  const playing = useApp((s) => s.playing);
  const busy = useApp((s) => s.busy);
  const speed = useApp((s) => s.settings.speed);
  const epoch = useApp((s) => s.snap?.epoch ?? 0);
  const { togglePlay, step, reset, fastForward, update } = useApp.getState();

  return (
    <Panel title="Control" area="control" actions={<ShareButtons />}>
      <div className="transport">
        <IconButton icon="reset" label="Reset model" onClick={reset} />
        <IconButton icon={playing ? 'pause' : 'play'} label={playing ? 'Pause training' : 'Start training'} onClick={togglePlay} variant="primary" />
        <IconButton icon="step" label="Train a single epoch" onClick={step} />
        <div className="epoch">
          <span className="epoch-label">Epoch</span>
          <span className="epoch-value" aria-live="polite">
            {epoch}
          </span>
        </div>
      </div>
      <div className="inline-field">
        <span className="field-label">Speed</span>
        <Select
          label="Training speed: epochs per second while playing"
          value={speed}
          options={SPEED_OPTIONS.map((s) => ({ value: s, label: speedLabel(s) }))}
          onChange={(v) => update({ speed: v })}
        />
      </div>
      <button type="button" className="btn btn-block" onClick={() => fastForward(100)} disabled={busy} title="Train 100 epochs as fast as possible">
        Fast-forward +100 epochs
      </button>
      <EpochScrubber />
    </Panel>
  );
}

function EpochScrubber() {
  const snap = useApp((s) => s.snap);
  const { seek, resumeFrom, goLive } = useApp.getState();
  if (!snap) return null;
  const { epoch, latestEpoch, firstEpoch } = snap;
  const viewing = epoch !== latestEpoch;
  return (
    <div className="scrubber">
      <div className="field-label">
        Replay
        <span className="muted">{latestEpoch > firstEpoch ? `ep ${epoch} / ${latestEpoch}` : ''}</span>
        {viewing && (
          <>
            <IconButton icon="play" variant="small" label="Resume training from this epoch (discards later epochs)" onClick={() => resumeFrom(epoch)} />
            <IconButton icon="latest" variant="small" label="Back to the latest epoch" onClick={goLive} />
          </>
        )}
      </div>
      <input
        type="range"
        aria-label="Replay epoch"
        min={firstEpoch}
        max={Math.max(latestEpoch, firstEpoch + 1)}
        value={epoch}
        disabled={latestEpoch === firstEpoch}
        onChange={(e) => seek(Number(e.target.value))}
        style={{ ['--fill' as string]: `${((epoch - firstEpoch) / Math.max(1, latestEpoch - firstEpoch)) * 100}%` }}
      />
    </div>
  );
}

function ShareButtons() {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = async (withParams: boolean) => {
    const url = useApp.getState().shareUrl(withParams);
    try {
      await navigator.clipboard.writeText(url);
      setCopied('Copied');
    } catch {
      window.prompt('Copy this link:', url);
    }
    setTimeout(() => setCopied(null), 1600);
  };
  return (
    <div className="panel-actions">
      <span className="copied muted small" aria-live="polite">
        {copied ?? ''}
      </span>
      <IconButton icon="link" variant="small" label="Copy a link to the current settings" onClick={() => copy(false)} />
      <IconButton icon="share" variant="small" label="Copy a link that also restores the trained model shown right now" onClick={() => copy(true)} />
    </div>
  );
}
