import { useId, type ReactNode } from 'react';

export function Panel({ title, area, info, children, actions }: { title: string; area: string; info?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="panel" style={{ gridArea: area }} aria-label={title}>
      <header className="panel-header">
        <h2>
          {title}
          {info && <InfoTip>{info}</InfoTip>}
        </h2>
        {actions}
      </header>
      {children}
    </section>
  );
}

/** Small "i" button that reveals an explanation on hover or keyboard focus. */
export function InfoTip({ children }: { children: ReactNode }) {
  const id = useId();
  return (
    <span className="infotip">
      <button type="button" className="infotip-btn" aria-describedby={id} aria-label="More information">
        i
      </button>
      <span role="tooltip" id={id} className="infotip-body">
        {children}
      </span>
    </span>
  );
}

export function Field({ label, info, children, hint }: { label: string; info?: ReactNode; children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="field">
      <div className="field-label">
        {label}
        {info && <InfoTip>{info}</InfoTip>}
      </div>
      {children}
      {hint && <div className="field-hint">{hint}</div>}
    </div>
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? 'active' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string | number>({
  value,
  options,
  onChange,
  label,
  disabled,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <select
      className="select"
      aria-label={label}
      value={String(value)}
      disabled={disabled}
      onChange={(e) => {
        const opt = options.find((o) => String(o.value) === e.target.value);
        if (opt) onChange(opt.value);
      }}
    >
      {options.map((o) => (
        <option key={String(o.value)} value={String(o.value)}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  label,
  format = String,
  disabled,
  ticks,
  tickLabels,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  label: string;
  format?: (v: number) => string;
  disabled?: boolean;
  /** Values that get a tick mark under the track; those in `tickLabels` are also labeled. */
  ticks?: number[];
  tickLabels?: number[];
}) {
  const frac = (v: number) => (v - min) / (max - min || 1);
  const input = (
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ ['--fill' as string]: `${frac(value) * 100}%` }}
    />
  );
  return (
    <div className="slider">
      {ticks ? (
        <div className="slider-track">
          {input}
          <div className="slider-ticks" aria-hidden="true">
            {ticks.map((t) => (
              // Offset by half the thumb width so a tick sits under the thumb's center.
              <span key={t} className={tickLabels?.includes(t) ? 'labeled' : ''} style={{ left: `calc(8px + (100% - 16px) * ${frac(t)})` }}>
                {tickLabels?.includes(t) && <b>{t}</b>}
              </span>
            ))}
          </div>
        </div>
      ) : (
        input
      )}
      <output className="slider-value">{format(value)}</output>
    </div>
  );
}

const ICONS = {
  play: 'M8 5.5v13l10.5-6.5z',
  pause: 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z',
  step: 'M6 5.5v13l9-6.5zM16 5h2.5v14H16z',
  reset: 'M12 5a7 7 0 1 0 6.93 8h-2.04A5 5 0 1 1 12 7c1.38 0 2.63.56 3.54 1.46L13 11h6V5l-2.05 2.05A6.97 6.97 0 0 0 12 5z',
  forward: 'M4 6v12l8-6zM12 6v12l8-6z',
  share: 'M18 16a3 3 0 0 0-2.4 1.2l-6.7-3.4a3 3 0 0 0 0-1.6l6.7-3.4A3 3 0 1 0 15 7l-6.7 3.4a3 3 0 1 0 0 3.2L15 17a3 3 0 1 0 3-1z',
  dice: 'M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm3 3.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm8 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm-4-4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z',
  sun: 'M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM11 1h2v3h-2zm0 19h2v3h-2zM1 11h3v2H1zm19 0h3v2h-3zM4.2 5.6l1.4-1.4 2.1 2.1-1.4 1.4zm12.1 12.1 1.4-1.4 2.1 2.1-1.4 1.4zM4.2 18.4l2.1-2.1 1.4 1.4-2.1 2.1zM16.3 6.3l2.1-2.1 1.4 1.4-2.1 2.1z',
  moon: 'M20 15.3A8 8 0 0 1 8.7 4a8 8 0 1 0 11.3 11.3z',
  auto: 'M12 3a9 9 0 1 0 0 18V3z M12 3a9 9 0 0 1 0 18',
  link: 'M10.6 13.4a1 1 0 0 1 0-1.4l3.5-3.5a1 1 0 1 1 1.4 1.4L12 13.4a1 1 0 0 1-1.4 0zM8.5 11.5l-2 2a3 3 0 0 0 4.2 4.2l2-2 1.4 1.4-2 2a5 5 0 0 1-7-7l2-2zM15.5 12.5l2-2a3 3 0 0 0-4.2-4.2l-2 2-1.4-1.4 2-2a5 5 0 0 1 7 7l-2 2z',
  latest: 'M4 5.5v13l7-6.5zM11 5.5v13l7-6.5zM18 5h2.5v14H18z',
  close: 'M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19l5.6-5.6 5.6 5.6 1.4-1.4-5.6-5.6L19 6.4 17.6 5 12 10.6z',
  pin: 'M15 3v2h-1v5.2l3 3V15h-4v6l-1 1-1-1v-6H7v-1.8l3-3V5H9V3z',
  github:
    'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z',
};

/** Icons drawn on a different grid than the default 24×24. */
const VIEWBOX: Partial<Record<keyof typeof ICONS, string>> = { github: '0 0 16 16' };

export type IconName = keyof typeof ICONS;

export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox={VIEWBOX[name] ?? '0 0 24 24'} aria-hidden="true" fill="currentColor">
      <path d={ICONS[name]} fillRule="evenodd" />
    </svg>
  );
}

export function IconButton({
  icon,
  label,
  onClick,
  variant = 'default',
  disabled,
}: {
  icon: IconName;
  label: string;
  onClick: () => void;
  variant?: 'default' | 'primary' | 'small';
  disabled?: boolean;
}) {
  const size = variant === 'primary' ? 24 : variant === 'small' ? 15 : 18;
  return (
    <button type="button" className={`icon-btn ${variant}`} onClick={onClick} aria-label={label} title={label} disabled={disabled}>
      <Icon name={icon} size={size} />
    </button>
  );
}
