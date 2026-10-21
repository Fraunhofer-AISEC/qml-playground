import { useEffect } from 'react';
import { useApp, type ThemeMode } from '../state/store';
import { Icon } from './controls/Controls';
import { useResolvedTheme } from './hooks';
import { ControlPanel } from './panels/ControlPanel';
import { DataPanel } from './panels/DataPanel';
import { MetricsPanel } from './panels/MetricsPanel';
import { CircuitPanel, LayerStatesPanel } from './panels/ModelPanel';
import { OutputPanel } from './panels/OutputPanel';
import { ResultsPanel } from './panels/ResultsPanel';
import { SettingsPanel } from './panels/SettingsPanel';

const REPO_URL = 'https://github.com/fraunhofer-aisec/qml-playground';
const PAPER_URL = 'https://doi.ieeecomputersociety.org/10.1109/MCG.2024.3456288';
const BASE = import.meta.env.BASE_URL;

const NEXT_THEME: Record<ThemeMode, ThemeMode> = { system: 'light', light: 'dark', dark: 'system' };
const THEME_ICON = { system: 'auto', light: 'sun', dark: 'moon' } as const;

export function App() {
  useResolvedTheme();
  const themeMode = useApp((s) => s.themeMode);
  const setThemeMode = useApp((s) => s.setThemeMode);
  const error = useApp((s) => s.error);
  const pinSample = useApp((s) => s.pinSample);

  // Esc stops following a sample.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') pinSample(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pinSample]);

  return (
    <div className="app">
      <header className="hero">
        <HeroArt />
        <div className="hero-brand">
          <img src={`${BASE}qmlpg-icon.svg`} alt="" width={52} height={52} />
          <div>
            <h1>QML Playground</h1>
            <p className="hero-tagline">Train and explore data re-uploading quantum circuits in real time.</p>
          </div>
        </div>
        <div className="header-actions">
          <a className="btn hero-link" href={PAPER_URL} target="_blank" rel="noreferrer">
            Paper
          </a>
          <button
            type="button"
            className="icon-btn"
            onClick={() => setThemeMode(NEXT_THEME[themeMode])}
            aria-label={`Theme: ${themeMode}. Switch to ${NEXT_THEME[themeMode]}`}
            title={`Theme: ${themeMode}`}
          >
            <Icon name={THEME_ICON[themeMode]} />
          </button>
          <a className="icon-btn" href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source on GitHub" title="Source on GitHub">
            <Icon name="github" />
          </a>
        </div>
      </header>

      {error && (
        <div className="error" role="alert">
          Simulation error: {error}
        </div>
      )}

      <main className="grid">
        <ControlPanel />
        <SettingsPanel />
        <MetricsPanel />
        <DataPanel />
        <CircuitPanel />
        <OutputPanel />
        <LayerStatesPanel />
        <ResultsPanel />
      </main>

      <footer className="app-footer">
        <img src={`${BASE}aisec.png`} alt="Fraunhofer AISEC" className="logo" />
        <nav>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source on GitHub
          </a>
          <a href={PAPER_URL} target="_blank" rel="noreferrer">
            Paper (IEEE CG&amp;A)
          </a>
          <a href="https://dsi-generator.fraunhofer.de/dsi/view/en/43ea542c-d1b7-42a6-a622-79a991ab2418/full/" target="_blank" rel="noreferrer">
            Privacy Notice
          </a>
          <a href="https://dsi-generator.fraunhofer.de/impressum/impressum_view/en/9e6f7056-74b4-42ba-a5ae-32e9e95eff4d/full/" target="_blank" rel="noreferrer">
            Imprint
          </a>
        </nav>
        <span className="muted">
          © {new Date().getFullYear()} Pascal Debus, Fraunhofer AISEC
        </span>
      </footer>
    </div>
  );
}

/** Decorative background of the header: faint qubit wires, gates and a Bloch circle. */
function HeroArt() {
  return (
    <svg className="hero-art" viewBox="0 0 520 130" preserveAspectRatio="xMaxYMid slice" aria-hidden="true">
      <g className="hero-art-lines">
        <line x1="0" x2="520" y1="44" y2="44" />
        <line x1="0" x2="520" y1="92" y2="92" />
        {[40, 150, 260].map((x) => (
          <g key={x}>
            <rect x={x} y="30" width="28" height="28" rx="5" />
            <rect x={x + 36} y="30" width="28" height="28" rx="5" />
            <rect x={x} y="78" width="28" height="28" rx="5" />
            <rect x={x + 36} y="78" width="28" height="28" rx="5" />
            <line x1={x + 84} x2={x + 84} y1="44" y2="100" />
            <circle cx={x + 84} cy="44" r="4" className="filled" />
            <circle cx={x + 84} cy="92" r="8" />
          </g>
        ))}
        <circle cx="440" cy="66" r="50" />
        <ellipse cx="440" cy="66" rx="50" ry="15" />
        <line x1="440" x2="468" y1="66" y2="34" />
        <circle cx="468" cy="34" r="4" className="filled" />
      </g>
    </svg>
  );
}
