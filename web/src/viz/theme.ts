export type ResolvedTheme = 'light' | 'dark';

/** Colors used inside Plotly figures (which cannot read CSS variables). Keep in sync with tokens.css. */
export interface PlotTheme {
  name: ResolvedTheme;
  text: string;
  muted: string;
  grid: string;
  line: string;
  accent: string;
  accentStrong: string;
  sceneBg: string;
  wrong: string;
  right: string;
}

export const PLOT_THEMES: Record<ResolvedTheme, PlotTheme> = {
  light: {
    name: 'light',
    text: '#1d2a26',
    muted: '#6b7a75',
    grid: '#e3e9e7',
    line: '#1d2a26',
    accent: '#179c7d',
    accentStrong: '#006247',
    sceneBg: 'rgba(0,0,0,0)',
    wrong: '#d62728',
    right: '#2ca02c',
  },
  dark: {
    name: 'dark',
    text: '#e3ebe8',
    muted: '#93a39e',
    grid: '#2c3835',
    line: '#c9d6d2',
    accent: '#2bc49f',
    accentStrong: '#7fe0c6',
    sceneBg: 'rgba(0,0,0,0)',
    wrong: '#ff6b6b',
    right: '#4cd47a',
  },
};

/** Plotly's qualitative D3 palette (class colors in the reference app). */
export const CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'];

/** Same system font stack as --font in tokens.css (no externally loaded fonts). */
const PLOT_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";

export function baseLayout(t: PlotTheme): Record<string, unknown> {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: PLOT_FONT, color: t.text, size: 11 },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: false,
    autosize: true,
    hoverlabel: { font: { family: PLOT_FONT } },
  };
}
