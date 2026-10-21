declare module 'plotly.js/lib/core' {
  const Plotly: {
    register(modules: unknown[]): void;
    react(el: HTMLElement, data: unknown[], layout: Record<string, unknown>, config?: Record<string, unknown>): Promise<unknown>;
    purge(el: HTMLElement): void;
    Plots: { resize(el: HTMLElement): void };
  };
  export default Plotly;
}

declare module 'plotly.js/lib/scatter';
declare module 'plotly.js/lib/bar';
declare module 'plotly.js/lib/heatmap';
declare module 'plotly.js/lib/scatter3d';
declare module 'plotly.js/lib/surface';
declare module 'plotly.js/lib/mesh3d';
declare module 'plotly.js/lib/cone';
