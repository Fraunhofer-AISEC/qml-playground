import Plotly from 'plotly.js/lib/core';
import scatter from 'plotly.js/lib/scatter';
import bar from 'plotly.js/lib/bar';
import heatmap from 'plotly.js/lib/heatmap';
import scatter3d from 'plotly.js/lib/scatter3d';
import surface from 'plotly.js/lib/surface';
import mesh3d from 'plotly.js/lib/mesh3d';
import cone from 'plotly.js/lib/cone';
import { useEffect, useRef, useState } from 'react';

// Partial bundle: only the trace types the playground uses.
Plotly.register([scatter, bar, heatmap, scatter3d, surface, mesh3d, cone]);

export type Trace = Record<string, unknown>;
export type Layout = Record<string, unknown>;
export interface Figure {
  data: Trace[];
  layout: Layout;
}

const CONFIG = { displayModeBar: false, responsive: false, displaylogo: false };

/**
 * Point events of a plot. Handlers receive the `customdata` of the point under the cursor
 * (traces without customdata are ignored).
 */
export interface PlotEvents {
  onHover?: (customdata: unknown) => void;
  /** Called when the pointer leaves the plot (not on Plotly's unhover, which also fires on redraws). */
  onUnhover?: () => void;
  onClick?: (customdata: unknown) => void;
}

interface PlotlyPoint {
  customdata?: unknown;
}
type PlotDiv = HTMLDivElement & { on?: (event: string, cb: (e?: { points?: PlotlyPoint[] }) => void) => void };

/**
 * Renders `figure` into the returned div with `Plotly.react`, which diffs against the previous figure and
 * only redraws what changed. A constant `layout.uirevision` keeps user interaction (e.g. 3D camera)
 * across updates. The plot is resized with its container.
 */
export function usePlot(figure: Figure | null, events?: PlotEvents): (el: HTMLDivElement | null) => void {
  // A callback ref, so plots in conditionally rendered elements are created and purged (releasing their
  // WebGL contexts) with the element.
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const handlers = useRef(events);
  handlers.current = events;
  const bound = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!el || !figure) return;
    void Plotly.react(el, figure.data, figure.layout, CONFIG).then(() => {
      // Plotly adds `on` to the div on the first draw; bind once per element.
      const gd = el as PlotDiv;
      if (bound.current === el || !gd.on) return;
      bound.current = el;
      const pick = (e?: { points?: PlotlyPoint[] }) => e?.points?.find((p) => p.customdata !== undefined)?.customdata;
      gd.on('plotly_hover', (e) => {
        const d = pick(e);
        if (d !== undefined) handlers.current?.onHover?.(d);
      });
      gd.on('plotly_click', (e) => {
        const d = pick(e);
        if (d !== undefined) handlers.current?.onClick?.(d);
      });
    });
  }, [el, figure]);

  useEffect(() => {
    if (!el) return;
    const onLeave = () => handlers.current?.onUnhover?.();
    el.addEventListener('mouseleave', onLeave);
    let frame = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (el.querySelector('.plot-container')) Plotly.Plots.resize(el);
      });
    });
    ro.observe(el);
    return () => {
      el.removeEventListener('mouseleave', onLeave);
      ro.disconnect();
      cancelAnimationFrame(frame);
      Plotly.purge(el);
      bound.current = null;
    };
  }, [el]);

  return setEl;
}
