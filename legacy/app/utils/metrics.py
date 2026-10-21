import numpy as np
import plotly.express as px


# Compute combined metric: Δ_total + bias_phase
def compute_combined_metric(W, B):
    delta_total = np.sqrt(np.sum((np.mod(W, 2 * np.pi)) ** 2, axis=-1))        # [layers, qubits]
    bias_phase = np.mod(np.linalg.norm(B, axis=-1), 2 * np.pi)               # [layers, qubits]
    combined_metric = delta_total + bias_phase
    return combined_metric, delta_total, bias_phase


# Normalize to [0, 1]
def normalize_metrics(values, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    return (values - vmin) / (vmax - vmin + 1e-8)


def get_color_from_combined_metric(value, vmin, vmax):
    # Plotly color mapping using BuGn
    color_scale = px.colors.sequential.BuGn

    norm_val = normalize_metrics(np.array([value]), vmin=vmin, vmax=vmax)[0]
    index = int(norm_val * (len(color_scale) - 1))
    index = min(max(index, 0), len(color_scale) - 1)
    return color_scale[index]
