import logging

import numpy as np
import pandas as pd
import torch

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from visualization.single_qubit import make_bloch_plot, make_bloch_model_plot, make_bloch_state_traces
from visualization.two_qubits import make_tetrahedron_plot, make_tetrahedron_model_plot, make_tetrahedron_state_traces
from utils.metrics import compute_combined_metric, normalize_metrics, get_color_from_combined_metric

logger = logging.getLogger(" [PLOTLY]")


def make_state_space_plot(num_qubits, states=None, labels=None, targets=None):
    """Create a visualization of quantum states in an appropriate state space.

    This function creates a visualization of quantum states based on the number of qubits.
    For single-qubit states, it generates a Bloch sphere representation.
    For two-qubit states, it generates a tetrahedron representation.

    Args:
        num_qubits (int): Number of qubits (1 or 2)
        states (list, optional): List of quantum states to visualize. Defaults to None.
        labels (list, optional): Class labels for each state. Defaults to None.
        targets (list, optional): Target states to include in the visualization. Defaults to None.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the visualization

    Raises:
        NotImplementedError: If num_qubits is not 1 or 2
    """
    if num_qubits == 1:
        return make_bloch_plot(states, labels, targets)
    elif num_qubits == 2:
        return make_tetrahedron_plot(states, labels, targets)
    else:
        raise NotImplementedError("num_qubits must be 1 or 2")


def make_state_space_model_plot(num_qubits, states, labels, num_layers, targets=None):
    """Create a visualization of quantum states evolution through model layers.

    This function creates a multi-panel visualization showing how quantum states
    evolve through the layers of a quantum circuit model. The representation
    depends on the number of qubits (Bloch sphere for 1 qubit, tetrahedron for 2 qubits).

    Args:
        num_qubits (int): Number of qubits (1 or 2)
        states (list): List of quantum states for each layer of the model
        labels (list): Class labels for each state
        num_layers (int): Number of layers in the model
        targets (list, optional): Target states to include in the visualization. Defaults to None.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the multi-panel visualization

    Raises:
        NotImplementedError: If num_qubits is not 1 or 2
    """
    if num_qubits == 1:
        return make_bloch_model_plot(states, labels, num_layers, targets)
    elif num_qubits == 2:
        return make_tetrahedron_model_plot(states, labels, num_layers, targets)
    else:
        raise NotImplementedError("num_qubits must be 1 or 2")


def make_state_traces(num_qubits, states, labels, num_classes=None):
    """Create trace objects for visualizing quantum states.

    This function generates Plotly trace objects for quantum states that can be
    added to existing figures. The type of traces depends on the number of qubits
    (Bloch sphere traces for 1 qubit, tetrahedron traces for 2 qubits).

    Args:
        num_qubits (int): Number of qubits (1 or 2)
        states (list): List of quantum states to visualize
        labels (list): Class labels for each state
        num_classes (int, optional): Number of distinct classes. Defaults to None.

    Returns:
        list: A list of Plotly trace objects that can be added to a figure

    Raises:
        NotImplementedError: If num_qubits is not 1 or 2
    """
    if num_qubits == 1:
        return make_bloch_state_traces(states, labels, num_classes)
    elif num_qubits == 2:
        return make_tetrahedron_state_traces(states, labels, num_classes)
    else:
        raise NotImplementedError("num_qubits must be 1 or 2")


def make_performance_plot(data):
    """Create a plot showing model performance metrics over training epochs.

    This function generates a line plot visualizing the loss and accuracy metrics
    during model training. It shows the training loss, training accuracy, and test
    accuracy over epochs.

    Args:
        data (pandas.DataFrame, optional): DataFrame containing performance metrics with columns
            'loss', 'train_accuracy', and 'test_accuracy'. If None, an empty plot is created.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the performance metrics plot
    """
    if data is None:
        data = pd.DataFrame(columns=["loss", "train_accuracy", "test_accuracy"])

    fig = go.Figure(
        layout_yaxis_range=[0, 1],)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["loss"],
        mode='lines',
        name='Loss',
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["train_accuracy"],
        mode='lines',
        name='Acc Train',
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["test_accuracy"],
        mode='lines',
        name='Acc Test'
    ))

    fig.update_layout(
        autosize=False,
        width=300,
        height=200,
        margin=dict(b=0, t=0, l=0, r=0)
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))

    return fig




def make_data_plot(points, labels):
    """Create a scatter plot visualizing the dataset with colored class labels.

    This function generates a scatter plot of 2D data points, with colors
    representing different class labels. It's used to visualize the training
    or test datasets for classification problems.

    Args:
        points (numpy.ndarray): Array of shape (n_samples, 2) containing the 2D coordinates
            of data points
        labels (torch.Tensor): Tensor of shape (n_samples,) containing the class labels
            for each data point

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the scatter plot
            of the dataset
    """
    fig = go.Figure(
        layout_xaxis_range=[-1, 1],
        layout_yaxis_range=[-1, 1],
    )
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers',
        marker=dict(color=labels,
                    colorscale=px.colors.qualitative.D3[:torch.max(labels).numpy() + 1],
                    size=5,
                    ),
    ))

    fig.update_layout(
        autosize=False,
        width=250,
        height=250,
        margin=dict(b=0, t=15, l=0, r=0)
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def make_result_plot(points, predictions, labels, dataset="circle"):
    """Create a visualization of model predictions and their accuracy.

    This function generates a two-panel plot showing:
    1. Data points colored by their predicted classes
    2. Data points colored to indicate correct (green) or incorrect (red) predictions

    The plot also includes dataset-specific decision boundaries or regions based on
    the dataset type.

    Args:
        points (pandas.DataFrame): DataFrame containing 'x' and 'y' coordinates for each point
        predictions (numpy.ndarray): Array of predicted class labels
        labels (pandas.DataFrame): DataFrame of true class labels
        dataset (str, optional): Type of dataset for drawing appropriate boundaries.
            Options: 'circle', '3_circles', 'square', '4_squares', 'crown', 'tricrown', 'wavy_lines'.
            Defaults to "circle".

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the visualization of model results
    """
    if points is None or predictions is None or labels is None:
        points = pd.DataFrame(columns=["x", "y"])
        points["x"] = -1.1 * np.ones(4)
        points["y"] = -1.1 * np.ones(4)

        predictions = np.arange(4, dtype=np.uint)
        labels = np.arange(4, dtype=np.uint)
    else:
        labels = labels.values

    class_colors = px.colors.qualitative.D3[:4]

    fig = make_subplots(rows=1, cols=2)

    for c in range(4):
        idx = np.where(predictions == c)
        class_points = points.iloc[idx]
        color = class_colors[c]

        fig.add_trace(go.Scatter(
            x=class_points["x"].tolist(),
            y=class_points["y"].tolist(),
            name=f"Class {c} States",
            showlegend=False,
            mode='markers',
            marker=dict(color=color,
                        size=4,
                        ),
        ), row=1, col=1)

    checks = (predictions == labels)
    wrong_points = points.values[checks == False]
    right_points = points.values[checks == True]

    fig.add_trace(go.Scatter(
        x=wrong_points[:, 0].tolist(),
        y=wrong_points[:, 1].tolist(),
        name=f"Wrongly Predicted",
        showlegend=False,
        mode='markers',
        marker=dict(color="red",
                    size=4,
                    ),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=right_points[:, 0].tolist(),
        y=right_points[:, 1].tolist(),
        name=f"Correctly Predicted",
        showlegend=False,
        mode='markers',
        marker=dict(color="green",
                    size=4,
                    ),
    ), row=1, col=2)

    for i in [1, 2]:

        num = "" if i == 1 else str(i)
        x = f"x{num}"
        y = f"y{num}"

        if dataset == "circle":
            fig.add_shape(type="circle",
                          xref=x, yref=y,
                          x0=-np.sqrt(2 / np.pi), y0=-np.sqrt(2 / np.pi), x1=np.sqrt(2 / np.pi), y1=np.sqrt(2 / np.pi),
                          line_color="Black",
                          )

        elif dataset == '3_circles':
            centers = np.array([[-1, 1], [1, 0], [-.5, -.5]])
            radii = np.array([1, np.sqrt(6 / np.pi - 1), 1 / 2])

            for (c, r) in zip(centers, radii):
                fig.add_shape(type="circle",
                              xref=x, yref=y,
                              x0=c[0]-r, y0=c[1]-r, x1=c[0]+r, y1=c[1]+r,
                              line_color="Black",
                              )

        elif dataset == 'square':
            p = .5 * np.sqrt(2)
            fig.add_trace(go.Scatter(
                x=[-p, p, p, -p, -p],
                y=[-p, -p, p, p, -p],
                showlegend=False,
                mode='lines',
                line=dict(color="Black"),
            ), row=1, col=i)

        elif dataset == '4_squares':
            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[-1, 1],
                showlegend=False,
                mode='lines',
                line=dict(color="Black"),
            ), row=1, col=i)
            fig.add_trace(go.Scatter(
                x=[-1, 1],
                y=[0, 0],
                showlegend=False,
                mode='lines',
                line=dict(color="Black"),
            ), row=1, col=i)

        elif dataset == 'crown' or dataset == 'tricrown':
            centers = [[0, 0], [0, 0]]
            radii = [np.sqrt(.8), np.sqrt(.8 - 2 / np.pi)]

            for (c, r) in zip(centers, radii):
                fig.add_shape(type="circle",
                              xref=x, yref=y,
                              x0=c[0] - r, y0=c[1] - r, x1=c[0] + r, y1=c[1] + r,
                              line_color="Black",
                              )

        elif dataset == 'wavy_lines':
            freq = 1

            def fun1(s):
                return s + np.sin(freq * np.pi * s)

            def fun2(s):
                return -s + np.sin(freq * np.pi * s)

            x = np.linspace(-1, 1)
            fig.add_trace(go.Scatter(
                x=x,
                y=np.clip(fun1(x), -1, 1),
                showlegend=False,
                mode='lines',
                line=dict(color="Black"),
            ), row=1, col=i)
            fig.add_trace(go.Scatter(
                x=x,
                y=np.clip(fun2(x), -1, 1),
                showlegend=False,
                mode='lines',
                line=dict(color="Black"),
            ), row=1, col=i)

    fig.update_layout(
        autosize=False,
        width=300,
        height=150,
        margin=dict(b=0, t=10, l=0, r=0),
    )

    fig.update_xaxes(showticklabels=False, range=[-1, 1])
    fig.update_yaxes(showticklabels=False, range=[-1, 1])

    return fig


def make_decision_boundary_plot(x, y, Z, points, labels):
    """Create a visualization of model decision boundaries with data points.

    This function generates a contour plot showing the decision boundaries of a trained model,
    overlaid with the actual data points colored by their true class labels. The plot helps
    visualize how well the model's decision boundaries align with the true data distribution.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates for the grid
        y (numpy.ndarray): 1D array of y-coordinates for the grid
        Z (numpy.ndarray): 2D array of shape (len(y), len(x)) containing the decision function values
        points (pandas.DataFrame): DataFrame containing 'x' and 'y' coordinates for each data point
        labels (numpy.ndarray): Array of true class labels for each point

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the decision boundary visualization
    """
    if x is None or y is None or Z is None or points is None or labels is None:
        x = np.linspace(-1, 1, 15)
        y = x
        Z = np.random.rand(15, 15)

        points = pd.DataFrame(columns=["x", "y"])
        points["x"] = -1.1 * np.ones(4)
        points["y"] = -1.1 * np.ones(4)

        labels = np.arange(4, dtype=np.uint)

    class_colors = px.colors.qualitative.D3[:4]

    fig = go.Figure(
        layout_xaxis_range=[-1, 1],
        layout_yaxis_range=[-1, 1],
    )

    fig.add_trace(go.Contour(
        z=Z.tolist(),
        x=x.tolist(),
        y=y.tolist(),
        showscale=False,
        line_smoothing=0.5,
        contours_coloring='heatmap',
        colorscale=[[0, 'orange'], [0.5, 'white'], [1, 'blue']],
    ))

    for c in range(4):
        idx = np.where(labels == c)
        class_points = points.iloc[idx]
        color = class_colors[c]

        fig.add_trace(go.Scatter(
            x=class_points["x"],
            y=class_points["y"],
            mode='markers',
            showlegend=False,
            marker=dict(color=color,
                        size=4,
                        line=dict(width=1,
                                  color='White')
                        ),
        ))

    fig.update_traces(colorbar_orientation='h', selector=dict(type='heatmap'))

    fig.update_layout(
        autosize=False,
        width=250,
        height=250,
        margin=dict(b=0, t=0, l=0, r=0)
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def make_quantum_classifier_plot(num_qubits=2, num_layers=5, W=None, B=None, x_spacing = 1.35, x_offset_cnot = 0.5):

    if num_qubits not in [1, 2]:
        raise ValueError("Only 1 or 2 qubits supported")
    if not (1 <= num_layers <= 15):
        raise ValueError("Number of layers must be between 1 and 15")

    y_positions = {1: [0], 2: [0.75, 0]}  # Q1 at top
    y_labels = {1: ["Q1"], 2: ["Q1", "Q2"]}

    plot_weights = False
    if W is not None and B is not None:
        plot_weights = True
        combined_metric, delta_total, bias_phase = compute_combined_metric(W, B)
        vmin, vmax = np.min(combined_metric), np.max(combined_metric)

    fig = go.Figure()

    # Draw horizontal qubit lines
    for y in y_positions[num_qubits]:
        fig.add_trace(go.Scatter(
            x=[0, num_layers * x_spacing + 1],
            y=[y, y],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False
        ))

    # Plot each layer
    for i in range(1, num_layers + 1):
        x_gate = i * x_spacing
        x_cnot = x_gate + x_offset_cnot

        for q, y in enumerate(y_positions[num_qubits]):

            color = "#179c7d"
            if plot_weights:
                metric_value = combined_metric[i - 1, q]
                color = get_color_from_combined_metric(metric_value, vmin, vmax)

            fig.add_shape(type="rect",
                          x0=x_gate - 0.15, y0=y - 0.15,
                          x1=x_gate + 0.15, y1=y + 0.15,
                          line=dict(color="black"),
                          fillcolor=color)

            if plot_weights:
                # Add invisible hover point for tooltip
                fig.add_trace(go.Scatter(
                    x=[x_gate], y=[y],
                    mode="markers",
                    marker=dict(size=20, opacity=0),
                    hovertext=(
                        f"Combined Metric: {metric_value:.2f}<br>"
                        f"Δ_total: {delta_total[i - 1, q]:.2f}<br>"
                        f"Bias Phase: {bias_phase[i - 1, q]:.2f} rad"
                    ),
                    hoverinfo="text",
                    showlegend=False
                ))


        # CNOT/CZ gate for 2 qubits, but NOT on last layer
        if num_qubits == 2 and i < num_layers:
            control_y, target_y = y_positions[2]

            controlled_z = True
            radius = 0.05

            # Control Qubit
            fig.add_shape(type="circle",
                          x0=x_cnot - radius, y0=control_y - radius,
                          x1=x_cnot + radius, y1=control_y + radius,
                          line=dict(color="black"),
                          fillcolor="black")
            # Target Qubit
            if controlled_z:
                fig.add_shape(type="circle",
                              x0=x_cnot - radius, y0=target_y - radius,
                              x1=x_cnot + radius, y1=target_y + radius,
                              line=dict(color="black"),
                              fillcolor="black")
                fig.add_shape(type="line",
                              x0=x_cnot, y0=control_y,
                              x1=x_cnot, y1=target_y,
                              line=dict(color="black"))
            else:
                fig.add_shape(type="circle",
                              x0=x_cnot - 2 * radius, y0=target_y - 2 * radius,
                              x1=x_cnot + 2 * radius, y1=target_y + 2 * radius,
                              line=dict(color="black"))
                fig.add_shape(type="line",
                              x0=x_cnot, y0=control_y,
                              x1=x_cnot, y1=target_y - radius,
                              line=dict(color="black"))

        # X-axis label for layer
        fig.add_trace(go.Scatter(
            x=[x_gate], y=[min(y_positions[num_qubits]) - 0.25],
            text=[f"L{i}"],
            mode="text",
            textposition="bottom center",
            showlegend=False
        ))

    # Y-axis labels
    for idx, y in enumerate(y_positions[num_qubits]):
        fig.add_trace(go.Scatter(
            x=[-0.4], y=[y],
            text=[y_labels[num_qubits][idx]],
            mode="text",
            showlegend=False
        ))

    # Layout and appearance
    fig.update_layout(
        width=700, #100 * (num_layers + 2),
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-1, num_layers * x_spacing + 2]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-1, 2]),
        plot_bgcolor='white'
    )

    return fig