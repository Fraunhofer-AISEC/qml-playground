import logging

import numpy as np

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

logger = logging.getLogger(" [PLOTLY]")


def compute_concurrence(states):
    states = np.atleast_2d(states)
    c = 2 * np.abs(states[:, 0] * states[:, 3] - states[:, 1] * states[:, 2])
    return c ** 2


def make_tetrahedron(targets=None):

    colorscale = px.colors.qualitative.D3

    traces = []

    # Define vertices of the 3-simplex (tetrahedron)
    vertices = np.array([
        [0, 0, 1],  # |00>
        [1, 0, 0],  # |01>
        [0, 1, 0],  # |10>
        [0, 0, 0]  # |11>
    ])

    # Make annotations
    if targets is None:
        target_states = list(range(4))
    else:
        target_states = targets[1]

    annotation_labels = [f"|{i:02b}>" for i in range(4)]
    colors = 4 * ["black"]
    font_sizes = 4 * [12]
    target_state = 0
    for i in range(4):
        colors[i] = colorscale[target_state] if i in target_states else "black"
        target_state += 1
        font_sizes[i] = 14 if i in target_states else 12

    annotations = []

    for c, l, col, s in zip(vertices, annotation_labels, colors, font_sizes):
        d = dict(
            x=c[0],
            y=c[1],
            z=c[2],
            text=l,
            font=dict(
                color=col,
                size=s
            ),
            showarrow=False
        )
        annotations.append(d)

    # Define midpoints of the edges (these correspond to Bell states)
    midpoints = {
        "|Φ±>": (vertices[0] + vertices[3]) / 2,
        "|Ψ±>": (vertices[1] + vertices[2]) / 2
    }

    for k, v in midpoints.items():
        d = dict(
            x=v[0],
            y=v[1],
            z=v[2],
            text=k,
            font=dict(
                color="#179c7d",
                size=12
            ),
            showarrow=False
        )
        annotations.append(d)

    # Define edges to highlight
    edges_to_highlight = [(0, 3), (1, 2)]

    # Plot the tetrahedron edges with low opacity
    for i, j, k in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
        traces.append(go.Mesh3d(x=[vertices[i][0], vertices[j][0], vertices[k][0], vertices[i][0]],
                                y=[vertices[i][1], vertices[j][1], vertices[k][1], vertices[i][1]],
                                z=[vertices[i][2], vertices[j][2], vertices[k][2], vertices[i][2]],
                                opacity=0.15,
                                color='#179c7d'))
    # Highlight specific edges
    for edge in edges_to_highlight:
        traces.append(go.Scatter3d(x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                                   y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                                   z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                                   mode='lines',
                                   opacity=0.7,
                                   line=dict(color='#179c7d', width=4)))

    return traces, annotations


def make_tetrahedron_state_traces(states, labels, num_classes=None):

    if states is None:
        states = np.eye(4)

    if labels is None:
        labels = np.array([0, 1, 2, 3], dtype=np.uint)
    else:
        labels = labels.values

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    min_size = 3
    max_size = 10
    colors = px.colors.qualitative.D3[:num_classes]

    state_traces = []

    for c in range(num_classes):
        idx = np.where(labels == c)
        c_states = states[idx]

        probs = np.abs(c_states) ** 2
        concurrences = compute_concurrence(c_states)
        sizes = min_size + (max_size - min_size) * concurrences

        state_trace = go.Scatter3d(
            x=probs[:, 1].tolist(),
            y=probs[:, 2].tolist(),
            z=probs[:, 0].tolist(),
            mode='markers',
            name="States",
            showlegend=False,
            marker=dict(
                size=sizes,
                color=colors[c],
                line=dict(width=0.05),
                opacity=0.7
            ))

        state_traces.append(state_trace)

    return state_traces


def make_tetrahedron_plot(states=None, labels=None, targets=None):
    fig = go.Figure()

    traces, annotations = make_tetrahedron(targets=targets)
    state_traces = make_tetrahedron_state_traces(states, labels)

    traces += state_traces

    for t in traces:
        fig.add_trace(t)

    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )

    camera = dict(
        #up=dict(x=0, y=0, z=1),
        #center=dict(x=0, y=0, z=0),
        eye=dict(x=1.4, y=1.1, z=0.2)
    )

    fig.update_layout(
        autosize=False,
        width=300,
        height=300,
        margin=dict(b=0, t=0, l=0, r=0),
        showlegend=False,
        scene_camera=camera,
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            annotations=annotations,
            xaxis=axes_config,
            yaxis=axes_config,
            zaxis=axes_config,
            aspectratio=dict(x=1, y=1, z=1),
        )
    )

    return fig


def make_tetrahedron_model_plot(states_seq, labels, num_layers, targets=None):

    if states_seq is None and num_layers is None:
        raise ValueError("Must specify either states_seq or num_layers")

    if num_layers is None:
        num_layers = len(states_seq) - 1
    else:
        num_layers -= 1

    num_rows = int(np.ceil(num_layers / 2))

    traces, annotations = make_tetrahedron(targets=targets)

    fig = make_subplots(rows=num_rows,
                        cols=2,
                        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}] for i in range(num_rows)],
                        #subplot_titles=([f"Layer {i+1}" for i in range(num_layers)])
                        )

    for n in range(num_layers):

        states = states_seq[n] if states_seq is not None else None
        state_traces = make_tetrahedron_state_traces(states, labels)

        for t in traces:
            fig.add_trace(t, row=n // 2 + 1, col=n % 2 + 1)

        for t in state_traces:
            fig.add_trace(t, row=n // 2 + 1, col=n % 2 + 1)

    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )

    camera = dict(
       # up=dict(x=0, y=0, z=1),
       # center=dict(x=0, y=0, z=0),
        eye=dict(x=1.2, y=0.9, z=0.3)
    )

    layout = dict(
        autosize=False,
        width=600,
        height=300 * num_layers / 2,
        showlegend=False,
        margin=dict(b=0, t=0, l=0, r=0),
    )

    for i in range(1, num_layers + 1):
        num = "" if i == 1 else i
        layout[f"scene{num}"] = dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            annotations=annotations,
            xaxis=axes_config,
            yaxis=axes_config,
            zaxis=axes_config,
        )
        layout[f"scene{num}_camera"] = camera

    fig.update_layout(layout)

    return fig