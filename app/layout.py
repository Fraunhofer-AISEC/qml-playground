from dash import dcc
from dash import html


control = html.Div([
    html.H2("Control"),
    html.A(children=html.Img(src="assets/github.svg", className="github-logo"),
           className="github_link",
           href='https://github.com/fraunhofer-aisec/qml-playground',
           target="_blank",
           title="Source on Github"),
    html.Div("QML Playground", id="app_title"),
    html.Div([
        html.Button(id='reset_button', n_clicks=0, className="button repeat"),
        html.Button(id='play_pause_button', n_clicks=0, className="button play"),
        html.Button(id='single_step_button', n_clicks=0, className="button next"),
    ], className="button-container"),
    html.Div([
        html.H3("Epoch", style={"font-size": "20pt"}),
        html.Div(id='epoch_display',
                 style={'font-size': '20pt',
                        'padding-top': '10px',
                        'padding-left': '15px',
                        'align': 'center', }),
    ], className="rowContainer"),
    dcc.Interval(
            id='interval_component',
            interval=500,  # in milliseconds
            n_intervals=0,
            max_intervals=-1,
            disabled=True
    )
], className="container", style={"height": "200px"})


# Settings
settings = html.Div([
    html.H2("Settings"),
    html.Div([

        html.Div([
            html.H3("Number of Qubits"),
            dcc.RadioItems(
                id="select_num_qubits",
                options=[{'label': 'Single Qubit', 'value': 1},
                         {'label': 'Two Qubits', 'value': 2},
                         ],
                value=1,
                inline=True,
            ),
        ], className="columnContainer"),
        html.Div([
        html.H3("Learning Rate"),
        dcc.Dropdown(
            id="select_lr",
            options=[
                {"value": 0.001, "label": 0.001},
                {"value": 0.01, "label": 0.01},
                {"value": 0.05, "label": 0.05},
                {"value": 0.1, "label": 0.1},
                {"value": 0.5, "label": 0.5},
            ],
            value=0.01,
            multi=False
        ),
        ], className="columnContainer"),
    ], className="rowContainer longContainer",
        style={'justify-content': 'space-around',
               'margin': '0px'}
    ),
    html.H3("Layers"),
    dcc.Slider(
        id="select_num_layers",
        min=1, max=15, step=1, value=5,
        # tooltip={"always_visible": True, "placement": "bottom"},
        marks={i: str(i) for i in range(1, 16)}
    ),
], className="longContainer container", style={"height": "200px"})


performance_plot = html.Div([
    html.H2("Metrics"),
    dcc.Graph(
        id="graph_loss_acc",
        style={"width": 300, "height": 150}
    ),
    html.P(
        id="paragraph_accuracies",
        children=[""]
    )
], className="container", style={"height": "200px"})

# Data
data_settings = html.Div([
    html.H2("Data"),
    html.H3("Data Set"),
    dcc.Dropdown(
        id="select_data_set",
        options=[
            {"label": "Circle", "value": "circle"},
            {"label": "3 Circles", "value": "3_circles"},
            {"label": "Square", "value": "square"},
            {"label": "4 Squares", "value": "4_squares"},
            {"label": "Crown", "value": "crown"},
            {"label": "Tri Crown", "value": "tricrown"},
            {"label": "Wavy Lines", "value": "wavy_lines"},
        ],
        value="circle",
    ),
    dcc.Graph(
        id="graph_data_set",
        style={"width": 250, "height": 250}
    ),
    html.H3("Batch Size"),
    dcc.Slider(
        id="select_batch_size",
        min=1, max=128, step=1, value=32,
        # tooltip={"always_visible": True, "placement": "bottom"},
        marks={i: str(i) for i in range(16, 129, 16)}
    ),
], className="container")

# Main plot
model_plot = html.Div([
    html.H2("Model"),
    dcc.Graph(
        id="graph_circuit",
        #style={"width": 300, "height": 300}
    ),
    dcc.Graph(
        id="graph_model",
        #style={"width": 300, "height": 300}
    ),
], className="longContainer container")

result_plot = html.Div([
    html.H2("Results"),
    dcc.Graph(
        id="graph_final_state",
        style={"width": 300, "height": 300}
    ),
    dcc.Graph(
        id="graph_decision",
        style={"width": 250, "height": 250}
    ),
    dcc.Graph(
        id="graph_results",
        style={"width": 300, "height": 150}
    ),
], className="container")

# Overall
layout_rows = [
    dcc.Store(id="simulation_state"),
    dcc.Store(id="train_datastore"),
    dcc.Store(id="test_datastore"),
    dcc.Store(id="model_parameters"),
    dcc.Store(id="metrics"),
    dcc.Store(id="quantum_state_store"),
    dcc.Store(id="predicted_test_labels"),
    dcc.Store(id="decision_boundary_store"),
    # html.Div([
    #     header,
    # ], className="rowContainer"),
    html.Div([
       control, settings, performance_plot,
    ], className="rowContainer"),
    html.Div([
        data_settings, model_plot, result_plot,
    ], className="rowContainer"),
    html.Div([
        html.Img(src="assets/aisec.png")
    ], className="logo"),
    html.Div([
        html.Div(children=html.A(children="Source on Github",
                                 href='https://github.com/fraunhofer-aisec/qml-playground',
                                 target="_blank",
                                 title="Source on Github"),
                 className="container footer",
                 style={"margin-left": "2px"}),
    	html.Div(children=html.A(children="Privacy Notice",
                                 href='https://dsi-generator.fraunhofer.de/dsi/view/en/43ea542c-d1b7-42a6-a622-79a991ab2418/full/',
                                 target="_blank",
                                 title="Privacy Notice"),
                 className="container footer",
                 style={"margin-left": "2px"}),
    	html.Div(children=html.A(children="Imprint",
                                 href='https://dsi-generator.fraunhofer.de/impressum/impressum_view/en/9e6f7056-74b4-42ba-a5ae-32e9e95eff4d/full/',
                                 target="_blank",
                                 title="Imprint"),
                 className="container footer",
                 style={"margin-left": "2px",
                        "width": "352px",}),
        html.Div("© 2025 Pascal Debus, Fraunhofer AISEC",
                 className="container footer",
                 style={"margin-left": "2px",
                        "text-align": "right"}),
    ], className="rowContainer"),
]

layout_overall = html.Div(layout_rows, className="overallContainer")
