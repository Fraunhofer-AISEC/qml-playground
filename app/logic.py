import logging
import json
from io import StringIO

import pandas as pd
import numpy as np
import torch

import dash
from dash import ctx, clientside_callback
from dash.dependencies import Input, Output, State, ClientsideFunction

from data.datasets_torch import create_dataset, create_target, classification_datasets, regression_datasets

from models.reuploading_classifier import QuantumReuploadingClassifier
from models.reuploading_regressor import QuantumReuploadingRegressor

from utils.serialization import serialize_quantum_states, unserialize_quantum_states, unserialize_model_dict
from utils.trace_updates import create_extendData_dicts

from layout import layout_overall
from plotting import *

qml_app = dash.Dash(__name__, url_base_pathname='/qml-playground/', title="QML Playground")
# Assign layout
qml_app.layout = layout_overall

logger = logging.getLogger(" [DASH Callback]")

reset_triggers = ["reset_button", "select_num_qubits", "select_num_layers", "select_data_set", "select_task_type"]

@qml_app.callback(
    [
        Output(component_id="play_pause_button", component_property="n_clicks"),
    ],
    inputs=[
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def reset_play_pause(num_clicks: int,
                     num_qubits: int,
                     num_layers: int,
                     selected_data_set:str,
                     selected_task:str
                     ):
    """
    Resets the play/pause button functionality upon specific input triggers. This is primarily
    used to set the state of the "play_pause_button" to its initial condition (n_clicks = 0)
    when certain interactions occur, such as resetting the system or adjusting required
    parameters.

    :param num_clicks: The current number of clicks of the play/pause button.
    :type num_clicks: int
    :param num_qubits: The selected number of qubits for the system.
    :type num_qubits: int
    :param num_layers: The number of layers selected during configuration.
    :type num_layers: int
    :param selected_data_set: The currently selected dataset for the application.
    :param select_task_type: The currently selected task type for the application.
    :return: A list containing the reset value for the play/pause button click count.
    :rtype: List[int]
    """
    return [0]


@qml_app.callback(
    [
        Output(component_id="select_data_set", component_property="options"),
        Output(component_id="select_data_set", component_property="value"),
    ],
    inputs=[
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def update_data_set_options(selected_task_type: str):

    if selected_task_type == 'classification':
        data_set_dict = classification_datasets
        dataset = 'circle'
    else:
        data_set_dict = regression_datasets
        dataset = 'fourier_1'

    options=[{"label": v, "value": k} for k, v in data_set_dict.items()]

    return [options, dataset]


@qml_app.callback(
    [
        Output(component_id="graph_final_state", component_property="figure"),
    ],
    inputs=[
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def reset_final_state_plot(num_clicks: int,
                           num_qubits: int,
                           num_layers: int,
                           selected_data_set: str,
                           selected_task_type: str,
                           ):
    """
    Resets the final state plot based on the input parameters provided, updating the graphical
    representation in the application to reflect new configurations such as selected data set,
    number of qubits, and number of layers. This function is triggered by interactions with the
    reset button or changes to any of the input fields.

    :param num_clicks: Number of clicks detected on the reset button.
    :param num_qubits: Number of qubits selected for the quantum system.
    :param num_layers: Number of layers specified in the quantum circuit.
    :param selected_data_set: Selected data set used to generate the target states.
    :param select_task_type: The currently selected task type for the application.
    :return: Updated figure object representing the final state graph.
    """

    targets = None
    if selected_task_type == 'classification':
        targets = create_target(selected_data_set, num_qubits=num_qubits)

    graph_final_state = make_state_space_plot(num_qubits, targets=targets)

    return [graph_final_state]


@qml_app.callback(
    [
        Output(component_id="graph_model", component_property="figure"),
    ],
    inputs=[
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def reset_model_plot(num_clicks: int, num_qubits: int, num_layers: int,  selected_data_set, selected_task_type: str):
    """
    Handles the model plot reset functionality. This function is
    triggered by user interactions and updates the state space model plot based on the selected parameters.

    :param num_clicks: The number of times the reset button is clicked.
    :param num_qubits: The number of qubits to be used for the state space model.
    :param num_layers: The number of layers to use in the quantum model.
    :param selected_data_set: The dataset selected by the user for target creation.
    :param select_task_type: The currently selected task type for the application.
    :return: A list containing the updated figure object for the state space model plot.
    """
    # For regression tasks, there are no class targets
    targets = None
    if selected_task_type == 'classification':
        targets = create_target(selected_data_set, num_qubits=num_qubits)

    graph_model = make_state_space_model_plot(num_qubits,
                                              states=None,
                                              labels=None,
                                              num_layers=num_layers,
                                              targets=targets)

    return [graph_model]


@qml_app.callback(
    [
        Output(component_id="graph_loss_acc", component_property="figure"),
    ],
    inputs=[
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def reset_performance_plot(num_clicks: int,
                           num_qubits: int,
                           num_layers: int,
                           selected_data_set: str,
                           selected_task_type: str):
    """
    This function resets the performance plot of a quantum machine learning application
    based on the specified input values. It is triggered by changes in the reset button 
    click count, number of qubits, number of layers, or the selected dataset.

    :param num_clicks: Number of times the reset button has been clicked.
    :type num_clicks: int
    :param num_qubits: The number of qubits for the quantum circuit.
    :type num_qubits: int
    :param num_layers: The number of layers in the quantum circuit.
    :type num_layers: int
    :param selected_data_set: The dataset selected for use in the quantum machine learning model.
    :type selected_data_set: Any
    :return: A list with the updated performance plot figure.
    :rtype: list
    """
    graph_loss_acc = make_performance_plot(data=None)

    return [graph_loss_acc]


@qml_app.callback(
    [
        Output(component_id="graph_decision", component_property="figure"),
    ],
    inputs=[
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def reset_decision_plot(num_clicks: int, num_qubits: int, num_layers: int, selected_data_set, selected_task: str,):
    """
    Resets the decision boundary plot to its initial state when triggered by changes in input parameters.
    """
    if selected_task == 'regression':
        graph_decision = make_regression_decision_plot()
    else:
        graph_decision = make_decision_boundary_plot(x=None, y=None, Z=None, points=None, labels=None)
    return [graph_decision]


@qml_app.callback(
    [
        Output(component_id="graph_results", component_property="figure"),
    ],
    inputs=[
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    prevent_initial_call=False,
)
def reset_results_plot(num_clicks: int, num_qubits: int, num_layers: int, selected_data_set, selected_task: str, ):
    """
    Resets the results plot showing model predictions and actual labels.
    """
    if selected_task == 'regression':
        graph_results = make_regression_results_plot()
    else:
        graph_results = make_result_plot(points=None, predictions=None, labels=None, dataset=selected_data_set)
    return [graph_results]


@qml_app.callback(
    [
        Output(component_id="play_pause_button", component_property="className"),
        Output(component_id="interval_component", component_property="disabled"),
    ],
    inputs=[
        Input(component_id="play_pause_button", component_property="n_clicks"),
    ],
    prevent_initial_call=True,
)
def play_pause_handler(num_clicks: int):
    classname = "button pause"
    disabled = False

    if num_clicks % 2 == 0:
        classname = "button play"
        disabled = True

    return [classname, disabled]


@qml_app.callback(
    [
        Output(component_id="graph_circuit", component_property="figure"),
    ],
    inputs=[
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="model_parameters", component_property="data"),
    ],
    prevent_initial_call=False,
)
def update_circuit_plot(num_qubits: int, num_layers: int, model_parameters: str):
    """
    Updates the circuit plot showing model architecture.

    :param num_qubits: Number of qubits in the quantum system
    :type num_qubits: int
    :param num_layers: Number of layers in the quantum circuit
    :type num_layers: int
    :param model_parameters: Current model parameters, serialized as JSON string
    :type model_parameters: str
    :return: A list containing the circuit plot figure
    :rtype: list
    """

    if model_parameters is not None:
        model_parameters = unserialize_model_dict(model_parameters)

        weights = model_parameters["weights"]
        biases = model_parameters["biases"]

        weights = torch.reshape(weights, (num_layers, num_qubits, 3)).numpy()
        biases = torch.reshape(biases, (num_layers, num_qubits, 3)).numpy()

    else:
        weights = None
        biases = None

    graph_circuit = make_quantum_classifier_plot(num_qubits=num_qubits,
                                                     num_layers=num_layers,
                                                     W=weights,
                                                     B=biases,)

    return [graph_circuit]


@qml_app.callback(
    [
        Output(component_id="model_parameters", component_property="data"),
        Output(component_id="epoch_display", component_property="children"),
    ],
    inputs=[
        Input(component_id="single_step_button", component_property="n_clicks"),
        Input(component_id="interval_component", component_property="n_intervals"),
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
    ],
    state=[
        State(component_id="select_lr", component_property="value"),
        State(component_id="select_batch_size", component_property="value"),
        State(component_id="select_reg_type", component_property="value"),
        State(component_id="select_reg_strength", component_property="value"),
        State(component_id="train_datastore", component_property="data"),
        State(component_id="model_parameters", component_property="data"),
    ],
)
def single_epoch(num_clicks: int,
                 num_intervals: int,
                 reset_clicks: int,
                 num_qubits: int,
                 num_layers: int,
                 selected_data_set: str,
                 selected_task: str,
                 lr: float,
                 batch_size: int,
                 reg_type: str,
                 reg_strength: float,
                 train_data,
                 model_parameters):
    """Performs a single training epoch for the quantum model using the provided parameters."""

    is_regression = selected_task == 'regression'
    model_cls = QuantumReuploadingRegressor if is_regression else QuantumReuploadingClassifier
    qmodel = model_cls(name=selected_data_set, num_qubits=num_qubits, layers=num_layers)

    if model_parameters is None or ctx.triggered_id in reset_triggers:
        model_parameters = qmodel.save_model()
        return [json.dumps(model_parameters), model_parameters["config"]["epoch"]]

    if train_data is not None:
        df_train = pd.read_json(StringIO(train_data), orient='split')
    else:
        return dash.no_update

    model_parameters = unserialize_model_dict(model_parameters)
    qmodel.load_model(model_parameters)

    if is_regression:
        X = df_train[["x"]].values
        y = df_train["y"].values
        qmodel.train_single_epoch(X, y, lr, batch_size, reg_type, reg_strength)
    else:
        qmodel.train_single_epoch(df_train[["x", "y"]].values, df_train["label"].values, lr, batch_size, reg_type, reg_strength)

    model_parameters = qmodel.save_model()

    return [json.dumps(model_parameters), model_parameters["config"]["epoch"]]


@qml_app.callback(
    [
        Output(component_id="metrics", component_property="data"),
        Output(component_id="predicted_test_labels", component_property="data"),
        Output(component_id="quantum_state_store", component_property="data"),
    ],
    inputs=[
        Input(component_id="model_parameters", component_property="data"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
        Input(component_id="reset_button", component_property="n_clicks"),
    ],
    state=[
        State(component_id="train_datastore", component_property="data"),
        State(component_id="test_datastore", component_property="data"),
        State(component_id="metrics", component_property="data"),
    ],
)
def evaluate_model(model_parameters,
                   num_qubits,
                   num_layers,
                   selected_data_set,
                   selected_task,
                   reset_clicks,
                   train_data,
                   test_data,
                   metrics):
    """
    Evaluates the quantum model's performance on both training and test datasets.
    """

    if ctx.triggered_id in reset_triggers:
        return dash.no_update

    is_regression = selected_task == 'regression'
    model_cls = QuantumReuploadingRegressor if is_regression else QuantumReuploadingClassifier
    qmodel = model_cls(name=selected_data_set, num_qubits=num_qubits, layers=num_layers)

    if model_parameters is not None:
        model_parameters = unserialize_model_dict(model_parameters)
        qmodel.load_model(model_parameters)
    else:
        return dash.no_update

    if train_data is not None and test_data is not None:
        df_train = pd.read_json(StringIO(train_data), orient='split')
        df_test = pd.read_json(StringIO(test_data), orient='split')
    else:
        return dash.no_update

    if is_regression:
        train_results = qmodel.evaluate(df_train[["x"]].values, df_train["y"].values)
        test_results = qmodel.evaluate(df_test[["x"]].values, df_test["y"].values)
        new_metric = {'loss': train_results["loss"],
                      'train_mse': train_results["loss"],
                      'test_mse':test_results["loss"],
                      }
        predictions = pd.Series(test_results["y_pred"].detach().numpy()).to_json(orient='values')
        states = serialize_quantum_states(num_qubits, test_results["states"])
    else:
        train_results = qmodel.evaluate(df_train[["x", "y"]].values, df_train["label"].values)
        test_results = qmodel.evaluate(df_test[["x", "y"]].values, df_test["label"].values)
        new_metric = {'loss': train_results["loss"],
                      'train_accuracy': train_results["accuracy"],
                      'test_accuracy': test_results['accuracy']
                      }
        predictions = pd.Series(test_results["predictions"].detach().numpy()).to_json(orient='values')
        states = serialize_quantum_states(num_qubits, test_results["states"])

    if metrics is not None:
        metrics = pd.read_json(StringIO(metrics), orient='split')
        metrics = pd.concat([metrics, pd.DataFrame([new_metric])], axis=0, ignore_index=True)
    else:
        metrics = pd.DataFrame([new_metric])

    return [metrics.to_json(orient='split'), predictions, states]


@qml_app.callback(
    [
        Output(component_id="decision_boundary_store", component_property="data"),
    ],
    inputs=[
        Input(component_id="model_parameters", component_property="data"),
        Input(component_id="select_num_qubits", component_property="value"),
        Input(component_id="select_num_layers", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
        Input(component_id="select_task_type", component_property="value"),
        Input(component_id="reset_button", component_property="n_clicks"),
        Input(component_id="select_noise_sigma", component_property="value"),
        Input(component_id="select_mc_runs", component_property="value"),
    ],
)
def evaluate_decision_boundary(model_parameters,
                               num_qubits,
                               num_layers,
                               selected_data_set,
                               selected_task,
                               reset_clicks,
                               noise_sigma,
                               mc_runs):
    """Compute decision data for plotting.
    - Classification: decision boundary heatmap Z as before.
    - Regression: dense x grid with ground truth y, predicted mean and std using MC weight noise.
    """

    if ctx.triggered_id in reset_triggers or model_parameters is None:
        return dash.no_update

    model_parameters = unserialize_model_dict(model_parameters)

    is_regression = selected_task == 'regression'

    if not is_regression:
        qcl = QuantumReuploadingClassifier(name=selected_data_set, num_qubits=num_qubits, layers=num_layers)
        qcl.load_model(model_parameters)

        # Compute decision boundary
        points_per_dim = 15
        x = np.linspace(-1.0, 1.0, points_per_dim)
        y = np.linspace(-1.0, 1.0, points_per_dim)
        xx, yy = np.meshgrid(x, y)
        X = torch.Tensor(np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())]))

        predictions, scores = qcl.predict(X)
        if num_qubits == 1:
            z = scores[0].detach().numpy()
        elif num_qubits == 2:
            z = scores[:, 0].detach().numpy()

        json_z = pd.Series(z).to_json(orient='values')
        return [json_z]

    # Regression path: compute ground truth and predictions on dense grid
    from data.datasets_torch import _fourier as fourier_fun
    qrg = QuantumReuploadingRegressor(name=selected_data_set, num_qubits=num_qubits, layers=num_layers)
    qrg.load_model(model_parameters)

    # Dense grid
    x_dense = torch.linspace(-1.0, 1.0, steps=400).unsqueeze(1)
    # Ground truth using same function as data
    _, y_true = fourier_fun(selected_data_set, samples=400, seed=0)

    # Use provided sigma and M for MC inference
    try:
        sigma = float(noise_sigma) if noise_sigma is not None else 0.0
    except Exception:
        sigma = 0.0
    try:
        M = int(mc_runs) if mc_runs is not None else 1
    except Exception:
        M = 1

    y_mean, y_std = qrg.predict_mc(x_dense, sigma=sigma, M=M)

    df_payload = pd.DataFrame({
        'x': x_dense.squeeze(1).numpy(),
        'y_true': y_true.numpy(),
        'y_mean': y_mean.detach().numpy(),
        'y_std': y_std.detach().numpy(),
    })
    return [df_payload.to_json(orient='split')]



# qml_app.clientside_callback(
#     ClientsideFunction(
#         namespace='qml_app',
#         function_name='updateDecisionPlot'
#     ),
#     Output(component_id="graph_decision", component_property="extendData"),
#     Input(component_id="decision_boundary_store", component_property="data"),
#     )
@qml_app.callback(
    [
        Output(component_id="graph_decision", component_property="extendData"),
    ],
    inputs=[
        Input(component_id="decision_boundary_store", component_property="data"),
        Input(component_id="test_datastore", component_property="data"),
        Input(component_id="train_datastore", component_property="data"),
    ],
)
def update_decision_plot(decision_boundary_store, test_data, train_data):
    """
    Updates the decision/regression plot depending on task type.
    - Classification: heatmap + test scatter overlay (handled by initial figure); extend heatmap.
    - Regression: overlay curves: ground truth, prediction mean, and uncertainty band.
    """

    if decision_boundary_store is None:
        return dash.no_update

    # Try classification path first (values array)
    try:
        z = pd.read_json(StringIO(decision_boundary_store), orient='values').values
        if test_data is None:
            return dash.no_update
        df_test = pd.read_json(StringIO(test_data), orient='split')

        n = int(np.sqrt(len(z)))
        x = np.linspace(-1, 1, n)
        z = z.reshape(n, n)

        decision_boundary_plot = make_decision_boundary_plot(x, x, z, df_test[["x", "y"]], df_test["label"])

        tracedata = [decision_boundary_plot["data"][0]]
        trace_idxs = [0]
        data_dict, max_points_dict = create_extendData_dicts(tracedata, keys=["x", "y", "z"])

        return [[data_dict, trace_idxs, max_points_dict]]
    except Exception:
        pass

    # Regression path: payload is a DataFrame with columns x, y_true, y_mean, y_std (orient='split')
    try:
        df_payload = pd.read_json(StringIO(decision_boundary_store), orient='split')
        df_train = pd.read_json(StringIO(train_data), orient='split')
    except Exception:
        return dash.no_update

    x = df_payload["x"].values.tolist()
    y_true = df_payload["y_true"].values.tolist()
    y_mean = df_payload["y_mean"].values.tolist()
    y_std = df_payload["y_std"].values
    # Use 2-sigma band
    y_lower = (df_payload["y_mean"] - 2.0 * y_std).values.tolist()
    y_upper = (df_payload["y_mean"] + 2.0 * y_std).values.tolist()

    train_x = df_train["x"].values.tolist()
    train_y = df_train["y"].values.tolist()

    data_dict = {
        "x": [x, x, x, x, train_x],
        "y": [y_true, y_mean, y_lower, y_upper, train_y],
    }
    max_points_dict = {
        "x": [len(x)] * 4 + [len(train_x)],
        "y": [len(x)] * 4 + [len(train_y)],
    }
    trace_idxs = [0, 1, 2, 3, 4]

    return [[data_dict, trace_idxs, max_points_dict]]


@qml_app.callback(
    [
        Output(component_id="train_datastore", component_property="data"),
        Output(component_id="test_datastore", component_property="data"),
        Output(component_id="graph_data_set", component_property="figure"),
    ],
    [
        Input(component_id="select_task_type", component_property="value"),
        Input(component_id="select_data_set", component_property="value"),
    ],
)
def update_data(selected_task: str, selected_data_set: str):
    """
    Creates and updates training and test datasets based on the selected dataset option.
    """

    if selected_task == 'regression':
        # Create full domain and then mask out gaps for training
        x_full, y_full = create_dataset(selected_data_set, samples=60, seed=42)
        x_full_np = x_full.squeeze(1).numpy()
        y_full_np = y_full.numpy()

        y_full_noisy = y_full_np + np.random.normal(0, 0.075, size=y_full_np.shape)

        # Define gaps: small in-domain gap and tail gap
        in_gap = (x_full_np > -0.1) & (x_full_np < 0.1)
        tail_gap = (x_full_np > 0.8)
        train_mask = ~(in_gap | tail_gap)

        # Train data: masked samples
        df_train = pd.DataFrame({"x": x_full_np[train_mask], "y": y_full_noisy[train_mask]})
        # Test data: full domain
        df_test = pd.DataFrame({"x": x_full_np, "y": y_full_np})

        data_plot = make_regression_data_plot(x_full_np, y_full_np, y_full_noisy, train_mask=train_mask)
        return [df_train.to_json(orient='split'), df_test.to_json(orient='split'), data_plot]

    # Classification datasets
    x_train, y_train = create_dataset(selected_data_set, samples=500, seed=42)
    x_test, y_test = create_dataset(selected_data_set, samples=300, seed=43)

    data_plot = make_data_plot(x_test, y_test)

    df_train = pd.DataFrame(x_train.numpy(), columns=["x", "y"])
    df_train["label"] = y_train.numpy()

    df_test = pd.DataFrame(x_test.numpy(), columns=["x", "y"])
    df_test["label"] = y_test.numpy()

    return [df_train.to_json(orient='split'), df_test.to_json(orient='split'), data_plot]




qml_app.clientside_callback(
     ClientsideFunction(
         namespace='qml_app',
         function_name='updateFinalPlot'
     ),
     Output(component_id="graph_final_state", component_property="extendData"),
     Input(component_id="quantum_state_store", component_property="data"),
     State(component_id="select_num_qubits", component_property="value"),
     State(component_id="select_data_set", component_property="value"),
     State(component_id="select_task_type", component_property="value"),
     State(component_id="test_datastore", component_property="data")
     )


qml_app.clientside_callback(
    ClientsideFunction(
        namespace='qml_app',
        function_name='updateLayerStatePlots'
    ),
    Output(component_id="graph_model", component_property="extendData"),
    Input(component_id="quantum_state_store", component_property="data"),
    State(component_id="select_num_qubits", component_property="value"),
    State(component_id="select_num_layers", component_property="value"),
    State(component_id="select_data_set", component_property="value"),
    State(component_id="select_task_type", component_property="value"),
    State(component_id="test_datastore", component_property="data")
    )


qml_app.clientside_callback(
    ClientsideFunction(
        namespace='qml_app',
        function_name='updateTrainingMetrics'
    ),
    Output('graph_loss_acc', 'extendData'),
    Input('metrics', 'data')
)


qml_app.clientside_callback(
    ClientsideFunction(
        namespace='qml_app',
        function_name='updateResultsPlot'
    ),
    Output(component_id="graph_results", component_property="extendData"),
    Input(component_id="predicted_test_labels", component_property="data"),
    State(component_id="select_data_set", component_property="value"),
    State(component_id="select_task_type", component_property="value"),
    State(component_id="test_datastore", component_property="data")
    )
