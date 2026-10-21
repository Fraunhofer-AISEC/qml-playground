// clientside.js

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    qml_app: {

        /**
         * Updates the training metrics plot with the latest performance data.
         *
         * This function processes metric data from the model training process and
         * prepares it for display in the performance plot. It extracts the most recent
         * metrics (loss, training accuracy, and test accuracy) and formats them for
         * Plotly's extendData method.
         *
         * @param {string} metric_store - JSON string containing metrics data with index and data fields
         * @returns {Array} Array containing [data_dict, trace_indices, max_points] for Plotly.extendData
         */
        updateTrainingMetrics: function(metric_store) {

            if (!metric_store) {
                return window.dash_clientside.no_update;
            }

            let metrics;
            try {
                metrics = JSON.parse(metric_store);
            } catch (err) {
                console.warn("Failed to parse metric_store:", err);
                return window.dash_clientside.no_update;
            }

            let lastIndex = metrics.index.length - 1;
            if (lastIndex < 0) {
                // No rows in metrics, so do nothing
                return window.dash_clientside.no_update;
            }

            let lastRow = metrics.data[lastIndex];
            // example: if lastRow = [val1, val2, val3]

            let x = [[lastIndex], [lastIndex], [lastIndex]];
            let y = lastRow.map((val) => [val]);

            let data_dict = {
                x: x,
                y: y
            };

            return [data_dict, [0, 1, 2], 100];
    },


    /**
     * Updates intermediate layer state plots in the quantum model visualization.
     *
     * This function processes quantum states from each layer of the quantum circuit
     * and prepares them for visualization in the model plot. It handles both single-qubit
     * states (visualized on the Bloch sphere) and multi-qubit states (visualized in a
     * probability simplex). Each class of data points is colored differently.
     *
     * @param {string} quantum_state_store - JSON string containing quantum states for each layer
     * @param {number} num_qubits - Number of qubits in the quantum system (1 or 2)
     * @param {number} num_layers - Number of layers in the quantum circuit
     * @param {string} selected_data_set - Name of the selected dataset
     * @param {string} selected_task - Name of the selected task
     * @param {string} test_data - JSON string containing test data with points and labels
     * @returns {Array} Array containing [data_dict, trace_indices, max_points] for Plotly.extendData
     */
    updateLayerStatePlots: function(quantum_state_store, num_qubits, num_layers, selected_data_set, selected_task, test_data) {

        if (!quantum_state_store || !test_data || !selected_task) {
            return window.dash_clientside.no_update;
        }

        let dfTest;
        try {
            dfTest = JSON.parse(test_data);
        } catch (err) {
            console.warn("Failed to parse test_data:", err);
            return window.dash_clientside.no_update;
        }

        let layer_states;
        try {
            layer_states = JSON.parse(quantum_state_store);
        } catch (err) {
            console.warn("Failed to parse quantum_state_store:", err);
            return window.dash_clientside.no_update;
        }

        let state_dim = num_qubits === 1 ? 3 : Math.pow(2, num_qubits);

        // Reshape layer_states from [num_layers, num_states * state_dim]
        // to [num_layers, num_states, state_dim]
        let num_states = layer_states[0].length / state_dim; // Assuming uniform structure

        layer_states = layer_states.map(layer =>
            Array.from({ length: num_states }, (_, i) =>
                layer.slice(i * state_dim, (i + 1) * state_dim)
            )
        );

        if (typeof selected_task === 'string' && selected_task === 'regression') {
            labels = dfTest.data.map(row => 1);
            numClasses = 2;
        }
        else {
            labels = dfTest.data.map(row => row[dfTest.columns.indexOf("label")]);
            numClasses = Math.max(...labels) + 1;
        }

        let traceOffset = 7;
        let tracesPerSubfigure = traceOffset + 4;

        let stateTraces = [];
        let traceIdx = [];

        for (let l = 0; l < num_layers - 1; l++) {

            let current_layer = layer_states[l]

            for (let c = 0; c < numClasses; c++) {
                let indices = labels.map((val, idx) => (val === c ? idx : -1)).filter(idx => idx !== -1);
                let states = indices.map(idx => current_layer[idx]);

                traceIdx.push(l * tracesPerSubfigure + traceOffset + c)

                // Case: Single Qubit (Real-Valued Bloch Vectors)
                if (num_qubits === 1) {

                    let points = {
                        x: states.map(state => state[0]),
                        y: states.map(state => state[1]),
                        z: states.map(state => state[2])
                    };
                    stateTraces.push(points);
                }
                // Case: Multi-Qubit (Complex States -> Probability Simplex Mapping)
                else if (num_qubits > 1) {

                    let probs = states.map(state =>
                    state.map(complex => Math.pow(complex.real, 2) + Math.pow(complex.imag, 2))  // |a + bi|^2 = a^2 + b^2
                    );

                    let points = {
                        x: probs.map(prob => prob[1]),  // Probabilities for |01⟩ state
                        y: probs.map(prob => prob[2]),  // Probabilities for |10⟩ state
                        z: probs.map(prob => prob[0])   // Probabilities for |00⟩ state
                    };

                    stateTraces.push(points);

                }
                 else {
                    console.warn("Invalid number of qubits:", num_qubits);
                    return window.dash_clientside.no_update;
                }
            }

        }

        let dataDict = { x: [], y: [], z: [] };
        let maxPointsDict = { x: [], y: [], z: [] };

        stateTraces.forEach(trace => {
            dataDict.x.push(trace.x);
            dataDict.y.push(trace.y);
            dataDict.z.push(trace.z);

            maxPointsDict.x.push(trace.x.length);
            maxPointsDict.y.push(trace.y.length);
            maxPointsDict.z.push(trace.z.length);
        });

        return [dataDict, traceIdx, maxPointsDict];
        },


        /**
         * Updates the final state plot with quantum states from the last circuit layer.
         *
         * This function extracts the quantum states from the final layer of the quantum circuit
         * and prepares them for visualization in the state space. It processes both single-qubit
         * states (Bloch sphere representation) and multi-qubit states (probability simplex),
         * grouping and coloring data points by their class labels.
         *
         * @param {string} quantum_state_store - JSON string containing quantum states for each layer
         * @param {number} num_qubits - Number of qubits in the quantum system (1 or 2)
         * @param {string} selected_data_set - Name of the selected dataset
         * @param {string} selected_task - Name of the selected task
         * @param {string} test_data - JSON string containing test data with points and labels
         * @returns {Array} Array containing [data_dict, trace_indices, max_points] for Plotly.extendData
         */
        updateFinalPlot: function(quantum_state_store, num_qubits, selected_data_set, selected_task, test_data) {

            if (!quantum_state_store || !test_data || !selected_task) {
                return window.dash_clientside.no_update;
            }

            let dfTest;
            try {
                dfTest = JSON.parse(test_data);
            } catch (err) {
                console.warn("Failed to parse test_data:", err);
                return window.dash_clientside.no_update;
            }

            let layer_states;
            try {
                layer_states = JSON.parse(quantum_state_store);
            } catch (err) {
                console.warn("Failed to parse quantum_state_store:", err);
                return window.dash_clientside.no_update;
            }

            let state_dim = num_qubits === 1 ? 3 : Math.pow(2, num_qubits);

            // Reshape layer_states from [num_layers, num_states * state_dim]
            // to [num_layers, num_states, state_dim]
            let num_layers = layer_states.length;
            let num_states = layer_states[0].length / state_dim; // Assuming uniform structure

            layer_states = layer_states.map(layer =>
                Array.from({ length: num_states }, (_, i) =>
                    layer.slice(i * state_dim, (i + 1) * state_dim)
                )
            );

            let lastLayer = layer_states[layer_states.length - 1];

            let labels;
            let numClasses;
            if (typeof selected_task === 'string' && selected_task === 'regression') {
                labels = dfTest.data.map(row => 1);
                numClasses = 2;
            }
            else {
                labels = dfTest.data.map(row => row[dfTest.columns.indexOf("label")]);
                numClasses = Math.max(...labels) + 1;
            }

            let stateTraces = [];

            if (num_qubits === 1) {
                // Case: Single Qubit (Real-Valued Bloch Vectors)
                for (let c = 0; c < numClasses; c++) {
                    let indices = labels.map((val, idx) => (val === c ? idx : -1)).filter(idx => idx !== -1);
                    let states = indices.map(idx => lastLayer[idx]);

                    let points = {
                        x: states.map(state => state[0]),
                        y: states.map(state => state[1]),
                        z: states.map(state => state[2])
                    };
                    stateTraces.push(points);
                }
            } else if (num_qubits > 1) {
                // Case: Multi-Qubit (Complex States -> Probability Simplex Mapping)
                for (let c = 0; c < numClasses; c++) {
                    let indices = labels.map((val, idx) => (val === c ? idx : -1)).filter(idx => idx !== -1);
                    let cStates = indices.map(idx => lastLayer[idx]);

                    let probs = cStates.map(state =>
                        state.map(complex => Math.pow(complex.real, 2) + Math.pow(complex.imag, 2))  // |a + bi|^2 = a^2 + b^2
                    );

                    let points = {
                        x: probs.map(prob => prob[1]),  // Probabilities for |01⟩ state
                        y: probs.map(prob => prob[2]),  // Probabilities for |10⟩ state
                        z: probs.map(prob => prob[0])   // Probabilities for |00⟩ state
                    };
                    stateTraces.push(points);
                }
            } else {
                console.warn("Invalid number of qubits:", num_qubits);
                return window.dash_clientside.no_update;
            }

            let traceOffset = 7;

            let traceIdx = stateTraces.map((_, i) => traceOffset + i);

            let dataDict = { x: [], y: [], z: [] };
            let maxPointsDict = { x: [], y: [], z: [] };

            stateTraces.forEach(trace => {
                dataDict.x.push(trace.x);
                dataDict.y.push(trace.y);
                dataDict.z.push(trace.z);

                maxPointsDict.x.push(trace.x.length);
                maxPointsDict.y.push(trace.y.length);
                maxPointsDict.z.push(trace.z.length);
            });

            return [dataDict, traceIdx, maxPointsDict];
        },

        /**
         * Updates the results plot with model predictions and accuracy visualization.
         *
         * This function processes the model's predictions on the test dataset and prepares
         * two main visualizations:
         * 1. Data points colored according to their true class labels
         * 2. Data points colored to indicate correct (green) or incorrect (red) predictions
         *
         * The resulting plots allow visual assessment of the model's classification performance
         * and error patterns.
         *
         * @param {string} prediction_store - JSON string containing model predictions
         * @param {string} selected_data_set - Name of the selected dataset
         * @param {string} selected_task - Name of the selected task
         * @param {string} test_data - JSON string containing test data with points and labels
         * @returns {Array} Array containing [data_dict, trace_indices, max_points] for Plotly.extendData
         */
        updateResultsPlot: function(prediction_store, selected_data_set, selected_task, test_data) {

            if (!prediction_store || !test_data || !selected_task) {
                return window.dash_clientside.no_update;
            }

            let dfTest;
            try {
                dfTest = JSON.parse(test_data);
            } catch (err) {
                console.warn("Failed to parse test_data:", err);
                return window.dash_clientside.no_update;
            }

            let predictions;
            try {
                predictions = JSON.parse(prediction_store);
            } catch (err) {
                console.warn("Failed to parse prediction_store:", err);
                return window.dash_clientside.no_update;
            }

            // Regression branch: fourier_* datasets
            if (typeof selected_task === 'string' && selected_task === 'regression') {
                // dfTest has columns ["x","y"]
                const xIdx = dfTest.columns.indexOf("x");
                const yIdx = dfTest.columns.indexOf("y");
                if (xIdx === -1 || yIdx === -1) {
                    return window.dash_clientside.no_update;
                }
                const x = dfTest.data.map(row => row[xIdx]);
                const yTrue = dfTest.data.map(row => row[yIdx]);
                if (!Array.isArray(predictions) || predictions.length !== yTrue.length) {
                    // Length mismatch; do nothing
                    return window.dash_clientside.no_update;
                }
                const residuals = predictions.map((yp, i) => yp - yTrue[i]);
                const dataDict = { x: [x], y: [residuals] };
                const maxPoints = { x: [x.length], y: [residuals.length] };
                return [dataDict, [0], maxPoints];
            }

            // Classification branch (default)
            let labels = dfTest.data.map(row => row[dfTest.columns.indexOf("label")]);
            let xPoints = dfTest.data.map(row => row[dfTest.columns.indexOf("x")]);
            let yPoints = dfTest.data.map(row => row[dfTest.columns.indexOf("y")]);

            let numClasses = Math.max(...labels) + 1;

            let traces = []
            let traceIdxs = []

            for (let c = 0; c < numClasses; c++) {
                let indices = labels.map((val, idx) => (val === c ? idx : -1)).filter(idx => idx !== -1);

                let points = {
                        x: indices.map(idx => xPoints[idx]),
                        y: indices.map(idx => yPoints[idx])
                    };

                traces.push(points);
                traceIdxs.push(c)
            }

            const checks = predictions.map((pred, i) => pred === labels[i]);

            let WrongPoints = {
                x: xPoints.filter((_, i) => !checks[i]),
                y: yPoints.filter((_, i) => !checks[i])
            };
            traces.push(WrongPoints);
            traceIdxs.push(4);

            let RightPoints = {
               x: xPoints.filter((_, i) => checks[i]),
               y: yPoints.filter((_, i) => checks[i])
            };
            traces.push(RightPoints);
            traceIdxs.push(5);

            let dataDict = { x: [], y: []};
            let maxPointsDict = { x: [], y: []};

            traces.forEach(trace => {
                dataDict.x.push(trace.x);
                dataDict.y.push(trace.y);

                maxPointsDict.x.push(trace.x.length);
                maxPointsDict.y.push(trace.y.length);
            });

            return [dataDict, traceIdxs, maxPointsDict];
        },

        updateDecisionPlot: function(decision_boundary_store) {

            if (!decision_boundary_store) {
                return window.dash_clientside.no_update;
            }

            let zValues;
            try {
                zValues = JSON.parse(decision_boundary_store);
            } catch (err) {
                console.warn("Failed to parse prediction_store:", err);
                return window.dash_clientside.no_update;
            }

            const numZValues = zValues.length;
            const n = Math.round((Math.sqrt(numZValues)));
            const step = 2.0 / (n - 1);

            let x = Array.from({ length: n }, (_, i) => 1.0 + i * step);
            let y = Array.from({ length: n }, (_, i) => 1.0 + i * step);
            let z = Array.from({ length: n }, (_, i) =>
                                zValues.slice(i * n, (i + 1) * n)
                             );

            let dataDict = {
                x: x,
                y: y,
                z: z,
            }

            return [dataDict, 0, n];
        },

    }
});
