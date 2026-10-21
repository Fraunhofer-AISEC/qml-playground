

def create_extendData_dicts(tracedata, keys=["x", "y"]):
    """Create data dictionaries for extending Plotly traces.

    This function processes a list of Plotly trace objects and converts them into the format
    required by Plotly's extendData method. It extracts the specified keys (e.g., 'x', 'y', 'z')
    from each trace and organizes them into dictionaries that can be used for efficient updates
    of Plotly graphs.

    Args:
        tracedata (list): List of Plotly trace objects to process
        keys (list, optional): List of keys to extract from each trace. Defaults to ["x", "y"].

    Returns:
        tuple: A pair of dictionaries:
            - data_dict: Contains lists of data points for each key
            - max_points_dict: Contains the number of points for each key and trace
    """
    data_dict = { k:list() for k in keys }
    max_points_dict = { k:list() for k in keys }

    for trace in tracedata:
        trace_json = trace.to_plotly_json()
        for k in keys:
            if k in trace_json:
                data_dict[k].append(trace_json[k])
                max_points_dict[k].append(len(trace_json[k]))

    return data_dict, max_points_dict