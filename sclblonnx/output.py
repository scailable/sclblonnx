from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2

from sclblonnx.utils import _parse_element, _value, _data_type, _print


# list_outputs list all outputs in a graph
def list_outputs(graph: xpb2.GraphProto):
    """ Tries to list the outputs of a given graph.

    Args:
        graph the ONNX graph
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    i = 1
    for elem in graph.output:
        name, dtype, shape = _parse_element(elem)
        print("Output {}: Name: '{}', Type: {}, Dimension: {}".format(i, name, dtype, shape))
        i += 1

    if i == 1:
        print("No outputs found.")

    return True


# add_output adds an output to a graph
def add_output(
        graph: xpb2.GraphProto,
        name: str,
        data_type: str,
        dimensions: [],
        **kwargs):
    """ Add an output to a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: String, the name of the input as used to determine the graph topology.
        data_type: String, the data type of the input. Run list_data_types() for an overview.
        dimensions: List[] specifying the dimensions of the input.
        **kwargs

    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        graph.output.append(xhelp.make_tensor_value_info(name, dtype, dimensions, **kwargs), **kwargs)
    except Exception as e:
        _print("Unable to add the input: " + str(e))
        return False
    return graph


# rename_output renames an existing output
def rename_output(graph, current_name, new_name):
    """ Rename an output to a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        current_name: String, the current output name.
        new_name: String, the name desired output name.

    Returns:
        The changed graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    for output in graph.output:
        if output.name == current_name:
            output.name = new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == current_name:
                node.output[index] = new_name

    return graph


# replace_output replaces an existing output
def replace_output(
        graph: xpb2.GraphProto,
        name: str,
        data_type: str,
        dimensions: [],
        **kwargs):
    """ Changes an existing output of a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: String, the name of the output as used to determine the graph topology.
        data_type: String, the data type of the output. Run list_data_types() for an overview.
        dimensions: List[] specifying the dimensions of the input.
        **kwargs

    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return graph

    # Remove the named output
    found = False
    try:
        for elem in graph.output:
            if elem.name == name:
                graph.output.remove(elem)
                found = True
    except Exception as e:
        _print("Unable to iterate the outputs. " + str(e))
        return False
    if not found:
        _print("Unable to find the output by name.")

    # Create the new value
    try:
        val = _value(name, data_type, dimensions, **kwargs)
    except Exception as e:
        _print("Unable to create value. " + str(e))
        return False

    # Add the value to the output
    try:
        graph.output.append(val, **kwargs)
    except Exception as e:
        _print("Unable to add the output: " + str(e))
        return False

    return graph


# delete_output deletes an existing output
def delete_output(
            graph: xpb2.GraphProto,
            name: str):
    """ Removes an existing output of a graph by name

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: String, the name of the output as used to determine the graph topology.


    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return graph

    # Remove the named output
    found = False
    try:
        for elem in graph.output:
            if elem.name == name:
                graph.output.remove(elem)
                found = True
    except Exception as e:
        _print("Unable to iterate the outputs. " + str(e))
        return False
    if not found:
        _print("Unable to find the output by name.")
        return False

    return graph
