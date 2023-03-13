from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2

from sclblonnx.utils import _value, _data_type, _parse_element, _print
import onnx

# list_inputs lists all inputs of a graph
def list_inputs(graph: xpb2.GraphProto):
    """ Tries to list the inputs of a given graph.

    Args:
        graph the ONNX graph
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    i = 1
    for elem in graph.input:
        name, dtype, shape = _parse_element(elem)
        print("Input {}: Name: '{}', Type: {}, Dimension: {}".format(i, name, dtype, shape))
        i += 1

    if i == 1:
        print("No inputs found.")

    return True


# add_input adds an input to a graph
def add_input(
        graph: xpb2.GraphProto,
        name: str,
        data_type: str,
        dimensions: [],
        **kwargs):
    """ Add an input to a graph

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
        graph.input.append(xhelp.make_tensor_value_info(name, dtype, dimensions, **kwargs), *kwargs)
    except Exception as e:
        _print("Unable to add the input: " + str(e))
        return False
    return graph


# rename_input renames an input
def rename_input(graph, current_name, new_name):
    """ Rename an input to a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        current_name: String, the current input name.
        new_name: String, the name desired input name.

    Returns:
        The changed graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    for input in graph.input:
        if input.name == current_name:
            input.name = new_name
            found = True
    if not found:
        _print("Unable to find the input to rename.")
        return False

    # And rename it in every nodes that takes this as input:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == current_name:
                node.input[index] = new_name

    return graph


# rename_input_image renames an image input
def rename_input_image(graph, image_input_name):
    """ Rename an image input

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        image_input_name: String, the current image input name.

    Returns:
        The changed graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    for input in graph.input:
        if input.name == image_input_name:
            input.name = "image-"
            found = True
    if not found:
        _print("Unable to find the input to rename.")
        return False

    # And rename it in every nodes that takes this as input:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == image_input_name:
                node.input[index] = "image-"
    return graph

# rename_input_image renames a binary mask input
def rename_input_mask(graph, mask_input_name):
    """ Rename a binary mask input

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        mask_input_name: String, the current mask input name.

    Returns:
        The changed graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    for input in graph.input:
        if input.name == mask_input_name:
            input.name = "mask-"
            found = True
    if not found:
        _print("Unable to find the input to rename.")
        return False

    # And rename it in every nodes that takes this as input:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == mask_input_name:
                node.input[index] = "mask-"
    return graph


# rename_input_image renames a threshold input
def rename_input_threshold(graph, threshold_input_name, class_list):
    """ Enable a model to raise an alarm when number of
    occurrences of an object is above the threshold

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        threshold_input_name: String, the current input name of threshold.
        class_list: List of classes.

    Returns:
        The changed graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = "thresholds-"
    for index, name in enumerate(class_list):
        new_name = new_name + str(index) + ':' + name + ";"
    new_name =  new_name[0:-1]

    for input in graph.input:
        if input.name == threshold_input_name:
            input.name = new_name
            found = True
    if not found:
        _print("Unable to find the input to rename.")
        return False

    # And rename it in every nodes that takes this as input:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == threshold_input_name:
                node.input[index] = new_name
    return graph

def rename_input_sensor(graph, sensor_input_name):
    """ Rename a sensor input

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        current_name: String, the current input name.

    Returns:
        The changed graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    for input in graph.input:
        if input.name == sensor_input_name:
            input.name = "sensor-"
            found = True
    if not found:
        _print("Unable to find the input to rename.")
        return False

    # And rename it in every nodes that takes this as input:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == sensor_input_name:
                node.input[index] = "sensor-"
    return graph


# replace input replaces and existing input
def replace_input(
        graph: xpb2.GraphProto,
        name: str,
        data_type: str,
        dimensions: [],
        **kwargs):
    """ Changes an existing input in a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: String, the name of the input as used to determine the graph topology.
        data_type: String, the data type of the input. Run list_data_types() for an overview.
        dimensions: List[] specifying the dimensions of the input.,
        **kwargs

    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return graph

    # Remove the named input
    found = False
    try:
        for elem in graph.input:
            if elem.name == name:
                graph.input.remove(elem)
                found = True
    except Exception as e:
        _print("Unable to iterate the inputs. " + str(e))
        return False
    if not found:
        _print("Unable to find the input by name.")

    # Create the new value
    try:
        val = _value(name, data_type, dimensions, **kwargs)
    except Exception as e:
        _print("Unable to create value. " + str(e))
        return False

    # Add the value to the input
    try:
        graph.input.append(val, *kwargs)
    except Exception as e:
        _print("Unable to add the input: " + str(e))
        return False

    return graph


# delete_input deletes an existing input
def delete_input(
        graph: xpb2.GraphProto,
        name: str):
    """ Removes an existing input of a graph by name

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: String, the name of the input as used to determine the graph topology.

    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        return graph

    # Remove the named output
    found = False
    try:
        for elem in graph.input:
            if elem.name == name:
                graph.input.remove(elem)
                found = True
    except Exception as e:
        _print("Unable to iterate the inputs. " + str(e))
        return False
    if not found:
        _print("Unable to find the input by name.")
        return False

    return graph
