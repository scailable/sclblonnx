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

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == current_name:
                node.input[index] = new_name

    return graph


def rename_bbox_output(graph, bboxes_output_name, format, class_list):
    """ Rename a bbox output of a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        bboxes_output_name: String, the current output name of bounding boxes.
        format: Format of output, choose among :
            "xy" if (x1, y1, x2, y2)
            "xyc" if (x1, y1, x2, y2, class)
            "xysc" if (x1, y1, x2, y2, score, class)
        class_list: List of classes.
    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = ""
    if format == "xy":
        new_name = "bboxes-format:xyxy;"
    elif format == "xyc":
        new_name = "bboxes-format:xyxyc;"
    elif format == "xysc":
        new_name = "bboxes-format:xyxysc;"
    else:
        print("Format input is incorrect, it must be 'xy', 'xyc' or 'xysc'")
        return False
    for index, name in enumerate(class_list):
        new_name = new_name + str(index) + ':' + name + ";"
    new_name =  new_name[0:-1]

    for output in graph.output:
        if output.name == bboxes_output_name:
            output.name =  new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == bboxes_output_name:
                node.output[index] = new_name

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == bboxes_output_name:
                node.input[index] = new_name
        return graph


def rename_barcode_output(graph, barcode_output_name):
    """ Rename a barcode bbox output of a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        barcode_output_name: String, the current name of bounding-boxes output for barcodes

    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    for output in graph.output:
        if output.name == barcode_output_name:
            output.name =  "barcode_bboxes-format:xyxy"
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == barcode_output_name:
                node.output[index] = "barcode_bboxes-format:xyxy"

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == barcode_output_name:
                node.input[index] = "barcode_bboxes-format:xyxy"
    return graph


def rename_licenseplate_output(graph, licenseplate_output_name, format):
    """ Rename a licenseplate bbox output of a graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        licenseplate_output_name: String, the current output name of licenseplate bounding boxes.
        format: Format of output, choose among :
            "xy" if (x1, y1, x2, y2)
            "xyxyxsxyxyxyxy" if (x1, y1, x2, y2, score, ... landmark coordinates)
    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = ""
    if format == "xy":
        new_name = "licenseplate_bboxes-format:xyxy"
    elif format == "xys":
        new_name = "licenseplate_bboxes-format:xyxyxsxyxyxyxy"
    else:
        print("Format input is incorrect, it must be 'xy' or 'xys'")
        return False

    for output in graph.output:
        if output.name == licenseplate_output_name:
            output.name =  new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == licenseplate_output_name:
                node.output[index] = new_name

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == licenseplate_output_name:
                node.input[index] = new_name
    return graph


def rename_class_probabilities_output(graph, output_name, class_list):
    """ Rename the output of a model that generates probabilities per class

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        output_name: String, the current output name of the graph
        class_list: List of classes.
    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = "scores-"
    for index, name in enumerate(class_list):
        new_name = new_name + str(index) + ':' + name + ";"
    new_name =  new_name[0:-1]

    for output in graph.output:
        if output.name == output_name:
            output.name =  new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == output_name:
                node.output[index] = new_name

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == output_name:
                node.input[index] = new_name
    return graph


def rename_object_count_output(graph, output_name, class_list):
    """ Rename the output of a model that generates number of objects per class

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        output_name: String, the current output name of the graph
        class_list: List of classes.
    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = "counts-"
    for index, name in enumerate(class_list):
        new_name = new_name + str(index) + ':' + name + ";"
    new_name =  new_name[0:-1]

    for output in graph.output:
        if output.name == output_name:
            output.name =  new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == output_name:
                node.output[index] = new_name

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == output_name:
                node.input[index] = new_name
    return graph


def rename_alarm_output(graph, output_name, class_list):
    """ Rename the output of a model that generates an alarm based
    on number of objects per class

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        output_name: String, the current output name of the graph
        class_list: List of classes.
    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = "alarm-"
    for index, name in enumerate(class_list):
        new_name = new_name + str(index) + ':' + name + ";"
    new_name =  new_name[0:-1]

    for output in graph.output:
        if output.name == output_name:
            output.name =  new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == output_name:
                node.output[index] = new_name

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == output_name:
                node.input[index] = new_name
    return graph


def rename_linecrossing_bboxes_output(graph, output_name):
    """ Rename the output of a model to be post-processed
    using the line-crossing counter

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        output_name: String, the current output name of the graph
    Returns:
        The changed graph.
    """

    if type(graph) is not xpb2.GraphProto:
        _print("graph is not a valid ONNX graph.")
        return False

    found = False
    new_name = "linecrossing_bboxes-format:xyxysc"

    for output in graph.output:
        if output.name == output_name:
            output.name =  new_name
            found = True
    if not found:
        _print("Unable to found the output by name.")
        return False

    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == output_name:
                node.output[index] = new_name

        # Handle the case when the output is fed to another node
        for index, name in enumerate(node.input):
            if name == output_name:
                node.input[index] = new_name
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
