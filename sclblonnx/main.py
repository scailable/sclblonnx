import argparse
import json
import os
import subprocess

import numpy as np
import onnxoptimizer
from onnx import ModelProto as xmp
from onnx import __version__ as xversion
from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2
from onnx import save as xsave
from onnx import checker
from packaging import version
from onnxsim import simplify
import sclblonnx._globals as glob
import onnxruntime as xrt


# empty_graph creates an empty graph
def empty_graph(
        _default_name: str = "sclblgraph"):
    """ empty_graph returns an empty graph

    Note, an empty graph does not pass the check() as it does not contain input and output.

    Args:
        _default_name: Graph name, default sclblgraph

    Returns:
        An empty graph
    """
    try:
        graph = xpb2.GraphProto(name=_default_name)
    except Exception as e:
        print("Unable to create graph: " + str(e))
        return False
    return graph


def node(
        op_type: str,
        inputs: [],
        outputs: [],
        name: str = "",
        **kwargs):
    """ Create a new node

    Args:
        op_type: Operator type, see https://github.com/onnx/onnx/blob/master/docs/Operators.md
        inputs: [] list of inputs (names to determine the graph topology)
        outputs: [] list of outputs (names to determine the graph topology)
        name: The name of this node (Optional)
    """
    if not name:
        name = "sclbl-onnx-node" + str(glob.NODE_COUNT)
        glob.NODE_COUNT += 1

    try:
        node = xhelp.make_node(op_type, inputs, outputs, name, **kwargs)
    except Exception as e:
        print("Unable to create node: " + str(e))
        return False
    return node


def add_node(
        graph: xpb2.GraphProto,
        node: xpb2.NodeProto,
        **kwargs):
    """ Add node appends a node to graph g and returns the extended graph

    Prints a message and returns False if fails.

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        node: A node, onnx.onnx_ml_pb2.NodeProto.

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("graph is not a valid ONNX graph.")
        return False
    if type(node) is not xpb2.NodeProto:
        if not glob.SILENT:
            print("node is not a valid ONNX node.")
        return False
    try:
        graph.node.append(node, **kwargs)
    except Exception as e:
        print("Unable to extend graph: " + str(e))
        return False
    return graph


def add_nodes(
        graph: xpb2.GraphProto,
        nodes: [xpb2.NodeProto],
        **kwargs):
    """ Add a list of nodes appends a node to graph g and returns the extended graph

    Prints a message and returns False if fails.

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        nodes: A list of nodes, [onnx.onnx_ml_pb2.NodeProto].

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("graph is not a valid ONNX graph.")
        return False

    for node in nodes:
        graph = add_node(graph, node, **kwargs)
        if not graph:
            return False
    return graph


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

    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("g is not a valid ONNX graph.")
        return False

    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        graph.input.append(xhelp.make_tensor_value_info(name, dtype, dimensions, **kwargs), *kwargs)
    except Exception as e:
        print("Unable to add the input: " + str(e))
        return False
    return graph


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

    Returns:
        The extended graph.

    """
    if type(graph) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("graph is not a valid ONNX graph.")
        return False

    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        graph.output.append(xhelp.make_tensor_value_info(name, dtype, dimensions, **kwargs), **kwargs)
    except Exception as e:
        print("Unable to add the input: " + str(e))
        return False
    return graph


# add_constant adds a constant node to a graph
def add_constant(
        graph: xpb2.GraphProto,
        name: str,
        value: np.array,
        data_type: str,
        **kwargs):
    """ Add a constant node to an existing graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: Name of the (output value of the) constant node to determine the graph topology
        value: Values of the node (as a np.array)
        data_type: Data type of the node

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("graph is not a valid ONNX graph.")
        return False

    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        constant_node = xhelp.make_node('Constant', inputs=[], outputs=[name], name=name + "-constant",
                                        value=xhelp.make_tensor(name=name + "-values", data_type=dtype,
                                                                dims=value.shape, vals=value.flatten()), **kwargs)
    except Exception as e:
        print("Unable to create the constant node: " + str(e))
        return False

    graph = add_node(graph, constant_node, **kwargs)
    if not graph:
        print("Unable to add constant node to graph.")
        return False
    return graph


def run(
        graph: xpb2.GraphProto,
        inputs: {},
        outputs: [],
        _tmpfile: str = ".tmp.onnx"):
    """ run executes a give graph with the given input and returns the output

    Args:
        graph: The onnx graph
        inputs: an object with the named inputs; please check the data types
        outputs: list of named outputs
        _tmpfile: String the temporary filename for the onnx file to run.

    Returns:
        The result (or False if it fails somewhere)
        """

    store = graph_to_file(graph, _tmpfile)
    if not store:
        print("Unable to store model for evaluation.")
        return False

    try:
        sess = xrt.InferenceSession(_tmpfile)
        out = sess.run(outputs, inputs)
    except Exception as e:
        print("Failed to run the model: " + str(e))
        return False

    try:
        os.remove(_tmpfile)
    except Exception as e:
        print("We were unable to delete the file " + _tmpfile)

    return out


# graph_from_file opens an existing onnx file and returns the graph
def graph_from_file(
        filename: str):
    """ Retrieve a graph object from an onnx file

    Function attempts to open a .onnx file and returns its graph.

    Args:
        filename: String indicating the filename / relative location.

    Returns:
        An ONNX graph or False if unable to open.

    """
    mod_temp = xmp()
    try:
        with open(filename, 'rb') as fid:
            content = fid.read()
            mod_temp.ParseFromString(content)
        graph = mod_temp.graph
    except Exception as e:
        print("Unable to open the file: " + str(e))
        return False
    return graph


# graph_to_file saves a graph to a file
def graph_to_file(
        graph: xpb2.GraphProto,
        filename: str,
        _producer: str = "sclblonnx"):
    """ graph_to_file stores an onnx graph to a .onnx file

    Stores a graph to a file

    Args:
        graph: An onnx graph
        filename: The filename of the resulting file
        _producer: Optional string with producer name. Default 'sclblonnx'

    Returns:
        True if successful, False otherwise.
    """
    if not filename:
        print("Please specify a filename.")
        return False

    if type(graph) is not xpb2.GraphProto:
        print("graph is not an ONNX graph")

    try:
        mod = xhelp.make_model(graph, producer_name=_producer)
    except Exception as e:
        print("Unable to convert graph to model: " + str(e))
        return False

    try:
        xsave(mod, filename)
    except Exception as e:
        print("Unable to save the model: " + str(e))
        return False

    return True


# clean
def clean(
        graph: xpb2.GraphProto,
        _optimize: bool = True,
        _simplify: bool = True,
        _remove_initializer: bool = True,
        _producer: str = "sclblonnx"):
    """ clean cleans an ONNX graph using onnx tooling

    This method will attempt to clean the supplied graph by
    a. Removing initializers from input
    b. Optimizing it using onnxoptimizer.optimize
    c. Simplifying it using onnxsim.simplify

    If one of these fails the method will print an error message and return the unaltered graph.

    Args:
        graph: An ONNX graph
        _optimize: Boolean, default True. Optimize the model using onnxoptimizer.
        _simplify: Boolean, default True. Simplify the model using simplify.
        _remove_initializer: Boolean, default True. Remove initializers from input.
        _producer: Optional string with producer name. Default 'sclblonnx' (used for internal conversion)

    Returns:
        The cleaned ONNX graph, or the old graph if an error occurs.
    """
    try:
        mod = xhelp.make_model(graph, producer_name=_producer)
    except Exception as e:
        print("Unable to create the model: " + str(e))
        return graph

    if _optimize:
        try:
            mod = onnxoptimizer.optimize(mod, glob.OPTIMIZER_PASSES)
        except Exception as e:
            print("Unable to optimize your model: " + str(e))
            return graph

    if _simplify:
        try:
            mod, _ = simplify(mod)
        except Exception as e:
            print("Unable to simplify your model: " + str(e))
            return graph

    # Ugly from: onnxruntime/tools/python/remove_initializer_from_input.py
    graph = mod.graph
    if _remove_initializer:
        inputs = graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input
        for initializer in graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

    print("The graph was successfully cleaned.")
    return graph


# check examines an ONNX graph
def check(
        graph: xpb2.GraphProto,
        _producer: str = "sclblonnx"):
    """ check whether or not an existing graph can be converted using the Scailable platform

    We assume that a user will use graph_to_file() in this package to store the model. This

     Args:
        graph: an ONNX graph
        _producer: String optional

    Returns:
        True if the graph passes all the test. False otherwise.
    """
    # Check if this is a valid graph:
    if type(graph) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("g is not a valid ONNX graph.")
        return False

    # Standard ONNX checking:
    try:
        mod = xhelp.make_model(graph, producer_name=_producer)
    except Exception as e:
        print("Unable to create the model: " + str(e))
        return False
    try:
        checker.check_model(mod)
    except Exception as e:
        print("Model fails on standard ONNX checker: " + str(e))

    # input / output checking:
    if not graph.input:
        print("This graph does not contain any inputs.")
        return False

    if not graph.output:
        print("This graph does not contain any outputs.")
        return False

    # Sclbl checking:
    if not glob.ONNX_VERSION_INFO:
        if not _load_version_info():
            print("Unable to load the ONNX_VERSION INFO.")

    # Check general ONNX version:
    if version.parse(xversion) < version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_min']):
        print("Your current onnx version is lower then our support minimum. Please update your ONNX to {}".format(
            glob.ONNX_VERSION_INFO['onnx_version']['version_min']))
        return False

    if version.parse(xversion) > version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_max']):
        print(
            "Your current onnx version is higher then our support max. Please downgrade your ONNX version to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['version_max']))
        return False

    if mod.ir_version < glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']:
        print("Your current IR version is lower then our support minimum. Please update to {}".format(
            glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']))
        return False

    if mod.ir_version > glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']:
        print(
            "Your current IR version is higher then our support max. Please downgrade to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']))
        return False

    # Interate through opset and check:
    for key in mod.opset_import:
        v = key.version
        if v < glob.ONNX_VERSION_INFO['onnx_version']['opset_min']:
            print("One or more operators use an opset version that is too low. Please update to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['opset_min']))
            return False

        if v > glob.ONNX_VERSION_INFO['onnx_version']['opset_max']:
            print(
                "One or more operators use an opset version that is too high. Please downgrade to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['opset_max']))
            return False

    # Check individual nodes:
    not_supported = []
    for n in graph.node:
        op = n.op_type
        if op not in glob.ONNX_VERSION_INFO['operators']:
            not_supported.append(op)
            # todo(McK): Add additional checks

    if not_supported:
        print("The operator(s) {} are currently not supported.".format(not_supported))
        return False

    print("Your graph was successfully checked.")
    return True


# display uses Netron to display a graph
def display(
        graph: xpb2.GraphProto,
        _tmpfile: str = '.temp.onnx'):
    """ display a onnx graph using netron.

    Pass a graph to the display function to open it in Netron.
    Note: Due to the complexities of cross platform opening of source and the potential lack of
    a Netron installation this function might not always behave properly.
    Note2: This function might leave a file called .temp.onnx if it fails to remove the file.

    Args:
        graph: an ONNX graph
        _tmpfile: an optional string with the temporary file name. Default .tmp.onnx

    Returns:
        True if one of the 3 methods to open the file did not raise any warnings.

    Raises:
        SclblONNXError
    """
    if type(graph) is not xpb2.GraphProto:
        print("graph is not a valid ONNX graph.")
        return False

    # store as tmpfile
    graph_to_file(graph, _tmpfile)

    file_open = False
    # try open on unix:
    if not file_open:
        try:
            subprocess.run(['xdg-open', _tmpfile])
            file_open = True
        except Exception:
            file_open = False

    # try open on mac:
    if not file_open:
        try:
            subprocess.run(['open', _tmpfile])
            file_open = True
        except Exception:
            file_open = False

    # try open on windows:
    if not file_open:
        try:
            os.startfile(_tmpfile)
            file_open = True
        except Exception:
            file_open = False

    # Result:
    return file_open


# list_data_types prints all data available data types
def list_data_types():
    """ List all available data types. """
    print(json.dumps(glob.DATA_TYPES, indent=2))
    print("Note: STRINGS are not supported at this time.")


# _load_version_info loads info of supported ONNX versions (see _globals.py)
def _load_version_info() -> bool:
    """Loads the version info json.
    Function opens and parses the file supported_onnx.json in the current
    package folder to check current Scailable toolchain requirements.

    Note: the supported models are loaded into the glob.ONNX_VERSION_INFO
    dictionary to make them available to the whole package.

    Args:

    Returns:
        True if the supported version info is successfully loaded.
    """
    try:
        with open(glob.VERSION_INFO_LOCATION, "r") as f:
            glob.ONNX_VERSION_INFO = json.load(f)
    except FileNotFoundError:
        print("Unable to locate the ONNX_VERSION INFO.")
        return False
    return True


# _data_type converts a data type string to the actual data type (see _globals.py)
def _data_type(data_type: str):
    """ convert the data type to the appropriate number

    See: https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html
    """
    for key, val in glob.DATA_TYPES.items():
        if key == data_type:
            return val
    print("Data type not found. Use `list_data_types()` to list all supported data types.")
    return False


# No command line options for this script:
if __name__ == '__main__':
    print("No command line options available for main.py.")
