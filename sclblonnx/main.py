import json
import os
import subprocess

import onnxoptimizer
from onnx import ModelProto as xmp
from onnx import __version__ as xversion
from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2
from onnx import save as xsave
from packaging import version
from onnxsim import simplify
import sclblonnx._globals as glob
from sclblonnx.errors import SclblONNXError
# from onnx import TensorProto as xtp  # TP
# from onnx import xchecker
# import onnxruntime as xrt
# import onnxoptimizer


# graph_from_file
def graph_from_file(fname):
    """ Retrieve a graph object from an onnx file

    Function attempts to open a .onnx file and returns its graph.

    Args:
        fname: String indicating the filename / relative location.

    Returns:
        An ONNX graph or False if unable to open.

    """
    mod_temp = xmp()
    try:
        with open(fname, 'rb') as fid:
            content = fid.read()
            mod_temp.ParseFromString(content)
        g = mod_temp.graph
    except Exception as e:
        if not glob.SILENT:
            print("Unable to open the file: " + str(e))
        return False
    return g


# graph_to_file
def graph_to_file(g, fname, _producer="sclblonnx", _optimize=True, _simplify=True, _check=True):
    """ graph_to_file stores an onnx graph to a .onnx file

    Optimizes a graph and stores it to a file. Optimizing done using the graph_to_model() method.

    Args:
        g: An onnx graph
        fname: The filename of the resulting file
        _producer: Optional string with producer name. Default 'sclblonnx'
        _optimize: Boolean, default True. Optimize the model using onnxoptimizer
        _simplify: Boolean, default True. Simplify the model using simplify
        _check: Boolean, default True. Run the default onnx.checker check.

    Returns:
        True if successful.

    Raises:
        SclblONNXError if an error occurs.
    """
    if not fname:
        if not glob.SILENT:
            print("Please specify a filename.")
        return False
    if type(g) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("g is not an ONNX graph")
        return False

    try:
        mod = graph_to_model(g, _producer, _optimize, _simplify, _check)
    except Exception as e:
        if not glob.SILENT:
            print("Unable to convert graph to model: " + str(e))
        return False

    try:
        xsave(mod, fname)
    except Exception as e:
        if not glob.SILENT:
            print("Unable to save the model: " + str(e))
        return False

    return True


# optimize_graph
def optimize_graph(g, _producer="sclblonnx", _optimize=True, _simplify=True, _check=True):
    """ optimize_graph optimizes an ONNX graph using onnx tooling

        This method is a simple wrapper around graph_to_model but returns the resulting graph.

        Args:
            g: An ONNX graph
            _producer: Optional string with producer name. Default 'sclblonnx'
            _optimize: Boolean, default True. Optimize the model using onnxoptimizer
            _simplify: Boolean, default True. Simplify the model using simplify
            _check: Boolean, default True. Run the default onnx.checker check.

        Returns:
            A ONNX graph, False if an error occurs.

        Raises:
            SclblONNXError if an error occurs."""
    try:
        mod = graph_to_model(g, _producer, _optimize, _simplify, _check)
    except Exception as e:
        if not glob.SILENT:
            print("Unable to optimize graph: " + str(e))
        return False

    if not mod:
        return False

    return mod.graph


# graph_to_model
def graph_to_model(g, _producer="sclblonnx", _optimize=True, _simplify=True, _check=True):
    """ graph_to_model converts an ONNX graph to an ONNX model

    This method automatically executes the optimizer and simplify methods.

    Args:
        g: An ONNX graph
        _producer: Optional string with producer name. Default 'sclblonnx'
        _optimize: Boolean, default True. Optimize the model using onnxoptimizer
        _simplify: Boolean, default True. Simplify the model using simplify
        _check: Boolean, default True. Run the default onnx.checker check.

    Returns:
        A ONNX model onnx.onnx_ml_pb2.ModelProto, False if an error occurs.

    Raises:
        SclblONNXError if an error occurs."""
    try:
        mod = xhelp.make_model(g, producer_name=_producer)
    except Exception as e:
        if not glob.SILENT:
            print("Unable to create the model: " + str(e))
        return False

    if _optimize:
        try:
            mod = onnxoptimizer.optimize(mod, glob.OPTIMIZER_PASSES)
        except Exception as e:
            if not glob.SILENT:
                print("Unable to optimize your model: " + str(e))
            return False

    if _simplify:
        try:
            mod, check = simplify(mod)
        except Exception as e:
            if not glob.SILENT:
                print("Unable to simplify your model: " + str(e))
            return False

    if _check:
        if not check:
            if not glob.SILENT:
                print("The model does not pass the ONNX checker: " + str(e))
            return False

    return mod


# display uses Netron to display a graph
def display(g, _tmpfile='.temp.onnx'):
    """ display a onnx graph using netron.

    Pass a graph to the display function to open it in Netron.
    Note: Due to the complexities of cross platform opening of files and the potential lack of
    a Netron installation this function might not always behave properly.

    Args:
        g: an ONNX graph
        _tmpfile: an optional string with the temporary file name. Default .tmp.onnx

    Returns:
        True if one of the 3 methods to open the file did not raise any warnings.

    Raises:
        SclblONNXError
    """
    if type(g) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("g is not a valid ONNX graph.")
        return False

    # store as tmpfile
    graph_to_file(g, _tmpfile, _optimize=False, _simplify=False, _check=False)

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


# check examines an ONNX graph
def check(g, _optimize=True):
    """ check whether or not an existing graph can be converted using the Scailable platform

    We assume that a user will use graph_to_file() in this package to store the model.

    Note: This method will print user feedback unless the package is set to silent mode using stop_print()

     Args:
        g: an ONNX graph
        _optimize: Boolean, default True. Whether or not the graph should be optimized.

    Returns:
        True if one of the 3 methods to open the file did not raise any warnings.

    Raises:
        SclblONNXError

    """
    # Check if this is a valid graph:
    if type(g) is not xpb2.GraphProto:
        if not glob.SILENT:
            print("g is not a valid ONNX graph.")
        return False

    # optimize the graph
    if _optimize:
        g = optimize_graph(g)
        if not g:
            print("Optimization failed, halting check.")
            return False

    # Load the version info
    if not glob.ONNX_VERSION_INFO:
        if not load_version_info():
            if not glob.SILENT:
                print("Unable to load the ONNX_VERSION INFO.")
            return False

    # Check general ONNX version:
    if version.parse(xversion) < version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_min']):
        if not glob.SILENT:
            print("Your current onnx version is lower then our support minimum. Please update your ONNX to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['version_min']))
        return False
    if version.parse(xversion) > version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_max']):
        if not glob.SILENT:
            print(
                "Your current onnx version is higher then our support max. Please downgrade your ONNX version to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['version_max']))
        return False

    # Create model and check IR version
    try:
        mod = xhelp.make_model(g, producer_name='sclblonnx')
    except Exception as e:
        if not glob.SILENT:
            print("Unable to make model: " + str(e))
        return False

    if mod.ir_version < glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']:
        if not glob.SILENT:
            print("Your current IR version is lower then our support minimum. Please update to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']))
        return False
    if mod.ir_version > glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']:
        if not glob.SILENT:
            print(
                "Your current IR version is higher then our support max. Please downgrade to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']))
        return False

    # Interate through opset and check:
    for key in mod.opset_import:
        v = key.version
        if v < glob.ONNX_VERSION_INFO['onnx_version']['opset_min']:
            if not glob.SILENT:
                print("One or more operators use an opset version that is too low. Please update to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['opset_min']))
            return False
        if v > glob.ONNX_VERSION_INFO['onnx_version']['opset_max']:
            if not glob.SILENT:
                print(
                    "One or more operators use an opset version that is too high. Please downgrade to {}".format(
                        glob.ONNX_VERSION_INFO['onnx_version']['opset_max']))
            return False

    # Check individual nodes:
    not_supported = []
    for n in g.node:
        op = n.op_type
        if op not in glob.ONNX_VERSION_INFO['operators']:
            not_supported.append(op)

        #print(dir(n))
        #att = n.attribute
        #for at in att:
        #    t = getattr(at, "t", None)
        #    print(at)
        #    dt = getattr(t, "data_type", None)
        #    print(dt)
        # Todo(McK): Check data types w. Robin

    if not_supported:
        if not glob.SILENT:
            print("The operator(s) {} are currently not supported.".format(not_supported))
        return False

    if not glob.SILENT:
        print("Your graph was successfully checked.")
    return True


# constant_node creates a constant node
def constant_node(
        name,
        val,
        data_type
):
    """ Create a constant node

    Args:
        name: Name of the (output value of the) node
        val: Value of the node
        data_type: Data type of the node
    """
    return xhelp.make_node('Constant', inputs=[], outputs=[name], name=name + "-node",
           value=xhelp.make_tensor(name=name + "-value", data_type=data_type,
           dims=val.shape, vals=val.flatten()))


# run evaluates a graph with a given list of inputs

# operators lists all available Scailable operators

# add_graph adds (subgraph) g2 as a child to graph g1

# add_graph_before adds (subgraph) g1 as a parent to g2

# add_node adds a node to a graph

# if (a way to make easy ifs...)

# store


# stop_print stops verbose printing by the package
def stop_print(_verbose=False) -> bool:
    """Stop ALL printing of user feedback from package.
    Args:
        _verbose: Boolean indicator whether or not feedback should be printed. Default False.
    Returns:
        True
    """
    if _verbose:
        print("Printing user feedback set to 'False'.")
    glob.SILENT = True
    return True


# start_print starts verbose printing of the package
def start_print() -> bool:
    """(re)start printing user feedback
    Returns:
        True
    """
    print("Printing user feedback set to 'True'.")
    glob.SILENT = False
    return True


def load_version_info() -> bool:
    """Loads the version info json.
    Function opens and parses the file supported_onnx.json in the current
    package folder to check current Scailable toolchain requirements.

    Note: the supported models are loaded into the glob.ONNX_VERSION_INFO
    dictionary to make them available to the whole package.

    Args:
    Returns:
        True if the supported version info is successfully loaded.
    Raises (in debug mode):
        SclblONNXError.
    """
    try:
        with open(glob.VERSION_INFO_LOCATION, "r") as f:
            glob.ONNX_VERSION_INFO = json.load(f)
    except FileNotFoundError:
        if not glob.SILENT:
            print("Unable to find the ONNX_VERSION INFO.")
        raise SclblONNXError("Model check error: Unable to find list of supported models.")
    return True


# No command line options for this script:
if __name__ == '__main__':
    print("No command line options available for main.py.")
