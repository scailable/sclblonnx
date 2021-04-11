import json
import os
import subprocess
from packaging import version
from onnx import __version__ as xversion
from onnx import ModelProto as xmp
from onnx import onnx_ml_pb2 as xpb2
from onnx import helper as xhelp
from onnx import save as xsave

# from onnx import TensorProto as xtp  # TP
# from onnx import xchecker
# import onnxruntime as xrt
from sclblonnx.errors import SclblONNXError
import sclblonnx._globals as glob


# graph_from_file
def graph_from_file(fname):
    """ Retrieve a graph object from an onnx file

    Function attempts to open a .onnx file and returns its graph.

    Args:
        fname: String indicating the filename / relative location.

    Returns:
        An ONNX graph.

    Raises:
        SclblONNXError if unable to open the file

    """
    mod_temp = xmp()
    try:
        with open(fname, 'rb') as fid:
            content = fid.read()
            mod_temp.ParseFromString(content)
        g = mod_temp.graph
    except Exception:
        raise SclblONNXError("A problem occurred opening your onnx file")
    return g


# graph_to_file
def graph_to_file(g, fname):
    """ graph_to_file stores an onnx graph to a .onnx file

    This is a simple wrapper around the onnx.helper function to create a model and the onnx.save function
    to store the model.

    Args:
        g: An onnx graph
        fname: The filename of the resulting file

    Returns:
        True if successful.

    Raises:
        SclblONNXError if an error occurs.
    """
    if not fname:
        raise SclblONNXError("Please specify a filename.")
    if type(g) is not xpb2.GraphProto:
        raise SclblONNXError("g does not seem to be a valid ONNX graph.")
    try:
        mod = xhelp.make_model(g, producer_name='sclblonnx')
        xsave(mod, fname)
    except Exception:
        raise SclblONNXError("An error occured trying to store your ONNX graph.")
        return False
    return True


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
        raise SclblONNXError("g does not seem to be a valid ONNX graph.")

    # store as tmpfile
    graph_to_file(g, _tmpfile)

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
def check(g):
    """ check whether or not an existing graph can be converted using the Scailable platform

    We assume that a user will use graph_to_file() in this package to store the model.

    Note: This method will print user feedback unless the pakcage is set to silent mode using stop_print()

     Args:
        g: an ONNX graph
        tmpfile: an optional string with the temporary file name. Default .tmp.onnx

    Returns:
        True if one of the 3 methods to open the file did not raise any warnings.

    Raises:
        SclblONNXError

    """
    # Check if this is a valid graph:
    if type(g) is not xpb2.GraphProto:
        raise SclblONNXError("g does not seem to be a valid ONNX graph.")

    # Load the version info
    if not glob.ONNX_VERSION_INFO:
        if not load_version_info():
            if not glob.SILENT:
                print("Unable to load the ONNX_VERSION INFO.")
            return False

    # Check general ONNX version:
    if version.parse(xversion) < version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_min']):
        if not glob.SILENT:
            print("Your current onnx version is lower then our support minimum. Please update your ONNX to {}".format(glob.ONNX_VERSION_INFO['onnx_version']['version_min']))
        return False
    if version.parse(xversion) > version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_max']):
        if not glob.SILENT:
            print("Your current onnx version is higher then our support max. Please downgrade your ONNX version to {}".format(glob.ONNX_VERSION_INFO['onnx_version']['version_max']))
        return False

    # Create model and check IR version
    mod = xhelp.make_model(g, producer_name='sclblonnx')
    if mod.ir_version < glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']:
        if not glob.SILENT:
            print("Your current IR version is lower then our support minimum. Please update to {}".format(glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']))
        return False
    if mod.ir_version > glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']:
        if not glob.SILENT:
            print(
                "Your current IR version is higher then our support max. Please downgrade to {}".format(glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']))
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
    for n in g.node:
        op = n.op_type
        if op not in glob.ONNX_VERSION_INFO['operators']:
            if not glob.SILENT:
                print("The operator {} is currently not supported.".format(op))
            return False

    # TODO(McK): Check data types...

    if not glob.SILENT:
        print("Your graph was successfully checked.")
    return True




# constant creates a constant node

# run evaluates a graph with a given list of inputs

# check examines the graph to see if it fits all Scailable requirements for conversion

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
