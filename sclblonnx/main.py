import base64
import json
import os
import subprocess
import onnxruntime as xrt
from onnx import ModelProto as xmp
from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2
from onnx import save as xsave
from onnx import numpy_helper as xnp
import sclblonnx._globals as glob
from sclblonnx.utils import _print


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
        _print("Unable to create graph: " + str(e))
        return False
    return graph


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
        _print("Unable to open the file: " + str(e))
        return False
    return graph


# graph_to_file saves a graph to a file
def graph_to_file(
        graph: xpb2.GraphProto,
        filename: str,
        _producer: str = "sclblonnx",
        **kwargs):
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
        _print("Unable to save: Please specify a filename.")
        return False

    if type(graph) is not xpb2.GraphProto:
        _print("Unable to save: Graph is not an ONNX graph")

    try:
        mod = xhelp.make_model(graph, producer_name=_producer, **kwargs)
    except Exception as e:
        print("Unable to convert graph to model: " + str(e))
        return False

    try:
        xsave(mod, filename, **kwargs)
    except Exception as e:
        print("Unable to save the model: " + str(e))
        return False

    return True


# run executes a given graph and returns its result
def run(
        graph: xpb2.GraphProto,
        inputs: {},
        outputs: [],
        _tmpfile: str = ".tmp.onnx",
        **kwargs):
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
        _print("Unable to store model for evaluation.")
        return False

    try:
        sess = xrt.InferenceSession(_tmpfile, **kwargs)
        out = sess.run(outputs, inputs)
    except Exception as e:
        _print("Failed to run the model: " + str(e))
        return False

    try:
        os.remove(_tmpfile)
    except Exception as e:
        print("We were unable to delete the file " + _tmpfile, "MSG")

    return out


# display uses Netron to display a graph
def display(
        graph: xpb2.GraphProto,
        _tmpfile: str = '.tmp.onnx'):
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
        _print("graph is not a valid ONNX graph.")
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


# sclbl_input generates the example input for a Scailable runtime
def sclbl_input(
        inputs: {},
        example_type: str = "pb",
        _verbose: bool = True):
    """ input_str returns an example input for a Scailable runtime

    The method takes a valid input object to an onnx graph (i.e., one used for the "inputs" argument
    in the run() function, and returns and prints an example input to a Scailable runtime / REST endpoint

    Args:
        inputs: The input object as supplied to the run() function to test an ONNX grph
        example_type: The type of example string ("raw" for base64 encoded, or "pb" for protobuf, default pb)
        _verbose: Print user feedback; default True (note, errors are always printed).

    Returns:
        An example input to a Scailable runtime.
    """
    if not inputs:
        _print("No input provided.")

    if example_type == "raw":
        if len(inputs) == 1:
            for val in inputs.values():
                bytes = val.tobytes()
                encoded = base64.b64encode(bytes)
                value_str = encoded.decode('ascii')
        else:
            value_str = '["'
            for val in inputs.values():
                bytes = val.tobytes()
                encoded = base64.b64encode(bytes)
                value_str += (encoded.decode('ascii') + '","')
            value_str = value_str.rstrip(',"')
            value_str += '"]'

        input_json = '{"input": ' + value_str + ', "type":"raw"}'
        if _verbose:
            _print("The following input string can be used for the Scailable runtime:", "MSG")
            _print(input_json, "LIT")
        return input_json

    elif example_type == "pb" or "protobuf":

        if len(inputs) == 1:
            for val in inputs.values():
                tensor = xnp.from_array(val)
                serialized = tensor.SerializeToString()
                encoded = base64.b64encode(serialized)
                value_str = encoded.decode('ascii')
        else:
            value_str = '["'
            for val in inputs.values():
                tensor = xnp.from_array(val)
                serialized = tensor.SerializeToString()
                encoded = base64.b64encode(serialized)
                value_str += (encoded.decode('ascii') + '","')
            value_str = value_str.rstrip(',"')
            value_str += '"]'

        input_json = '{"input": ' + value_str + ', "type":"pb"}'
        if _verbose:
            _print("The following input string can be used for the Scailable runtime:", "MSG")
            _print(input_json, "LIT")
            _print("The following input string can be used for the web front-end:", "MSG")
            _print(value_str, "LIT")
        return input_json


# list_data_types prints all available data types
def list_data_types():
    """ List all available data types. """
    _print(json.dumps(glob.DATA_TYPES, indent=2), "MSG")
    _print("Note: STRINGS are not supported at this time.", "LIT")
    return True


# list_operators prints all operators available within Scailable
def list_operators():
    """ List all available Scailable ONNX operators. """
    try:
        with open(glob.VERSION_INFO_LOCATION, "r") as f:
            glob.ONNX_VERSION_INFO = json.load(f)
    except FileNotFoundError:
        print("Unable to locate the ONNX_VERSION INFO.")
        return False
    _print(json.dumps(glob.ONNX_VERSION_INFO['operators'], indent=2), "MSG")
    return  True


# No command line options for this script:
if __name__ == '__main__':
    print("No command line options available for main.py.")
