import json
import os
import subprocess

import onnxruntime as xrt
from onnx import ModelProto as xmp
from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2
from onnx import save as xsave

import sclblonnx._globals as glob


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


# run executes a given graph and returns its result
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


# sclbl_input generates the example input for a Scailable runtime
def sclbl_input(inputs: {}):
    """ input_str returns an example input for a Scailable runtime

    The method takes a valid input object to an onnx graph (i.e., one used for the "inputs" argument
    in the run() function, and returns and prints an example input to a Scailable runtime / REST endpoint

    Args:
        inputs The input object

    Returns:
        An example input, base64 formatted.
    """
    # todo(Robin): Fix input_str() function.
    print("WARNING: input_str is not yet implemented.")
    return "Not yet..."


# list_data_types prints all data available data types
def list_data_types():
    """ List all available data types. """
    print(json.dumps(glob.DATA_TYPES, indent=2))
    print("Note: STRINGS are not supported at this time.")


# No command line options for this script:
if __name__ == '__main__':
    print("No command line options available for main.py.")
