import os
import onnx
import pytest

from sclblonnx.errors import SclblONNXError
from sclblonnx.main import display, graph_from_file, graph_to_file, check, clean


# Test graph from file:
def test_graph_from_file():
    # Test non-existing file
    g = graph_from_file('not_existing.onnx')
    assert not g, "This should be false"
    # Test existing file
    g = graph_from_file("files/example01.onnx")
    assert type(g) is onnx.onnx_ml_pb2.GraphProto


# Test graph storage:
def test_graph_to_file():
    # Test not a proper graph
    with pytest.raises(SclblONNXError) as excinfo:
        graph_to_file({}, 'name.onnx')
    assert "valid ONNX graph" in str(excinfo.value)
    # Test existing file
    g = graph_from_file("files/example01.onnx")
    result = graph_to_file(g, '_tmp.onnx')
    assert result is True
    # Remove the file
    os.remove("_tmp.onnx")


# Test alle examples in test/files
def test_check():
    dir = "files"
    for fname in os.listdir(dir):
        if fname.endswith(".onnx"):
            print(fname)
            fpath = dir + "/" + fname
            g = graph_from_file(fpath)
            check(g, False)


# Functional test off all bits of the package:
def test_functional():
    # Open an existing graph:
    g = graph_from_file("files/example07.onnx")

    # Display the graph:
    display(g)

    # Clean up:
    g = clean(g, _optimize=True, _simplify=True, _check=False)
    display(g)


    # Check the graph:
    #check(g)
    #print(g)

#test_graph_from_file()
#test_graph_to_file()
#test_functional()
#test_check()

g = graph_from_file("files/example01.onnx")
display(g)
