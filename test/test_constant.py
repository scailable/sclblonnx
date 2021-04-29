from sclblonnx import constant, empty_graph, node, add_node, add_constant, add_output, run, display, clean
import numpy as np


def test_constant():
    c = constant("", np.array([1,2,]), "FLOAT")
    assert not c, "Constant creation should have failed without a name."
    c = constant("constant", np.array([1,2,]), "NONE")
    assert not c, "Constant creation should have failed without a valid data type."
    c = constant("constant", np.array([1,2,]), "FLOAT")
    check = getattr(c, "output", False)
    assert check[0] == "constant", "Constant creation should have worked."


def test_add_constant():

    # Simple add graph
    g = empty_graph()
    n1 = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
    g = add_node(g, n1)

    # Add input and constant
    g = add_constant(g, 'x1', np.array([1]), "INT64")
    g = add_constant(g, 'x2', np.array([5]), "INT64")

    # Output:
    g = add_output(g, 'sum', "INT64", [1])

    # This works, but seems to fail for other data types...
    result = run(g, inputs={}, outputs=["sum"])
    assert result[0] == 6, "Add constant failed."
    # todo(McK): Does not work for INT16 / INT8, check?