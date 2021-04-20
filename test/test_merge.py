from sclblonnx import add_output, add_input, add_node, node, empty_graph, add_constant, display, merge, run
import numpy as np


def test_merge():

    # Subgraph 1
    sg1 = empty_graph("Graph 1")
    n1 = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
    sg1 = add_node(sg1, n1)
    sg1 = add_input(sg1, 'x1', "FLOAT", [1])
    sg1 = add_input(sg1, 'x2', "FLOAT", [1])
    sg1 = add_output(sg1, 'sum', "FLOAT", [1])

    # Subgraph 2
    sg2 = empty_graph("Graph 2")
    sg2 = add_constant(sg2, "const", np.array([7]), "FLOAT")
    n2 = node("Equal", inputs=['sum', 'const'], outputs=['equal'])
    sg2 = add_node(sg2, n2)

    sg2 = add_input(sg2, 'sum', "FLOAT", [1])
    sg2 = add_output(sg2, 'equal', "BOOL", [1])

    g = merge(sg1, sg2, outputs=["sum"], inputs=["sum"])

    in1 = {"x1": np.array([2]).astype(np.float32), "x2": np.array([5]).astype(np.float32)}
    result = run(g, inputs=in1, outputs=["equal"])
    assert result[0], "Sum of 2 and 5 should be equal to constant 7."

    in2 = {"x1": np.array([4]).astype(np.float32), "x2": np.array([5]).astype(np.float32)}
    result = run(g, inputs=in2, outputs=["equal"])
    assert not result[0], "Sum of 4 and 5 should not be equal to constant 7."

    # todo(McK): Add tests for multiple inputs-outputs
    # todo(McK): Add tests for graphs containing If



