from sclblonnx import add_output, add_input, add_node, node, empty_graph, add_constant, run, merge, split, display, \
    join, concat
import numpy as np
"""
Some rudimentary tests of the functions in merge.py; should be extended.

For example usage see example_merge.py in /examples.
"""


def test_merge():
    """
    Functional test of merge().
    """
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

    data = {"x1": np.array([2]).astype(np.float32), "x2": np.array([5]).astype(np.float32)}
    result = run(g, inputs=data, outputs=["equal"])
    assert result[0], "Sum of 2 and 5 should be equal to constant 7. Merged failed."


def test_join():
    """
    Functional test for join:
    """
    g1 = empty_graph("G1")
    n1 = node('Add', inputs=['x1_1', 'x1_2'], outputs=['sum_1'], name="n1")
    g1 = add_input(g1, 'x1_1', "FLOAT", [1])
    g1 = add_input(g1, 'x1_2', "FLOAT", [1])
    g1 = add_output(g1, 'sum_1', "FLOAT", [1])
    g1 = add_node(g1, n1)

    g2 = empty_graph("G2")
    n2 = node('Add', inputs=['x2_1', 'x2_2'], outputs=['sum_2'], name="n2")
    g2 = add_input(g2, 'x2_1', "FLOAT", [1])
    g2 = add_input(g2, 'x2_2', "FLOAT", [1])
    g2 = add_output(g2, 'sum_2', "FLOAT", [1])
    g2 = add_node(g2, n2)

    g3 = empty_graph("G3")
    n3 = node('Add', inputs=['x3_1', 'x3_2'], outputs=['sum_3'], name="n3")
    g3 = add_input(g3, 'x3_1', "FLOAT", [1])
    g3 = add_input(g3, 'x3_2', "FLOAT", [1])
    g3 = add_output(g3, 'sum_3', "FLOAT", [1])
    g3 = add_node(g3, n3)

    g = join(g1, g2, g3, [("sum_1", "x3_1")], [("sum_2", "x3_2")])

    data = {
        "x1_1": np.array([1]).astype(np.float32),
        "x1_2": np.array([2]).astype(np.float32),
        "x2_1": np.array([3]).astype(np.float32),
        "x2_2": np.array([4]).astype(np.float32),
    }
    result = run(g, inputs=data, outputs=["sum_3"])
    assert result[0], "Sum of 1,2, 3, and 4 should be equal to constant 10. Join failed."


def test_split():
    """
    Functional test for split
    """
    g1 = empty_graph("G1")
    n1 = node('Add', inputs=['x1_1', 'x1_2'], outputs=['sum_1'], name="n1")
    g1 = add_input(g1, 'x1_1', "FLOAT", [1])
    g1 = add_input(g1, 'x1_2', "FLOAT", [1])
    g1 = add_output(g1, 'sum_1', "FLOAT", [1])
    g1 = add_node(g1, n1)

    g2 = empty_graph("G2")
    n2 = node('Add', inputs=['x2_1', 'x2_2'], outputs=['sum_2'], name="n2")
    g2 = add_input(g2, 'x2_1', "FLOAT", [1])
    g2 = add_input(g2, 'x2_2', "FLOAT", [1])
    g2 = add_output(g2, 'sum_2', "FLOAT", [1])
    g2 = add_node(g2, n2)

    g3 = empty_graph("G3")
    n3 = node('Add', inputs=['x3_1', 'x3_2'], outputs=['sum_3'], name="n3")
    g3 = add_input(g3, 'x3_1', "FLOAT", [1])
    g3 = add_input(g3, 'x3_2', "FLOAT", [1])
    g3 = add_output(g3, 'sum_3', "FLOAT", [1])
    g3 = add_node(g3, n3)

    g = split(g1, g2, g3, cg1_match=[("sum_1", "x2_2")], cg2_match=[("sum_1", "x3_1")])

    data = {
        "x1_1": np.array([1]).astype(np.float32),
        "x1_2": np.array([2]).astype(np.float32),
        "x2_1": np.array([3]).astype(np.float32),
        "x3_2": np.array([4]).astype(np.float32),
    }
    result = run(g, inputs=data, outputs=["sum_2", "sum_3"])
    assert result[0], "Sum of 1,2, and 3 should be equal to constant 6. Split failed."


def test_concat():
    """
    Functional test for concat
    """
    g1 = empty_graph("G1")
    n1 = node('Add', inputs=['x1_1', 'x1_2'], outputs=['sum_1'], name="node_name")
    g1 = add_input(g1, 'x1_1', "FLOAT", [1])
    g1 = add_input(g1, 'x1_2', "FLOAT", [1])
    g1 = add_output(g1, 'sum_1', "FLOAT", [1])
    g1 = add_node(g1, n1)

    g2 = empty_graph("G2")
    n2 = node('Add', inputs=['x2_1', 'x2_2'], outputs=['sum_2'], name="node_name")
    g2 = add_input(g2, 'x2_2', "FLOAT", [1])
    g2 = add_output(g2, 'sum_2', "FLOAT", [1])
    g2 = add_node(g2, n2)

    g = concat(g1, g2, False, True, edge_match=[("x1_2", "x2_1")])

    data = {
        "x1_1": np.array([2]).astype(np.float32),
        "x1_2": np.array([5]).astype(np.float32),
        "x2_2": np.array([5]).astype(np.float32)}
    result = run(g, inputs=data, outputs=["sum_1", "sum_2"])
    assert result[0], "Sum of 2 and 5 should be equal to constant 7. Concat failed."


# Run tests, all passes:
test_merge()
test_join()
test_split()
test_concat()







