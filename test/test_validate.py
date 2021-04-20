from sclblonnx import empty_graph, node, add_node, add_input, add_output, check, clean
from onnx import onnx_ml_pb2 as xpb2


def test_clean():
    g = empty_graph()
    n1 = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
    g = add_node(g, n1)
    g = add_input(g, 'x1', "FLOAT", [1])
    g = add_input(g, 'x2', "FLOAT", [1])
    g = add_output(g, 'sum', "FLOAT", [1])
    g = clean(g)
    assert type(g) == xpb2.GraphProto, "Clean failed."


def test_check():
    g = empty_graph()
    n1 = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
    g = add_node(g, n1)
    assert not check(g), "Graph is not complete."

    g = add_input(g, 'x1', "FLOAT", [1])
    g = add_input(g, 'x2', "FLOAT", [1])
    g = add_output(g, 'sum', "FLOAT", [1])
    assert check(g), "Graph should pass checks."

    g = empty_graph()
    n1 = node('None', inputs=['x1', 'x2'], outputs=['sum'])
    g = add_node(g, n1)
    g = add_input(g, 'x1', "FLOAT", [1])
    g = add_input(g, 'x2', "FLOAT", [1])
    g = add_output(g, 'sum', "FLOAT", [1])
    assert not check(g), "Graph should not pass checks."

    check(g, _sclbl_check=False, _onnx_check=False)
    check(g, _onnx_check=False)  # Operator check.