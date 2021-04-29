from onnx import onnx_ml_pb2 as xpb2
from sclblonnx import node, empty_graph, add_node, delete_node


def test_node():
    n1 = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
    assert type(n1) is xpb2.NodeProto, "Error creating node."
    n2 = node('Add', inputs=['x1', 'x2'], outputs=['sum'], name="node_name")
    name = getattr(n2, "name", False)
    assert name == "node_name", "Node name should be node_name."


def test_add_node():
    g = empty_graph()
    n = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
    g = add_node(g, n)
    assert len(g.node) == 1, "Node not properly added."


def test_add_nodes():
    g = empty_graph()
    for i in range(10):
        n = node('Add', inputs=['x1', 'x2'], outputs=['sum'])
        g = add_node(g, n)
    assert len(g.node) == 10, "Nodes not properly added."
    check = add_node(False, True)
    assert not check, "Incorrectly able to add to non-graph."
    check = add_node(g, False)
    assert not check, "Incorrectly able to add a non-node."


def test_delete_node():
    g = empty_graph()
    n = node('Add', inputs=['x1', 'x2'], outputs=['sum'], name="node_name")
    g = add_node(g, n)
    assert len(g.node) == 1, "Node not properly added."
    g = delete_node(g, "node_name")
    assert len(g.node) == 0, "Node not properly deleted."
