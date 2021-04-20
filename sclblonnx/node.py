from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2
import sclblonnx._globals as glob
from sclblonnx.utils import _print


# Node creates a new node
def node(
        op_type: str,
        inputs: [],
        outputs: [],
        name: str = "",
        **kwargs):
    """ Create a new node

    Args:
        op_type: Operator type, see https://github.com/onnx/onnx/blob/master/docs/Operators.md
        inputs: [] list of inputs (names to determine the graph topology)
        outputs: [] list of outputs (names to determine the graph topology)
        name: The name of this node (Optional)
        **kwargs
    """
    if not name:
        name = "sclbl-onnx-node" + str(glob.NODE_COUNT)
        glob.NODE_COUNT += 1

    try:
        node = xhelp.make_node(op_type, inputs, outputs, name, **kwargs)
    except Exception as e:
        _print("Unable to create node: " + str(e))
        return False
    return node


# add_node adds a node to a graph
def add_node(
        graph: xpb2.GraphProto,
        node: xpb2.NodeProto,
        **kwargs):
    """ Add node appends a node to graph g and returns the extended graph

    Prints a message and returns False if fails.

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        node: A node, onnx.onnx_ml_pb2.NodeProto.
        **kwargs

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("The graph is not a valid ONNX graph.")
        return False

    if type(node) is not xpb2.NodeProto:
        _print("The node is not a valid ONNX node.")
        return False

    try:
        graph.node.append(node, **kwargs)
    except Exception as e:
        _print("Unable to extend graph: " + str(e))
        return False
    return graph


def add_nodes(
        graph: xpb2.GraphProto,
        nodes: [xpb2.NodeProto],
        **kwargs):
    """ Add a list of nodes appends a node to graph g and returns the extended graph

    Prints a message and returns False if fails.

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        nodes: A list of nodes, [onnx.onnx_ml_pb2.NodeProto].
        **kwargs

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        print("graph is not a valid ONNX graph.")
        return False

    for node in nodes:  # error handling in add_node
        graph = add_node(graph, node, **kwargs)
        if not graph:
            return False

    return graph


# delete_node deletes a node from a graph
def delete_node(
        graph: xpb2.GraphProto,
        node_name: str = "",
        **kwargs):
    """ Add node appends a node to graph g and returns the extended graph

    Prints a message and returns False if fails.

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        node_name: Name of the node to remove.
        **kwargs

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        _print("The graph is not a valid ONNX graph.")
        return False

    if not node_name:
        _print("Please specify a node name.")
        return False

    found = False
    try:
        for elem in graph.node:
            if elem.name == node_name:
                graph.node.remove(elem)
                found = True
    except Exception as e:
        _print("Unable to iterate the nodes. " + str(e))
        return False
    if not found:
        _print("Unable to find the node by name.")
        return False

    return graph
