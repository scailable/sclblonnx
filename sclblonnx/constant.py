import numpy as np
from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2

from sclblonnx.utils import _data_type
from sclblonnx.node import add_node


# constant creates a constant node.
def constant(name: str,
             value: np.array,
             data_type: str,
             **kwargs):
    """ Create a constant node

    Args:
        name: Name of the (output value of the) constant node to determine the graph topology
        value: Values of the node (as a np.array)
        data_type: Data type of the node

    Returns:
        The extended graph.
    """
    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        constant_node = xhelp.make_node('Constant', inputs=[], outputs=[name], name=name + "-constant",
                                        value=xhelp.make_tensor(name=name + "-values", data_type=dtype,
                                                                dims=value.shape, vals=value.flatten()), **kwargs)
    except Exception as e:
        print("Unable to create the constant node: " + str(e))
        return False

    return constant_node


# add_constant adds a constant node to a graph
def add_constant(
        graph: xpb2.GraphProto,
        name: str,
        value: np.array,
        data_type: str,
        **kwargs):
    """ Add a constant node to an existing graph

    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: Name of the (output value of the) constant node to determine the graph topology
        value: Values of the node (as a np.array)
        data_type: Data type of the node

    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        print("graph is not a valid ONNX graph.")
        return False

    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        constant_node = xhelp.make_node('Constant', inputs=[], outputs=[name], name=name + "-constant",
                                        value=xhelp.make_tensor(name=name + "-values", data_type=dtype,
                                                                dims=value.shape, vals=value.flatten()), **kwargs)
    except Exception as e:
        print("Unable to create the constant node: " + str(e))
        return False

    graph = add_node(graph, constant_node, **kwargs)
    if not graph:
        print("Unable to add constant node to graph.")
        return False
    return graph

