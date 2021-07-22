import copy
from onnx import onnx_ml_pb2 as xpb2
from sclblonnx import add_output, add_input, add_node, node, empty_graph, add_constant, display, run, check, \
    clean, delete_output, delete_input
import numpy as np
from sclblonnx.utils import _print


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


## Scratchpad:


def _paste_graphs(sg1, sg2):
    """
    _paste_graphs simply takes two graphs and marges all initializers and nodes of both into a single
    graph.
    """
    g = copy.deepcopy(sg1)

    # Copy initilizers from sg2
    for init in sg2.initializer:
        g.initializer.append(init)

    # Copy nodes from sg2
    for node in sg2.node:
        g.node.append(node)

    # Copy inputs and outputs from sg2
    for item in sg2.input:
        g.input.append(item)
    for item in sg2.output:
        g.output.append(item)

    return g


def postfix_names(g, postfix="_g1", elem="node"):
    if elem == 'node':
        for item in g.node:
            item.name = item.name + postfix
        return g
    elif elem == 'init':
        for init in g.initializer:
            init.name = init.name + postfix
        return g
    elif elem == 'edge':
        for init in g.node:
            for index, name in enumerate(init.input):
                init.input[index] = init.input[index] + postfix
            for index, name in enumerate(init.output):
                init.output[index] = init.output[index] + postfix
        return g
    elif elem == 'input':
        for item in g.input:
            item.name = item.name + postfix
        return g
    elif elem == 'output':
        for item in g.output:
            item.name = item.name + postfix
        return g
    elif elem == 'io':
        cg = postfix_names(g, postfix, "input")
        cg = postfix_names(cg, postfix, "output")
        return cg
    elif elem == 'all':
        cg = postfix_names(g, postfix, "node")
        cg = postfix_names(cg, postfix, "init")
        cg = postfix_names(cg, postfix, "edge")
        cg = postfix_names(cg, postfix, "input")
        cg = postfix_names(cg, postfix, "output")
        return cg
    else:
        _print("No names have been changed.", "MSG")

    return g


def concat(
        sg1,
        sg2,
        complete=True,
        rename_nodes=True,
        io_match=[],
        rename_io=False,
        edge_match=[],
        rename_edges=False,
        **kwargs):
    """
    join connects sg1 and sg2 together. Topologically, sg2 is added to sg1 as a child.

    Args:
        sg1: A subgraph
        sg2:
        complete: Boolean indicating whether the resulting graph should be checked using so.check()
        rename_nodes: Boolean indicating whether the names of the nodes in the graph should be made unique
        io_match: Dict containing pairs of outputs of sg1 that should be matched to inputs of sg2. If this dict is empty the function will try to match by name.
        rename_io: Boolean indicating whether the inputs and outputs of the graph should be renamed (default False)
        edge_match: Dict containing pairs edge names of sg1 (i.e., node outputs) that should be matched to edges of sg2 (i.e., node inputs)
        rename_edges: Boolean indicating whether the edges should be renamed (default False)


    """

    # Check input types:
    if type(sg1) is not xpb2.GraphProto:
        _print("Graph sg1 is not an ONNX graph.")
        return False
    if type(sg2) is not xpb2.GraphProto:
        _print("Graph sg2 is not an ONNX graph.")
        return False

    # Rename node names if requested (default True)
    if rename_nodes:
        sg1 = postfix_names(sg1, "_sg1", "node")
        sg2 = postfix_names(sg2, "_sg2", "node")

    if io_match:
        for io_pair in io_match:
            print(io_pair)
            for outputs in sg1.output:
                print("Deleting output " + io_pair[0])
                if outputs.name == io_pair[0]:
                    sg1 = delete_output(sg1, io_pair[0])
            for inputs in sg2.input:
                print("Deleting input " + io_pair[1])
                if inputs.name == io_pair[1]:
                    sg2 = delete_input(sg2, io_pair[1])
            for item in sg2.node:
                print("Create edge" + io_pair[0] + "," + io_pair[1])
                for index, name in enumerate(item.input):
                    if name == io_pair[1]:
                        item.input[index] = io_pair[0]

    if rename_io:
        sg1 = postfix_names(sg1, "_sg1", "io")
        sg2 = postfix_names(sg2, "_sg2", "io")

    if edge_match:
        for edge_pair in edge_match:
            print("Matching edge")
            for item in sg2.node:
                for index, name in enumerate(item.input):
                    if name == edge_pair[1]:
                        item.input[index] = edge_pair[0]

    if rename_edges:
        sg1 = postfix_names(sg1, "_sg1", "edge")
        sg2 = postfix_names(sg2, "_sg2", "edge")

    # Paste graphs together:
    g = _paste_graphs(sg1, sg2)

    if complete:
        if not check(g, **kwargs):
            _print("BLA...")
            return False

    return g


# g1 = empty_graph("G1")
# n1 = node('Add', inputs=['x1_1', 'x1_2'], outputs=['sum_1'], name="node_name")
# g1 = add_input(g1, 'x1_1', "FLOAT", [1])
# g1 = add_input(g1, 'x1_2', "FLOAT", [1])
# g1 = add_output(g1, 'sum_1', "FLOAT", [1])
# g1 = add_node(g1, n1)

# g2 = empty_graph("G2")
# n2 = node('Add', inputs=['x2_1', 'x2_2'], outputs=['sum_2'], name="node_name")
# g2 = add_input(g2, 'x2_1', "FLOAT", [1])
# g2 = add_input(g2, 'x2_2', "FLOAT", [1])
# g2 = add_output(g2, 'sum_2', "FLOAT", [1])
# g2 = add_node(g2, n2)

# g = concat(g1, g2, False)
# display(g)

# g = concat(g1, g2, False, True, edge_match=[("x1_2", "x2_1")])
# display(g)


def merge(sg1: xpb2.GraphProto,
          sg2: xpb2.GraphProto,
          outputs: [],
          inputs: [],
          _verbose: bool = True):
    _print("Merge will be phased out; please use concat(), join(), and split() to combine graphs.", "MSG")

    if len(outputs) == 0:
        _print("Please specify the outputs of sg1 that need to be matched.")
        return False
    if len(inputs) == 0:
        _print("Please specify the inputs of sg2 that need to be matched.")
        return False
    if len(outputs) != len(inputs):
        _print("The number of outputs and inputs do not match.")
        return False

    io_pairs = []
    for idx, val in enumerate(outputs):
        io_pairs.append((val, inputs[idx]))

    print(io_pairs)
    g = concat(sg1, sg2, io_match=io_pairs, complete=True)
    return g


# merge is concat with io_match, False, etc.
# test_merge()

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


def join(pg1, pg2, cg, pg1_match=[], pg2_match=[], complete=False, **kwargs):
    """
    Join takes two parents and merges them with a child
    """
    io_match = pg1_match
    io_match.extend(pg2_match)
    g1 = concat(pg1, pg2, rename_nodes=False, complete=complete, **kwargs)
    g = concat(g1, cg, rename_nodes=False, io_match=io_match, complete=complete, **kwargs)
    return g


#g = join(g1, g2, g3, [("sum_1", "x3_1")], [("sum_2", "x3_2")])
# display(g)


def split(pg, cg1, cg2, cg1_match=[], cg2_match=[], complete=False, **kwargs):

    print("split...")
    g1 = concat(pg, cg1, rename_nodes=False, io_match=cg1_match, complete=complete, **kwargs)
    g = concat(g1, cg2, rename_nodes=False, io_match=cg2_match, complete=complete, **kwargs)
    return g



g = split(g1, g2, g3, cg1_match=[("sum_1", "x2_2")], cg2_match=[("sum_1", "x3_1")])
display(g)
exit()

# Little check for unkown input size:


g1 = empty_graph("G1")
n1 = node('Add', inputs=['x1_1', 'x1_2'], outputs=['sum_1'], name="node_name")
g1 = add_input(g1, 'x1_1', "FLOAT", [])
g1 = add_input(g1, 'x1_2', "FLOAT", [1])
g1 = add_output(g1, 'sum_1', "FLOAT", [1])
g1 = add_node(g1, n1)
check(g1)
display(g1)

# Chec stuff...
for inputs in g1.input:
    if not inputs.type.tensor_type.shape.dim:
        print("No dims...")
    for elem in inputs.type.tensor_type.shape.dim:
        print("Dim value: " + str(elem.dim_value))
        if elem.dim_value == 0 or elem.dim_value == "":
            print("Dynamic input size detected....")

