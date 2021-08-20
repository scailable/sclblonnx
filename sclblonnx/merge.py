import copy
from onnx import onnx_ml_pb2 as xpb2
from sclblonnx import check, delete_output, delete_input
from sclblonnx.utils import _print
"""
merge.py contains a number of utilities to merge / combine existing graphs. The functions merge(), join(), and split()
provide easy to use wrappers around the actual workhorse concat(). The concat function is versatile; please
see the example_merge.py script in /examples to 
"""


def merge(
        sg1: xpb2.GraphProto,
        sg2: xpb2.GraphProto,
        outputs: [] = None,
        inputs: [] = None,
        io_match: [] = None,
        complete: bool = True,
        _verbose: bool = True,
        **kwargs):
    """
    merge merges two graphs.

    Given subgraph sg1 and subgraph sg2 merge attempts to link the identified outputs of sg1 to the
    inputs of sg2 resulting in a graph in which sg1 is the parent of sg2.

    Merge expects two complete graphs (i.e., it expects sg1 and sg2 to pass check(). If you would like more
    flexible merge options or partial merge please see the concat function (merge is merely a constrained wrapper
    around concat).

    Note: The args inputs and outputs are present for legacy reasons, we recommend using io_match directly.

    Args:
        sg1: Subgraph 1, the parent.
        sg2: Subgraph 2, the child.
        outputs: (Optional) A list of strings containing the names of the outputs of sg1 that are matched to inputs (in order of the desired match).
        inputs: (Optional) A list of strings containing the names of the inputs of sg2 to which the outputs of sg1 are matched.
        io_match: (Optional) A list of names pairs [("out1","in1"), ("out2","in2"),...]. This is an alternative for the inputs/outputs arguments.
        complete: (Optional) Boolean indicating whether the resulting graph should be complete (i.e., should pass check). Default True.
        _verbose: (Optional) Boolean indicating whether or not verbose user feedback should be provided. Default True.
    Returns:
        The merged graph g, or False (with a printed error message) if something is wrong.
    """
    # immutable defaults:
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    if io_match is None:
        io_match = []

    # prevent changes to original
    sg1 = copy.deepcopy(sg1)
    sg2 = copy.deepcopy(sg2)

    # Check the inputs:
    if type(sg1) is not xpb2.GraphProto:
        _print("Graph sg1 is not an ONNX graph.")
        return False
    if type(sg2) is not xpb2.GraphProto:
        _print("Graph sg2 is not an ONNX graph.")
        return False
    if len(outputs) != len(inputs):
        _print("The number of outputs and inputs do not match.")
        return False
    if len(inputs) > 0 and len(io_match) > 0:
        _print("Please use either the inputs/outputs arguments OR the io_match argument (not both).")
        return False

    # Construct IO pairs
    if len(inputs) > 0:
        _print("Constructing the io_match list from your input and output.", "MSG", (not _verbose))
        io_match = []
        for idx, val in enumerate(outputs):
            io_match.append((val, inputs[idx]))

    # Use concat to do the merge
    g = concat(sg1, sg2, io_match=io_match, complete=complete, **kwargs)
    if not g:
        _print("Graph merge failed. Please checkout concat for additional options.", "MSG", (not _verbose))

    return g


def join(
        pg1: xpb2.GraphProto,
        pg2: xpb2.GraphProto,
        cg: xpb2.GraphProto,
        pg1_match: [] = None,
        pg2_match: [] = None,
        complete: bool = True,
        _verbose: bool = True,
        **kwargs):
    """
    join takes two parent graphs (pg1 & pg2) and merges them with a child graph cg.

    Join matches the outputs of pg1 to the inputs of cg specified in pg1_match, and similarly for pg2 and pg2_match.
    Desired matches are specified in pairs: [("out1","in1"), ("out2","in2"),...].

    Join by default assumes the resulting joined graph to be complete. Join is merely a wrapper around concat (used
    twice). For more flexible combinations of graphs please see concat().

    Note: ONNX concat operations might give unexpected results if names of elements collide, please use postfix_names()
    to prevent this (and always critically inspect the resulting graph).

    Args:
        pg1: Parent graph 1.
        pg2: Parent graph 2.
        cg: Child graph, the graph that will join together pg1 and pg2.
        pg1_match: (Optional) List of pairs matching outputs of pg1 to inputs of cg. Default [].
        pg2_match: (Optional) List of pairs matching outputs of pg2 to inputs of cg. Default [].
        complete: (Optional) Boolean indicating whether the resulting graph should be complete (i.e., should pass check). Default True.
        _verbose: (Optional) Boolean indicating whether or not verbose user feedback should be provided. Default True.
    Returns:
        The joined graph g (of False is something fails along the way).
    """
    # immutable defaults:
    if pg1_match is None:
        pg1_match = []
    if pg2_match is None:
        pg2_match = []

    # prevent changes to original
    pg1 = copy.deepcopy(pg1)
    pg2 = copy.deepcopy(pg2)
    cg = copy.deepcopy(cg)

    if type(pg1) is not xpb2.GraphProto:
        _print("Graph pg1 is not an ONNX graph.")
        return False
    if type(pg2) is not xpb2.GraphProto:
        _print("Graph pg2 is not an ONNX graph.")
        return False
    if type(cg) is not xpb2.GraphProto:
        _print("Graph cg is not an ONNX graph.")
        return False

    # Construct the match list
    io_match = pg1_match
    io_match.extend(pg2_match)

    # Do the joint (2x concat)
    g1 = concat(pg1, pg2, rename_nodes=True, complete=False, _verbose=False, **kwargs)
    g = concat(g1, cg, rename_nodes=True, io_match=io_match, complete=complete, _verbose=False, **kwargs)
    if not g:
        _print("Graph merge failed. Please checkout concat for additional options.", "MSG", (not _verbose))

    return g


def split(
        pg: xpb2.GraphProto,
        cg1: xpb2.GraphProto,
        cg2: xpb2.GraphProto,
        cg1_match: [] = None,
        cg2_match: [] = None,
        complete: bool = True,
        _verbose: bool = True,
        **kwargs):
    """
    split takes takes a single parent and matches the outputs to the inputs of two childs (cg1 & cg2)

    Split matches the outputs of pg to the inputs of cg1 and cg2 as specified in cg1_match and cg2_match.
    Desired matches are specified in pairs: [("out1","in1"), ("out2","in2"),...].

    Split by default assumes the resulting joined graph to be complete. Split is merely a wrapper around concat (used
    twice). For more flexible combinations of graphs please see concat().

    Note: ONNX concat operations might give unexpected results if names of elements collide, please use postfix_names()
    to prevent this (and always critically inspect the resulting graph).

    Args:
        pg: The parent graph
        cg1: The left child.
        cg2: The right child.
        cg1_match: (Optional) List of pairs matching outputs of pg to inputs of cg1. Default [].
        cg2_match: (Optional) List of pairs matching outputs of pg to inputs of cg2. Default [].
        complete: (Optional) Boolean indicating whether the resulting graph should be complete (i.e., should pass check). Default True.
        _verbose: (Optional) Boolean indicating whether or not verbose user feedback should be provided. Default True.
    Returns:
        The joined graph g (of False is something fails along the way).
    """
    # immutable defaults:
    if cg1_match is None:
        cg1_match = []
    if cg2_match is None:
        cg2_match = []

    # prevent changes to original
    pg = copy.deepcopy(pg)
    cg1 = copy.deepcopy(cg1)
    cg2 = copy.deepcopy(cg2)

    if type(pg) is not xpb2.GraphProto:
        _print("Graph pg is not an ONNX graph.")
        return False
    if type(cg1) is not xpb2.GraphProto:
        _print("Graph cg1 is not an ONNX graph.")
        return False
    if type(cg2) is not xpb2.GraphProto:
        _print("Graph cg2 is not an ONNX graph.")
        return False

    # Create the split (using concat 2x)
    g1 = concat(pg, cg1, rename_nodes=True, io_match=cg1_match, complete=False, _verbose=False, **kwargs)
    g = concat(g1, cg2, rename_nodes=True, io_match=cg2_match, complete=complete, _verbose=False, **kwargs)
    if not g:
        _print("Graph merge failed. Please checkout concat() for additional options.", "MSG", (not _verbose))

    return g


def concat(
        sg1: xpb2.GraphProto,
        sg2: xpb2.GraphProto,
        complete: bool = False,
        rename_nodes: bool = True,
        io_match: [] = None,
        rename_io: bool = False,
        edge_match: [] = None,
        rename_edges: bool = False,
        rename_init: bool = False,
        _verbose: bool = True,
        **kwargs):
    """
    concat concatenates two graphs.

    Concat is the flexible (but also rather complex) workhorse for the merge, join, and split functions and
    can be used to quite flexibly paste together two (sub)graphs. Contrary to merge, join, and split, concat
    does not by default assume the resulting onnx graph to be complete (i.e., to contain inputs and outputs and to
    pass check()), and it can thus be used as an intermediate function when constructing larger graphs.

    Concat is flexible and versatile, but it takes time to master. See example_merge.py in the examples folder
    for a number of examples.

    Args:
        sg1: Subgraph 1, the parent.
        sg2: Subgraph 2, the child.
        complete: (Optional) Boolean indicating whether the resulting graph should be checked using so.check(). Default False.
        rename_nodes: (Optional) Boolean indicating whether the names of the nodes in the graph should be made unique. Default True.
        io_match: (Optional) Dict containing pairs of outputs of sg1 that should be matched to inputs of sg2. Default [].
        rename_io: (Optional) Boolean indicating whether the inputs and outputs of the graph should be renamed. Default False.
        edge_match: (Optional) Dict containing pairs edge names of sg1 (i.e., node outputs) that should be matched to edges of sg2 (i.e., node inputs). Default [].
        rename_edges: (Optional) Boolean indicating whether the edges should be renamed (default False)
        _verbose: (Optional) Boolean indicating whether verbose output should be printed (default False)
    Returns:
        The concatenated graph g, or False if something goes wrong along the way.
    """
    # immutable defaults:
    if io_match is None:
        io_match = []
    if edge_match is None:
        edge_match = []

    # prevent changes to original
    sg1 = copy.deepcopy(sg1)
    sg2 = copy.deepcopy(sg2)

    # Check input types:
    if type(sg1) is not xpb2.GraphProto:
        _print("Graph sg1 is not an ONNX graph. Abort.")
        return False
    if type(sg2) is not xpb2.GraphProto:
        _print("Graph sg2 is not an ONNX graph. Abort.")
        return False

    # Rename node names if requested (default True)
    if rename_nodes:
        _print("Renaming node names in graph.", "MSG", (not _verbose))
        sg1 = postfix_names(sg1, "_sg1", "node")
        sg2 = postfix_names(sg2, "_sg2", "node")

    if io_match:
        _print("Matching specified inputs and outputs..", "MSG", (not _verbose))
        for io_pair in io_match:
            for outputs in sg1.output:
                if outputs.name == io_pair[0]:
                    sg1 = delete_output(sg1, io_pair[0])
            for inputs in sg2.input:
                if inputs.name == io_pair[1]:
                    sg2 = delete_input(sg2, io_pair[1])
            for item in sg2.node:
                for index, name in enumerate(item.input):
                    if name == io_pair[1]:
                        item.input[index] = io_pair[0]

    if rename_io:
        _print("Renaming inputs and outputs.", "MSG", (not _verbose))
        sg1 = postfix_names(sg1, "_sg1", "io")
        sg2 = postfix_names(sg2, "_sg2", "io")

    if edge_match:
        _print("Matching edges.", "MSG", (not _verbose))
        for edge_pair in edge_match:
            for item in sg2.node:
                for index, name in enumerate(item.input):
                    if name == edge_pair[1]:
                        item.input[index] = edge_pair[0]

    if rename_edges:
        _print("Renaming edges.", "MSG", (not _verbose))
        sg1 = postfix_names(sg1, "_sg1", "edge")
        sg2 = postfix_names(sg2, "_sg2", "edge")

    if rename_init:
        _print("Renaming init.", "MSG", (not _verbose))
        sg1 = postfix_names(sg1, "_sg1", "init")
        sg2 = postfix_names(sg2, "_sg2", "init")

    # Paste graphs together:
    _print("Pasting graphs.", "MSG", (not _verbose))
    g = _paste_graphs(sg1, sg2)

    if complete:
        if not check(g, _verbose=_verbose, **kwargs):
            _print("The end result does not pass check(). Are you sure you want a complete result? Set complete=False "
                   "to continue concat without checking.")
            return False

    return g


def postfix_names(
        g: xpb2.GraphProto,
        postfix: str = "_g1",
        elem: str = "node"):
    """
    postfix_names is a utility function used by concat() to rename parts of an onnx graph.

    When merging (or otherwise manipulating) onnx graphs it is often useful to create unique names of the
    various elements of the graph. This function postfixes each name in supplied graph g of elements of type elem
    by the supplied postfix.

    Args:
        g: The graph
        postfix: (Optional) The postfix for the names of the elements. Default "_g1".
        elem: (Optional) The type of element. Options are "node", "init", "edge", "input", "output", "io", and "all". Default "node".
    """
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
        _print("No names have been changed; did you select the right element?", "MSG")

    return g


def _paste_graphs(
        sg1: xpb2.GraphProto,
        sg2: xpb2.GraphProto):
    """
    _paste_graphs takes two subgraphs and pastes all of their objects into a single graph.

    Note, _paste_graphs does not conduct any checks, it just blindly copies. It is used internally
    by the concat() function.

    Args:
        sg1: The first subgraph
        sg2: The second subgraph

    Returns:
        g: the joined graph
    """
    g = copy.deepcopy(sg1)

    # Copy initializers from sg2
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
