import copy
from onnx import onnx_ml_pb2 as xpb2


# merge merges two graphs
from sclblonnx import check, clean, delete_output
from sclblonnx.utils import _print, _output_details, _input_details


def merge(
        sg1: xpb2.GraphProto,
        sg2: xpb2.GraphProto,
        outputs: [],
        inputs: [],
        _verbose: bool=True):
    """ merge merges two graphs.

    Given (sub)graphs sg1 and sg2, the merge function tries to paste the inputs of sg2 to the outputs
    of sg1.

    The function performs a number of checks to see if the initial graphs are valid, if the resulting graph
    is valid, and whether or not the inputs and outputs match.

    Note that the outputs orguments list the outputs of sg1 that will be matched to the inputs of sg2
    (the inputs argument) in order of appearance. Outputs must span all outputs of sg1, and inputs must
    span all inputs of sg2. The number of elements in sg1 and sg2 must be equal, and the types and
    dimensions of each must be equal.

    Args:
        sg1: Subgraph 1, the parent.
        sg2: Subgrpah 2, the child.
        outputs: A list containing all outputs of sg1 by name, in order of the desired match to the inputs of sg2
        inputs: A list containing all inputs of sg2 by name, in order of appearance for the desired match.
        _verbose: Print user feedback; default True (note, errors are always printed).
    """
    # First, a bunch of checks:
    if type(sg1) is not xpb2.GraphProto:
        _print("Graph sg1 is not an ONNX graph.")
        return False
    if type(sg2) is not xpb2.GraphProto:
        _print("Graph sg2 is not an ONNX graph.")
        return False
    if len(outputs) == 0:
        _print("Please specify the outputs of sg1 that need to be matched.")
        return False
    if len(inputs) == 0:
        _print("Please specify the inputs of sg2 that need to be matched.")
        return False
    if len(outputs) != len(inputs):
        _print("The number of outputs and inputs do not match.")
        return False
    if not check(sg1, _verbose=False):
        _print("Graph sg1 does not pass check(). Please fix.")
        return False
    if not check(sg2, _verbose=False):
        _print("Graph sg2 does not pass check(). Please fix.")
        return False

    # Check the inputs and outputs more elaborately:
    sg1_out = _output_details(sg1)
    if len(sg1_out) != len(outputs):
        _print("The number of specified outputs does not match the number of outputs of sg1")
        return False

    sg2_in = _input_details(sg2)
    if len(sg2_in) != len(inputs):
        _print("The number of specified inputs does not match the number of inputs of sg2")
        return False

    for index, output in enumerate(outputs):
        o = sg1_out.get(output, False)
        i = sg2_in.get(inputs[index], False)
        if not o or not i:
            _print("Your outputs or inputs lists contain names that are not foung in the graphs.")
            return False
        if o != i:
            _print("Output "+output+" does not match Input "+inputs[index]+" type or dims.")
            return False

    # All checks passed, let's give it a go:
    # 1: copy sg1 to new graph g
    g = copy.deepcopy(sg1)

    # 2: copy all initializers from sg2 to g
    for init in sg2.initializer:
        g.initializer.append(init)

    # 3: Add all nodes to sg2 to g
    for node in sg2.node:
        g.node.append(node)

    # 4. For each node in g...
    try:
        i = 1
        names = []
        for node in g.node:

            # make sure the resulting graph has no duplicate names:
            if node.name in names:
                node.name = node.name + str(i)
            names.append(node.name)

            # match input and output
            for index, name in enumerate(node.input):
                if name in inputs:
                    idx = inputs.index(name)
                    node.input[index] = outputs[idx]
            i += 1
    except Exception as e:
        _print("Unable to merge graphs: "+str(e))
        return False

    # Delete original outputs sg1:
    for output in outputs:
        g = delete_output(g, output)

    # Add outputs sg2 to the graph:
    for output in sg2.output:
        g.output.append(output)

    # check again (message if fails):
    if not check(g, _verbose=False):
        _print("Your merged graph does not pass check(). Please inspect the returned graph.", "MSG", (not _verbose))

    # clean (fail if fails):
    g = clean(g, _verbose=False)
    if not g:
        _print("Your merged graph does not pass clean().")

    _print("Graphs successfully merged.", "MSG", (not _verbose))
    return g
