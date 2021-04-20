import onnxoptimizer
from onnx import __version__ as xversion
from onnx import checker
from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2
from onnxsim import simplify
from packaging import version

import sclblonnx._globals as glob
from sclblonnx.utils import _load_version_info, _print


# clean cleans a graph if possible (but also provides a stringent check)
def clean(
        graph: xpb2.GraphProto,
        _optimize: bool = True,
        _simplify: bool = True,
        _remove_initializer: bool = True,
        _producer: str = "sclblonnx",
        _verbose: bool = True,
        **kwargs):
    """ clean cleans an ONNX graph using onnx tooling

    This method will attempt to clean the supplied graph by
    a. Removing initializers from input
    b. Optimizing it using onnxoptimizer.optimize
    c. Simplifying it using onnxsim.simplify

    If one of these fails the method will print an error message and return the unaltered graph.

    Args:
        graph: An ONNX graph
        _optimize: Boolean, default True. Optimize the model using onnxoptimizer.
        _simplify: Boolean, default True. Simplify the model using simplify.
        _remove_initializer: Boolean, default True. Remove initializers from input.
        _producer: Optional string with producer name. Default 'sclblonnx' (used for internal conversion)
        _verbose: Print user feedback; default True (note, errors are always printed).
        **kwargs

    Returns:
        The cleaned ONNX graph, or the old graph if an error occurs.
    """
    try:
        mod = xhelp.make_model(graph, producer_name=_producer)
    except Exception as e:
        _print("Unable to create the model: " + str(e))
        return graph

    if _optimize:
        try:
            mod = onnxoptimizer.optimize(mod, glob.OPTIMIZER_PASSES, **kwargs)
        except Exception as e:
            _print("Unable to optimize your model: " + str(e))
            return graph

    if _simplify:
        try:
            mod, _ = simplify(mod, **kwargs)
        except Exception as e:
            _print("Unable to simplify your model: " + str(e))
            return graph

    # From: onnxruntime/tools/python/remove_initializer_from_input.py
    graph = mod.graph
    if _remove_initializer:
        inputs = graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input
        for initializer in graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

    _print("The graph was successfully cleaned.", "MSG", (not _verbose))
    return graph


# check checks the graph and inspects whether it is valid.
def check(
        graph: xpb2.GraphProto,
        _producer: str = "sclblonnx",
        _onnx_check: bool = True,
        _sclbl_check: bool = True,
        _verbose: bool = True,
        **kwargs):
    """ check whether or not an existing graph can be converted using the Scailable platform

    We assume that a user will use graph_to_file() in this package to store the model. This

     Args:
        graph: an ONNX graph
        _producer: String optional
        _onnx_check: Bool, default True. Run ONNX checker.check().
        _sclbl_check: Bool, default True.  Run Scailable checks.
        _verbose: Print user feedback; default True (note, errors are always printed).
        **kwargs

    Returns:
        True if the graph passes all the test. False otherwise.
    """
    # Check if this is a valid graph:
    if type(graph) is not xpb2.GraphProto:
        _print("Graph is not a valid ONNX graph.")
        return False

    # Convert to model:
    try:
        mod = xhelp.make_model(graph, producer_name=_producer, **kwargs)
    except Exception as e:
        _print("Unable to create the model: " + str(e))
        return False

    # Standard ONNX checking:
    if _onnx_check:
        try:
            checker.check_model(mod, **kwargs)
        except Exception as e:
            _print("Model fails on standard ONNX checker: " + str(e))
            return False

    if _sclbl_check:
        # input / output checking:
        if not graph.input:
            _print("This graph does not contain any inputs.")
            return False

        if not graph.output:
            _print("This graph does not contain any outputs.")
            return False

        # Sclbl checking:
        if not glob.ONNX_VERSION_INFO:
            if not _load_version_info():
                _print("Unable to load the ONNX_VERSION INFO.")

        # Check general ONNX version:
        if version.parse(xversion) < version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_min']):
            _print("Your current onnx version is lower then our support minimum. Please update your ONNX to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['version_min']))
            return False

        if version.parse(xversion) > version.parse(glob.ONNX_VERSION_INFO['onnx_version']['version_max']):
            _print(
                "Your current onnx version is higher then our support max. Please downgrade your ONNX version to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['version_max']))
            return False

        if mod.ir_version < glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']:
            _print("Your current IR version is lower then our support minimum. Please update to {}".format(
                glob.ONNX_VERSION_INFO['onnx_version']['ir_version_min']))
            return False

        if mod.ir_version > glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']:
            _print(
                "Your current IR version is higher then our support max. Please downgrade to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['ir_version_max']))
            return False

        # Interate through opset and check:
        for key in mod.opset_import:
            v = key.version
            if v < glob.ONNX_VERSION_INFO['onnx_version']['opset_min']:
                _print("One or more operators use an opset version that is too low. Please update to {}".format(
                    glob.ONNX_VERSION_INFO['onnx_version']['opset_min']))
                return False

            if v > glob.ONNX_VERSION_INFO['onnx_version']['opset_max']:
                _print(
                    "One or more operators use an opset version that is too high. Please downgrade to {}".format(
                        glob.ONNX_VERSION_INFO['onnx_version']['opset_max']))
                return False

        # Check individual nodes:
        not_supported = []
        for n in graph.node:
            op = n.op_type
            if op not in glob.ONNX_VERSION_INFO['operators']:
                not_supported.append(op)
                # todo: Add additional type checks...

        if not_supported:
            _print("The operator(s) {} are currently not supported.".format(not_supported))
            return False

    if not _sclbl_check and not _onnx_check:
        _print("Set _sclbl_check or _onnx_check to True to run any checks.")

    _print("Your graph was successfully checked.", "MSG", (not _verbose))
    return True
