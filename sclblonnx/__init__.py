"""
The sclblonnx package provides onnx tools
"""
# Ran on import of the package. Check version:
import sys

if sys.version_info < (3, 0):
    print('Sclblonnx requires Python 3, while Python ' + str(sys.version[0] + ' was detected. Terminating... '))
    sys.exit(1)

from .version import __version__

from .main import \
    empty_graph, \
    graph_from_file, \
    graph_to_file, \
    run, \
    display, \
    sclbl_input, \
    list_data_types, \
    list_operators

from .validate import \
    clean, \
    check

from .node import \
    node, \
    add_node, \
    add_nodes, \
    delete_node

from .constant import \
    constant, \
    add_constant

from .input import \
    list_inputs, \
    add_input, \
    rename_input, \
    replace_input, \
    delete_input

from .output import \
    list_outputs, \
    add_output, \
    rename_output, \
    replace_output, \
    delete_output

from .merge import \
    merge







