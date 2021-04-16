"""
The sclblonnx package provides onnx tools
"""
# Ran on import of the package. Check version:
import sys

if sys.version_info < (3, 0):
    print('Sclblonnx requires Python 3, while Python ' + str(sys.version[0] + ' was detected. Terminating... '))
    sys.exit(1)

from .main import display, graph_from_file, empty_graph, node, add_input, add_node, add_output, \
    clean, run, graph_to_file, check, add_constant, add_nodes, list_data_types, constant, value, input_str, \
    list_inputs, replace_input, replace_output, list_outputs, rename_input, rename_output, delete_output, \
    delete_input
from .version import __version__


