import json
import requests
import numpy as np
import sclblpy as sp  # Import the sclblpy package
import sclblonnx as so

"""
EXAMPLE 1: Adding two scalars.

This example shows the basic usage of the sclblonnx package by creating an onnx graph from scratch that adds two
scalars together.
"""

# Use the empty_graph() method to create a named xpb2.GraphProto object:
g = so.empty_graph()

# Add a node to the graph.
# Please note the list of available operators at: https://github.com/onnx/onnx/blob/master/docs/Operators.md
# Run so.list_data_types() and / or so.list_operators() so. to see all Scailable supported data types.
n1 = so.node('Add', inputs=['x1', 'x2'], outputs=['sum'])
g = so.add_node(g, n1)

# We should explicitly specify the named inputs to the graph -- note that the names determine the graph topology.
# Also, we should specify the data type and dimensions of any input.
# Use so.list_data_types() to see available data types.
g = so.add_input(g, 'x1', "FLOAT", [1])
g = so.add_input(g, 'x2', "FLOAT", [1])

# Similarly, we add the named output with its corresponding type and dimension.
# Note that types will need to "match", as do dimensions. Please see the operator docs for more info.
g = so.add_output(g, 'sum', "FLOAT", [1])

# so.check() checks the current graph to see if it matches Scailable's upload criteria for .wasm conversion.
so.check(g)

# Now, a few tricks to sanitize the graph which are always useful.
# so.clean() provides lossless reduction of the graph. If successful cleaned graph is returned.
g = so.clean(g)

# so.display() tries to open the graph using Netron to inspect it. This worsk on most systems if Netron is installed.
# Get Netron at https://github.com/onnx/onnx/blob/master/docs/Operators.md
so.display(g)

# Now, use the default ONNX runtime to do a test run of the graph.
# Note that the inputs dimensions and types need to match the specification of the graph.
# The outputs returns all the outputs named in the list.
example = {"x1": np.array([1.2]).astype(np.float32), "x2": np.array([2.5]).astype(np.float32)}
result = so.run(g,
                inputs=example,
                outputs=["sum"]
                )
print(result)

# We can easily store the graph to a file for upload at http://admin.sclbl.net:
so.graph_to_file(g, "onnx/add-scalars.onnx")

# Or, we can upload it to Scailable using the sclblpy package,
# See the sclblpy package docs for more details. https://pypi.org/project/sclblpy/
# sp.upload_onnx("onnx/add-scalars.onnx", docs={"name": "Example_01: Add", "documentation": "Empty.."})


# so.sclbl_input(inputs) converts an example input to the input that can be used on the device:
example_input = so.sclbl_input(example, "pb")
print(example_input)


# You can use the example to setup your own REST call:
def do_REST_call(cfid: str, data: str):
    """ Do rest call calls a REST endpoint on the Scailable cloud"""
    # This does work
    url = "https://taskmanager.sclbl.net:8080/task/" + cfid
    payload = "{\"input\":{\"content-type\":\"json\",\"location\":\"embedded\",\"data\":" \
              + json.dumps(data) + \
              "},\"output\":{\"content-type\":\"json\",\"location\":\"echo\"}," \
              "\"control\":1,\"properties\":{\"language\":\"WASM\"}}"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text.encode('utf8')


# Do the actual call using requests:
print(do_REST_call("403cd8a0-a10f-11eb-9acc-9600004e79cc", example_input))

# Or again use the sclblpy package, this time use the .run() function to execute the graph on the Scailable cloud:
print(sp.run("403cd8a0-a10f-11eb-9acc-9600004e79cc", example_input))
