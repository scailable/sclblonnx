import sclblonnx as so
import numpy as np
from PIL import Image

"""
EXAMPLE 2: Rudimentary image analysis using ONNX.

This example is a reworked version of the tutorial presented at:
https://towardsdatascience.com/onnx-for-image-processing-from-scratch-6694f9b141b0

This example relies on the image "source/images/empty-average.JPG" which provides an average of several
pictures of an empty container.

The logic that we build is simple:
- Given an image of an object (for example 1.JPG in the source/images/ folder
- Subtract the empty-image (which is encoded as a constant in the ONNX graph)
- Compute absolute values
- Sum all elements of the result into a single scalar
- Compare the scalar to a threshold (another constant)
If the threshold is reached, we conclude that the container is filled.
"""

# Start with the empty graph:
g = so.empty_graph()

# Create the constant node encoding the empty image and add it to the graph:
# Note the type encoding as np.int64.
reference_image = np.array(Image.open("images/empty-average.JPG"), dtype=np.int32)
g = so.add_constant(g, "c1", reference_image, "INT32")

# Add the first input (note, same shape):
g = so.add_input(g, 'in', "INT32", reference_image.shape)

# Add the Subtract, Absolute, ReduceSum, and Less nodes
# Node how the names again enforce the topology of the graph
n1 = so.node("Sub", inputs=['in', 'c1'], outputs=['sub'])
n2 = so.node("Abs", inputs=['sub'], outputs=['abs'])
n3 = so.node("ReduceSum", inputs=['abs'], outputs=['sum'], keepdims=0) # Note the keepdims additional parameter.
g = so.add_nodes(g, [n1, n2, n3])

# And, we need to add the threshold (constant c2):
threshold = np.array([3000000]).astype(np.int32)
g = so.add_constant(g, "c2", threshold, "INT32")

# Add the less node. Please note that the nodes have to be added in topological order:
n4 = so.node("Less", inputs=['sum', 'c2'], outputs=['result'])
g = so.add_node(g, n4)

# Check: says that there is no output defined (which is true...)
so.check(g)

# Add output:
g = so.add_output(g, "result", "BOOL", [1])

# After which is passes all the checks
so.check(g)

# Let's inspect:
so.display(g)

# Let's clean:
g = so.clean(g)

# Let's try it out for the first image:
img_data = np.array(Image.open("images/2.JPG"), dtype=np.int32)
example = {"in": img_data.astype(np.int32)}
result = so.run(g,
                inputs=example,
                outputs=['result'])

# Print the result
if result[0]:
    print("The container is empty.")
else:
    print("The container is filled.")

# Example input for a Scailable runtime:
input_example = so.input_str(example)
print(input_example)

# Store the graph
so.graph_to_file(g, "onnx/check_container.onnx")
