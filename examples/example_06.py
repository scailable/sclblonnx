import copy

import sclblonnx as so
import numpy as np
from PIL import Image
"""
EXAMPLE 6: Merging two existing graphs.

This example combines two (sub) graphs into a single graph describing a longer pipeline.

The setup builds on example II (example_02.py). We create 2 separate graphs:
1. Graph that resizes an image from 600x450 to 400x300
2. The empty container graph (check_container.onnx) which takes a 400x300 image

Next, we merge the two graphs into one single ONNX file.
"""

# Let's open the large image and inspect the shape:
large_img = np.array(Image.open("images/1-BIG.JPG"), dtype=np.int32)
print(large_img.shape)  # 450x600x3

# First subgraph for resize:
sg1 = so.empty_graph("resize_graph")
sg1 = so.add_input(sg1, "large_image", "INT32", [450, 600, 3])  # Add the input

# The resize node:
c1 = so.constant("size", np.array([300, 400, 3]), "INT64")
n1 = so.node("Resize", inputs=['large_image', '', '', 'size'], outputs=['small_image'])
sg1 = so.add_nodes(sg1, [c1, n1])
sg1 = so.add_output(sg1, "small_image", "INT32", [300, 400, 3])

# Check and clean
sg1 = so.clean(sg1)
so.check(sg1)

# Test the resize graph:
large_input = {"large_image": large_img.astype(np.int32)}
result = so.run(sg1, inputs=large_input, outputs=['small_image'])

# Round values in array and cast as 8-bit integer to store back as JPG:
img_arr = np.array(np.round(result[0]), dtype=np.uint8)
out = Image.fromarray(img_arr, mode="RGB")
out.save("images/1-Resized.JPG")  # Yes, this works.

# Store the resize onnx:
so.graph_to_file(sg1, "onnx/resize-image-450x600-300x400.onnx")

# So, now we have a working (sub)graph that resizes an image (which obviously we can just load next time)
# Now, we open up the original image processing graph
sg2 = so.graph_from_file("onnx/check-container.onnx")

# The outputs of sg1 and the inputs of sg2 need to match; lets examine them
so.list_outputs(sg1)
so.list_inputs(sg2)

# Merge the two graphs, the outputs will be merged with the inputs in order of appearance:
g = so.merge(sg1, sg2, outputs=["small_image"], inputs=["in"])
so.check(g)
so.display(g)

# And now it works with the large image:
result = so.run(g, inputs=large_input, outputs=['result'])
# Print the result
if result[0]:
    print("The container in the large image is empty.")
else:
    print("The container in the large image is filled.")

# Store the merged graph
# Todo(McK): Check this merged graph; it does not compile...
g = so.graph_to_file(g, "onnx/check-container-resize.onnx")

