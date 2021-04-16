import copy

import sclblonnx as so
import numpy as np
from PIL import Image
"""
EXAMPLE 6: Merging two existing graphs.

This example combines two (sub) graphs into a single graph describing a longer pipeline.

The setup builds on example II (CreatingONNXFromScratchII.py). We create 2 seperate graphs:
1. Graph that resizes an image from 600x450 to 400x300
2. The empty container graph (check_container.onnx) which takes a 400x300 image

Next, we merge the two graphs.
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
out.save("images/1-Resized.JPG")

# Store the resize image onnx:
so.graph_to_file(sg1, "onnx/resize-image-450x600-300x400.onnx")

# So, now we have a working (sub)graph that resizes an image (which obviously we can just load next time)
# Now, we open up the original image processing graph
sg2 = so.graph_from_file("onnx/check_container.onnx")




# And next, we merge the two graphs
g = copy.deepcopy(sg1)


for init in sg2.initializer:
    g.initializer.append(init)

for node in sg2.node:
    g.node.append(node)

for node in g.node:
    for index, name in enumerate(node.input):
        if name == "small_image":
            node.input[index] = "in"
#

i = 1
names = []
for node in g.node:

    if node.name in names:
        node.name = node.name + str(i)
    names.append(node.name)

    for index, name in enumerate(node.input):
        if name == "in":
            node.input[index] = "small_image"
            #print("found...")
        print(name)
    # print(node)
    i += 1
#
# for init in g.initializer:
#     print(dir(init))
#     if init.name != "c1":
#         print(init)

so.list_outputs(sg1)  # small image, INT32, [300,400,3]
g = so.delete_output(g, "small_image")

#so.list_inputs(sg2)  # in, INT32,, [300,400,3]
#g = so.delete_input(g, "in")

#so.list_outputs(sg2)
g = so.add_output(g, "result", "BOOL", [1])  # add sg2 output(s)


g = so.clean(g)
so.check(g)
so.display(g)



