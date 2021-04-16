from PIL import Image
import numpy as np
import sclblonnx as so
"""
EXAMPLE 5:

Editing an existing graph: while the previous examples already shows some graph editing, here we more elaborate
editing of the graph; we demonstrate changing inputs, and we demonstrate the use of the If operator for
post-processing.

We will build on the PyTorch example (see ExportingFromPyTorch.py) to, instead of outputting the scores from the
cifar model, output 99 if there is a horse on the image and 0 otherwise.
"""


# First, we load the the example image:
# To open an image we write a small utility function using Pillow to transform an image to a numpy array.
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img / 255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.4914) / 0.2023
    img[1] = (img[1] - 0.4822) / 0.1994
    img[2] = (img[2] - 0.4465) / 0.2010

    # Add a fourth dimension to the beginning to indicate batch size
    # img = img[np.newaxis,:].astype(np.float16)
    img = img[np.newaxis, :]

    return img


# Open the image
img_input = process_image("images/horse5.png").astype(np.float32)

# Load the cifar model:
g = so.graph_from_file("onnx/cifar10-resnet20-clean.onnx")

# Check its output, we see the name, type, and dimensions
so.list_outputs(g)

# Run the model to see the outputs:
result = so.run(g, inputs={"input": img_input}, outputs=["output"])
print(result)

# Add and arg_max node to find the highest output in the output vector
# Note the keepdims and axis; the output of the Argmax node should align with the defined output.
n1 = so.node('ArgMax', inputs=['output'], outputs=['argmax'], keepdims=0, axis=1)
g = so.add_node(g, n1)  # Note, this adds the node, but the output is still "output"
g = so.delete_output(g, "output")  # Remove the old output
g = so.add_output(g, 'argmax', "INT64", [1])  # Add the new output (for testing only)

# Test:
result = so.run(g, inputs={"input": img_input}, outputs=["argmax"])
print(result)

# So, this works. Let's remove the output argmax again before we continue:
g = so.delete_output(g, 'argmax')

# Because the if statement to switch between values of 100 and 0 requires a boolean input condition, we add
# a constant node with the value of 7, and add an equals node:
g = so.add_constant(g, "cut", np.array([7]), "INT64")
n2 = so.node("Equal", inputs=['argmax', 'cut'], outputs=['seven'])
g = so.add_node(g, n2)

# Lets again test:
g = so.add_output(g, 'seven', "BOOL", [1])
result = so.run(g, inputs={"input": img_input}, outputs=["seven"])
print(result)  # Prints true... we are getting closer!
g = so.delete_output(g, 'seven')


# Here we build an if statement. Note that the if "switches" between two graphs, so let's first create the
# two graphs (which can obviously be much more complex). We start with the if:
then_graph = so.empty_graph("then-graph")
then_graph = so.add_constant(then_graph, "then_value", np.array([100]), "FLOAT")
then_graph = so.add_output(then_graph, "then_value", "FLOAT", [1])
so.display(then_graph)  # See, this is a very small graph, no input, only output

# Same for else
else_graph = so.empty_graph("else-graph")
else_graph = so.add_constant(else_graph, "iff_value", np.array([0]), "FLOAT")
else_graph = so.add_output(else_graph, "iff_value", "FLOAT", [1])


# Now, the If node which switches between the if and the else graph
n3 = so.node("If", inputs=['seven'], outputs=['horse'], then_branch=then_graph, else_branch=else_graph)
g = so.add_node(g, n3)

# Add the output
g = so.add_output(g, "horse", "FLOAT", [1])
result = so.run(g, inputs={"input": img_input}, outputs=["horse"])
print(result) # Prints 100!

# Store the augmented graph
g = so.clean(g)
so.check(g)
so.graph_to_file(g, "onnx/cifar10-resnet20-augmented.onnx")

