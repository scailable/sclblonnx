import sclblonnx as so
import numpy as np
from PIL import Image

"""
EXAMPLE 4: Using previously exported pyTorch model

Here we open an exisiting and pre-trained Resnet model (trained on the cifar data).

For training details see:
https://github.com/scailable/sclbl-tutorials/tree/master/sclbl-pytorch-onnx

Here we simply evaluate one specific image.
"""

# Retrieve the graph from the stored .onnx model:
g = so.graph_from_file("onnx/cifar10-resnet20.onnx")

# Clean, check, and display (this model passes all the checks).
g = so.clean(g)
so.check(g)
so.display(g)

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


# Open the image and execute the graph:
input_example = process_image("images/horse5.png").astype(np.float32)
out = so.run(g,
             inputs={"input": input_example},
             outputs=['output']
             )

# Pretty printing
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("The ONNX model predicts the image is a", classes[np.argmax(out[0])] + ".")

# Or inspect the input for the Scailable runtime
print(so.input_str(input_example))

# And store the file (since we did clean it):
so.graph_to_file(g, "onnx/cifar10-resnet20-clean.onnx")