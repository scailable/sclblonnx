# SclblONNX examples.

The following examples are provided:

* [Example 01](example_01.py) - **Add**: 
  Create an ONNX graph to add two numbers from scratch and evaluate it using Scailable tools.
  This tutorial covers the basic usage of the `sclblonnx` package.
* [Example_02](example_02.py) - **Image**:
  Create an ONNX graph that checks whether or not an image is empty. 
* [Example_03](example_03.py) - **PyTorch**:
  This example imports a resnet model trained using PyTorch and show how to use it with image input.
* [Example_04](example_04.py) - **TensorFlow**:
  This example imports a model trained using TensorFlow (including the model training code.)
  The example shows how to rename the inputs and outputs of an existing model and fix dynamic inputs.
* [Example_05](example_05.py) - **Post-processing**:
  This more elaborate example builds on example 2, but extends the onnx file generated in this example
  to include post-processing. It demonstrates how to extend an existing graph and it demonstrates the use
  of If statements.
* [Example_06](example_06.py) - **Graph Merge**:
  This example shows how to merge two existing ONNX graphs using `so.merge()`.