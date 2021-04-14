import sclblonnx as so
# https://github.com/scailable/sclbl-tutorials/tree/master/sclbl-pytorch-onnx

g = so.graph_from_file("onnx/cifar20.onnx")
g = so.clean(g)
so.check(g)
so.display(g)
# print(g2.input[0])