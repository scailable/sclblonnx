# sclblonnx

[![PyPI Release](https://github.com/scailable/sclblonnx/workflows/PyPI%20Release/badge.svg)](https://pypi.org/project/sclblonnx/)


The `sclblonnx` package provides a high level API to construct and alter ONNX graphs.

The basic usage is as follows:
```python

import sclblonnx as so

g = so.empty_graph()
n1 = so.node('Add', inputs=['x1', 'x2'], outputs=['sum'])
g = so.add_node(g, n1)
# etc.

```
Please see the `examples/` folder in this repo for examples.

