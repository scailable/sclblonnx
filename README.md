# sclblonnx
The `sclblonnx` package provide a high level API to construct and alter ONNX graphs.

The basic usage is as follows:
```python

import sclblonnx as so

g = so.empty_graph()
n1 = so.node('Add', inputs=['x1', 'x2'], outputs=['sum'])
g = so.add_node(g, n1)
# etc.

```
Please see the `examples/` folder in this repo for examples.

Todo:
- Work through all files, change print to _print and check.
- Add tests for all files in main package
  - add filename_test for each .py file
- Finalize example 06

