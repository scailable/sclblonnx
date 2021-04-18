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

- Fix merge.py / merge function
- Finalize "MergingGraphs.py"
- Fix sclbl_input()
- (re)Test all tutorials
- Add tests for all files in main package
  - add filename_test for each .py file
- Check remaining todo's
- Fix README
- Clean up examples / tutorials