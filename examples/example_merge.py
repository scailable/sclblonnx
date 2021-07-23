import copy
import sclblonnx as so
import numpy as np
"""
EXAMPLE MERGE: a number of examples usages of the merge, join, split, and concat functions.

Note that merge(), join(), and split() are high level wrappers around concat(), each effectively assuming that the
resulting graph is "complete" (i.e., it is a valid onnx graph including input and output). Concat itself is more
flexible and can be used for intermediate merging/concatenation of partial graphs (i.e., graphs that are not yet
finished).

Below we provide a number of examples of each of the functions. We recommend using so.display() throughout to visualize
the resulting graphs and truly understand how the graphs are joined together. Examples are all very simple (small graphs,
scalar operations, etc.), but don't underestimate the complexities involved; with larger graphs the behavior of
the concat function can be challenging.
"""

# # Lets start by creating a few simple (and complete) graphs which we will use throughout:
# # Simple absolute value graph:
g1 = so.empty_graph("g_1")
n1 = so.node('Abs', inputs=['in_1_1'], outputs=['out_1_1'], name="node_1_1")
g1 = so.add_input(g1, 'in_1_1', "FLOAT", [1])
g1 = so.add_output(g1, 'out_1_1', "FLOAT", [1])
g1 = so.add_node(g1, n1)
# so.display(g1)
# data = {"in_1_1": np.array([2]).astype(np.float32)}
# print(so.run(g1, inputs=data, outputs=["out_1_1"]))

# # Simple max value graph:
g2= so.empty_graph("g_2")
n2 = so.node('Max', inputs=['in_2_1', 'in_2_2'], outputs=['out_2_1'], name="node_2_1")
g2 = so.add_input(g2, 'in_2_1', "FLOAT", [1])
g2 = so.add_constant(g2, "in_2_2", np.array([10]), "FLOAT")
g2 = so.add_output(g2, 'out_2_1', "FLOAT", [1])
g2 = so.add_node(g2, n2)
# so.display(g2)
# data = {"in_2_1": np.array([2]).astype(np.float32)}
# print(so.run(g2, inputs=data, outputs=["out_2_1"]))

# # Simple add two values graph:
g3 = so.empty_graph("g_3")
n3 = so.node('Add', inputs=['in_3_1', 'in_3_2'], outputs=['out_3_1'], name="node_3_1")
g3 = so.add_input(g3, 'in_3_1', "FLOAT", [1])
g3 = so.add_input(g3, 'in_3_2', "FLOAT", [1])
g3 = so.add_output(g3, 'out_3_1', "FLOAT", [1])
g3 = so.add_node(g3, n3)
# so.display(g3)
# data = {
#     "in_3_1": np.array([2]).astype(np.float32),
#     "in_3_2": np.array([5]).astype(np.float32)}
# print(so.run(g3, inputs=data, outputs=["out_3_1"]))


# # MERGE:
# # Merge takes two complete graphs and links the output of the parent to the inputs of the child.
# # Merge assumes the result is complete.
g_merge = so.merge(sg1=g1, sg2=g2, io_match=[("out_1_1", "in_2_1")])
# so.display(g_merge)
# data = {"in_1_1": np.array([2]).astype(np.float32)}
# print(so.run(g_merge, inputs=data, outputs=["out_2_1"]))


# # JOIN:
# # Join takes two parents and links their outputs to one child
# # Join assumes the result is complete.
g_join = so.join(pg1=g1, pg2=g2, cg=g3, pg1_match=[("out_1_1", "in_3_1")], pg2_match=[("out_2_1", "in_3_2")])
# so.display(g_join)
# data = {
#     "in_1_1": np.array([2]).astype(np.float32),
#     "in_2_1": np.array([2]).astype(np.float32)}
# print(so.run(g_join, inputs=data, outputs=["out_3_1"]))


# # SPLIT:
# # Split takes a single parent and links its output to the inputs of two children.
# # Split assumes the result is complete.
g_split = so.split(pg=g3, cg1=g1, cg2=g2, cg1_match=[("out_3_1", "in_1_1")], cg2_match=[("out_3_1", "in_2_1")])
# so.display(g_split)
# data = {
#     "in_3_1": np.array([2]).astype(np.float32),
#     "in_3_2": np.array([5]).astype(np.float32)}
# print(so.run(g_split, inputs=data, outputs=["out_1_1", "out_2_1"]))


# # CONCAT
# # Here we provide a number of uses of concat, please inspect the resulting graphs
# # Note, these result are by default not checked for completeness. Hence, the returned graph need not contain
# # valid inputs and outputs.
g_c1 = so.concat(g1, g2)  # Note, these are just the two graphs "side-by-side"
g_c2 = so.concat(g1, g2, io_match=[("out_1_1", "in_2_1")])  # Merge
g_c3 = so.concat(g1, g2, io_match=[("out_2_1", "in_1_1")])  # No merge
g_c4 = so.concat(g2, g1, io_match=[("out_2_1", "in_1_1")])  # Merge flipped, the order matters
g_c5 = so.concat(g1, g2, io_match=[("out_1_1", "in_2_1")], rename_nodes=False)  # Akin g_c2, but without the node names changed

g4 = copy.deepcopy(g1)  # an exact copy of g1
g_c6 = so.concat(g1, g4)  # Ugly...
g_c7 = so.concat(g1, g4, rename_edges=True, rename_io=True)  # Side by side

g5 = copy.deepcopy(g4)  # Another exact copy,
g5 = so.delete_input(g5, "in_1_1")  # Removing input and output
g5 = so.delete_output(g5, "out_1_1")
g_c8 = so.concat(g1, g5)  # Edge created, but unable to link a single output to two named edges

g6 = so.empty_graph("g_6")
n4 = so.node('Add', inputs=['in_1_1', 'in_6_2'], outputs=['out_6_1'], name="node_6_1")
g6 = so.add_node(g6, n4)
g_c9 = so.concat(g1, g6)  # Similarly named edges are also linked
g_c10 = so.concat(g1, g6, rename_edges=True)  # All edges renamed, but not i/o broken
g_c11 = so.concat(g1, g6, rename_edges=True, rename_io=True)  # g6 did not have inputs and outputs
g_c12 = so.concat(g1, g6, edge_match=[("out_1_1", "in_6_2")])  # Explicit edge matching (akin io_match but for internal edges)

# # Again, please use so.display(g..) to see the results of the above uses of concat.

