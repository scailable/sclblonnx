# # Open the stored onnx:
# from onnx import ModelProto
# import numpy as np
#
# mod_temp = ModelProto()
# with open("dnn.onnx", 'rb') as fid:
#     content = fid.read()
#     mod_temp.ParseFromString(content)
#
# # retrieve the graph from opened model
# g2 = mod_temp.graph
#
# # retrieve output name
# outputs =[node.name for node in g2.output]
# output_name = outputs[0]
#
# # Param:
# hc = 10  # I.e., the height at which the cow needs feeding; we compute the expected time for this event.
#
# # Remove the original input (=feature) and output (=variable):
# g2.input.remove(g2.input[0])
# g2.output.remove(g2.output[0])
#
# # Input feature as constant
# feature = constant_node("input_1", np.array([hc, 0]), tp.FLOAT)
# # Rename feature name, otherwise named input_1-node
# feature.name = "input_1"
#
# # Get t30
# starts = constant_node("starts", np.array([1, n_obs_per_cow - 1]), tp.INT64)  # Slice from
# ends = constant_node("ends", np.array([2, n_obs_per_cow]), tp.INT64)  # Slice to
# n1 = h.make_node('Slice', inputs=['in', 'starts', 'ends'], outputs=['t30'])  # Select the last timepoint
#
# # Compute dt
# n2 = h.make_node('Sub', inputs=[output_name, 't30'], outputs=['dt'])  # Difference with predicted empty time
#
# # Append the nodes
# g2.node.append(feature)
# g2.node.append(starts)
# g2.node.append(ends)
# g2.node.append(n1)
# g2.node.append(n2)
#
# # Change the input
# g2.input.append(h.make_tensor_value_info('in', tp.FLOAT, [3, 30]))
#
# # Change the output
# g2.output.append(h.make_tensor_value_info('dt', tp.FLOAT, [1, 1]))
#
# m2 = h.make_model(g2, producer_name='scailable-demo')
#
# save(m2, 'lely_mod_2.onnx')