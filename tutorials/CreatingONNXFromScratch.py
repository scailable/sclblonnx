import sclblonnx as so
"""
First a simple example of a graph that adds two numbers
"""
g = so.empty_graph()
n1 = so.new_node('Add', inputs=['x1', 'x2'], outputs=['sum'], name='add_node')
g.node.append(n1)
g = so.add_input(g, 'x1', "float", [1])
g = so.add_input(g, 'x2', "float", [1])
g = so.add_output(g, 'sum', "float", [1])
g = so.clean(g)  # Use cleaning and run onnx checker
so.check(g)  # Check if can be uploaded to scailable
so.display(g)  # Open netron to inspect the graph
result = so.run(g, {"x1": 1.2, "x2": 2.5})
print(result)
so.graph_to_file(g, "test_sclblonnx.onnx")   # And yay, this one converts
# Todo(McK): discuss the input with robin (for online version)

#
# # Next, an example of a graph used for image processing.
#
# # 1. We start by opening the reference image and creating the necessary ONNX constants:
#
# # The baseline empty container image (average of the 7 empty images)
# reference_image=np.array(Image.open(image_folder+"empty-average.JPG"),dtype=np.int64)
#
# # The baseline image as ONNX constant:
# c_base = h.make_node('Constant', inputs=[], outputs=['c_base'], name="c_base_node",
#         value=h.make_tensor(name="c_base_value", data_type=tp.INT64,
#         dims=reference_image.shape,
#         vals=reference_image.flatten()))
#
# # The threshold value as ONNX constant; here we select an average of 25 points difference (3000000=300*400*25)
# image_threshold = numpy.array([3000000]).astype(numpy.int64)
# c_cut = h.make_node('Constant', inputs=[], outputs=['c_cut'], name="c_cut_node",
#         value=h.make_tensor(name="c1v", data_type=tp.INT64,
#         dims=image_threshold.shape,
#         vals=image_threshold.flatten()))
#
#
# # 2. Next, we declare the functional ONNX nodes in order of appearance:
#
# # Subtract input xin from baseline
# n1 = h.make_node('Sub', inputs=['xin', 'c_base'], outputs=['min'], name='n1')
#
# # Compute absolute values of the remaining difference
# n2 = h.make_node('Abs', inputs=['min'], outputs=['abs'], name="n2")
#
# # Sum all the absolute differences
# n3 = h.make_node('ReduceSum', inputs=['abs'], outputs=['sum'], name="n3", keepdims=0)
#
# # See if the sum is less than image_threshold; if it is the image is empty
# n4 = h.make_node('Less', inputs=['sum','c_cut'], outputs=['out'], name="n4")
#
#
# # 3. Finally, we create the resulting ONNX graph
#
# # Create the graph
# g1 = h.make_graph([c_base, c_cut, n1,n2,n3,n4], 'convert_image',
#         [h.make_tensor_value_info('xin', tp.INT64, target.shape)],
#         [h.make_tensor_value_info('out', tp.BOOL, [1])])
#
# # Create the model and check
# m1 = h.make_model(g1, producer_name='scailable-demo')
# checker.check_model(m1)
#
# # Save the model
# save(m1, 'empty-container.onnx')

# Simple test for the first example
# sess = rt.InferenceSession('lely_mod_1.onnx')
# out = sess.run(["dt"], {"in": example.astype(np.float32)})[0]
# print("The predicted time till empty for the first example is {}".format(out))