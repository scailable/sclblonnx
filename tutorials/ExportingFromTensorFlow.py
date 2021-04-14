# import keras2onnx
# import numpy as np
# from sklearn import datasets
# import tensorflow as tf
#
#
# # Get data from sklearn Example:
# X, y = datasets.load_diabetes(return_X_y=True)
# print(X.shape)
#
#
# # Setup a DNN using TF
# # tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
# #
# # dnn_model = tf.keras.Sequential()
# # dnn_model.add(layers.Dense(64, activation='relu'))
# # dnn_model.add(layers.Dense(64, activation='relu'))
# # dnn_model.add(layers.Dense(1))
# # dnn_model.compile(loss='mse', optimizer='sgd')
# # # train the model
# # history = model.fit( X, y,validation_split=0.2,verbose=0, epochs=1)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[10]),
#     tf.keras.layers.Dense(32, activation=tf.nn.relu),
#     tf.keras.layers.Dense(32, activation=tf.nn.relu),
#     tf.keras.layers.Dense(1)
#   ])
#
# optimizer = tf.keras.optimizers.RMSprop(0.0099)
# model.compile(loss='mean_squared_error', optimizer=optimizer)
# model.fit(X, y, epochs=10)
#
# yhat = model.predict(X)  # generate predictions locally
# print(yhat)
#
#
#
# # save model
# # tf.saved_model.save(dnn_model, "tmp_model")
# mod = keras2onnx.convert_keras(model, model.name, target_opset=13)
# #keras2onnx.save_model(mod,  "onnx/dnn-tf.onnx")

# load the model using sclblonnx
import sclblonnx as so
g = so.graph_from_file("onnx/dnn-sklearn-test.onnx")
so.display(g)
g = so.clean(g)
so.check(g)