# #import tensorflow as tf
# import keras2onnx
# from tensorflow.keras import layers
#
# # Generate data for model training
# dataset = df = np.row_stack([i.T for i in test_data])
# X = dataset[:, [0, 2]]  # Height and feed vectors
# y = dataset[:, [1]]  # Time vector
#
# dnn_model = tf.keras.Sequential()
# dnn_model.add(layers.Dense(64, activation='relu'))
# dnn_model.add(layers.Dense(64, activation='relu'))
# dnn_model.add(layers.Dense(1))
#
# dnn_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.SGD())
#
# # train the model
# history = dnn_model.fit(
#     X, y,
#     validation_split=0.2,
#     verbose=0, epochs=300)
#
# # save model
# onnx_model = keras2onnx.convert_keras(dnn_model, dnn_model.name)
# keras2onnx.save_model(onnx_model,  'dnn.onnx')