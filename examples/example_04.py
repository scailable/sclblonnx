#import tensorflow as tf
#import keras2onnx
#from tensorflow.keras import layers
from sklearn import datasets
import sclblonnx as so
import numpy as np

"""
EXAMPLE 4: Converting a model to ONNX from TensorFlow and fixing the dynamic input & output.

Here we first train a simple tf model (using keras) using the sklearn diabetes dataset. 
We store the model to onnx using the keras2onnx package (please note that this is in flux).
See https://pypi.org/project/keras2onnx/

Next, we open and inspect the model graph using Scailable tools. Cleaning the graph gives a warning
regarding the input being dynamic; we show how to fix this and how to evaluate the graph.

"""

# Get training data from sklearn
X, y = datasets.load_diabetes(return_X_y=True)

train = False  # Prevent training on every run; the model is stored, /onnx/tf-keras-dynamic.onnx
if train:  # Don't retrain everytime, the model is stored.

    # Create the model
    dnn_model = tf.keras.Sequential()
    dnn_model.add(layers.Dense(64, activation='relu'))
    dnn_model.add(layers.Dense(64, activation='relu'))
    dnn_model.add(layers.Dense(1))

    dnn_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.SGD())

    # train the model (use .predict for local predictions)
    history = dnn_model.fit(
        X, y,
        validation_split=0.2,
        verbose=0, epochs=300)

    # Save model (note, the convert_keras() function is undergoing change in different
    # versions of tf / onnx).
    # You might need: tf.compat.v1.disable_eager_execution()
    # or use the tf2onnx tool at https://github.com/onnx/tensorflow-onnx
    onnx_model = keras2onnx.convert_keras(dnn_model, dnn_model.name)
    keras2onnx.save_model(onnx_model,  "onnx/tf-keras-dynamic.onnx")


# load the model using sclblonnx
g = so.graph_from_file("onnx/tf-keras-dynamic.onnx")
# so.display(g)

# check() and clean()
so.check(g)
g = so.clean(g)  # Fails due to dynamic size

# Note, while this model passes check(), clean() provides a warning message due to the dynamic input (Nx10).
# This occurs because the training data is N long. However, for inference we would like it to be 1x10
# Let's fix this by changing the input to static.
so.list_inputs(g)
g = so.replace_input(g, "input_1", "FLOAT", [1, 10])

# And do the same for the output
output = so.replace_output(g, "output_1", "FLOAT", [1, 1])  # Check this one...

# Now we do pass all checks, and we can look at the graph
so.check(g)
g = so.clean(g)
so.display(g)

# However, we might not like the tf default input and output names:
g = so.rename_input(g, "input_1", "in")
g = so.rename_output(g, "output_1", "result")
so.display(g)

# And now we can call it locally:
input_example = np.array([X[1, ]]).astype(np.float32)  # Note the extra brackets to create 1x10
example = {"in": input_example}
result = so.run(g,
                inputs=example,
                outputs=["result"]
                )
print(result)

# Finally, we can store the changed graph:
so.graph_to_file(g, "onnx/tf-keras-static.onnx")


'''
Additional usage of sclblpy for upload and evaluation:

# Import sclblpy
import sclblpy as sp

# Upload model
sp.upload_onnx("onnx/tf-keras-static.onnx", docs={"name": "Example_04: TF-Keras-static", "documentation": "None provided."})

# Example input for a Scailable runtime:
input_str = so.sclbl_input(example, _verbose=False)

# Run
sp.run("0d7db3c7-a111-11eb-9acc-9600004e79cc", input_str)
'''
