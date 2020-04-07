from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as k
from keras import losses
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

model = Sequential()
model.add(Dense(12, input_dim=4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

inputs = np.random.random((1, 4))
outputs = model.predict(inputs)
targets = np.random.random((1, 1))
rmse = sqrt(mean_squared_error(targets, outputs))

print("===BEFORE WALKING DOWN GRADIENT===")
print("outputs:\n", outputs)
print("targets:\n", targets)
print("RMSE:", rmse)


def descend(steps=40, learning_rate=5.0, learning_decay=0.95):
    for s in range(steps):

        # If your target changes, you need to update the loss
        loss = losses.mean_squared_error(targets, model.output)

        #  ===== Symbolic Gradient =====
        # Tensorflow Tensor Object
        gradients = k.gradients(loss, model.trainable_weights)

        # ===== Numerical gradient =====
        # Numpy ndarray Objcet
        evaluated_gradients = sess.run(gradients, feed_dict={model.input: inputs})

        # For every trainable layer in the network
        for i in range(len(model.trainable_weights)):

            layer = model.trainable_weights[i]  # Select the layer

            # And modify it explicitly in TensorFlow
            sess.run(tf.assign_sub(layer, learning_rate * evaluated_gradients[i]))

        # decrease the learning rate
        learning_rate *= learning_decay

        outputs = model.predict(inputs)
        rmse = sqrt(mean_squared_error(targets, outputs))
        print(f"RMSE: {rmse}, Learning Rate: {learning_rate}")
        print(outputs)
        print(targets)


if __name__ == "__main__":
    # Begin TensorFlow
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    descend(steps=200)

    final_outputs = model.predict(inputs)
    final_rmse = sqrt(mean_squared_error(targets, final_outputs))

    print("===AFTER STEPPING DOWN GRADIENT===")
    print("outputs:\n", final_outputs)
    print("targets:\n", targets)