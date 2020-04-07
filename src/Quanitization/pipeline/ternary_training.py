# import tensorflow as tf
# import math
#
# weight_init = "he_normal"
# batch_size = 128
# num_classes = 10
# epochs = 1
#
# img_rows, img_cols = 28, 28
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = x_train.astype('float32')
# X_test = x_test.astype('float32')
# x_train = x_train/255
# x_test = x_test/255
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

import math
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

def bound_policy(w):
    if w < -1.0:
        return -1
    elif w > 1.0:
        return 1
    else:
        return w

def ternarize(w):
    if w < -0.5:
        return -1
    elif w > 0.5:
        return 1
    else:
        return 0

apply_bounds = np.vectorize(bound_policy)
apply_ternarization = np.vectorize(ternarize)

def train_step(real_x, real_y):
    # Make prediction with quantized model
    tern_pred_y = ternary_model.predict(real_x.reshape((-1, 28, 28, 1)))
    tern_model_loss = tf.keras.losses.categorical_crossentropy(real_y, tern_pred_y)
    with tf.GradientTape() as tape:
        pred_y = model(real_x.reshape((-1, 28, 28, 1)))
        pred_y.numpy = tern_pred_y
        # Calculate loss w.r.t. quantized model
        model_loss = tf.keras.losses.categorical_crossentropy(real_y, pred_y)
    model_gradients = tape.gradient(model_loss, model.trainable_variables)
    # Update FP model
    optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    # Apply ternary bounds to updated weights
    cnt = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            ws = layer.get_weights()
            ws[0] = apply_bounds(ws[0])
            model.layers[cnt].set_weights(ws)
        cnt += 1
    # todo: set ternary_model weights to ternarized model weights

# Load and pre-process training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train / 255).reshape((-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = (x_test / 255).reshape((-1, 28, 28, 1))
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Hyper-parameters
batch_size = 64
epochs = 50
weight_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None)
optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer2 = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init,
                                 input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init,
#                                  input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1))) # include at front of FC networks
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer=weight_init))
model.add(tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer=weight_init))
model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=weight_init))

ternary_model = tf.keras.models.clone_model(model)

model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
ternary_model.compile(optimizer=optimizer2, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])

cnt = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        ws = layer.get_weights()
        ws[0] = apply_ternarization(ws[0])
        ternary_model.layers[cnt].set_weights(ws)
    cnt += 1

print('Before training (FP):\n', model.evaluate(x_test, y_test, verbose=0))
print('Before training (TN):\n', ternary_model.evaluate(x_test, y_test, verbose=0))

# Training loop
bat_per_epoch = math.floor(len(x_train) / batch_size)
for epoch in range(epochs):
    print(f'\n[epoch {epoch+1}]')
    for i in range(bat_per_epoch):
        n = i * batch_size
        # Ternarize FP model weights and set ternary model weights
        cnt = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                ws = layer.get_weights()
                ws[0] = apply_ternarization(ws[0])
                ternary_model.layers[cnt].set_weights(ws)
            cnt += 1
        train_step(x_train[n:n + batch_size], y_train[n:n + batch_size])
    print('Saving models...')
    ternary_model.save("ternary_conv_model.h5")
    model.save("fp_conv_model.h5")
    print('32 bit model:\n', model.evaluate(x_test, y_test, verbose=0))
    print('2 bit model:\n', ternary_model.evaluate(x_test, y_test, verbose=0))
    # print(model.get_weights())
    # print(ternary_model.get_weights())

print('After training (FP):\n', model.evaluate(x_test, y_test, verbose=0))
print('After training (TN):\n', ternary_model.evaluate(x_test, y_test, verbose=0))



















