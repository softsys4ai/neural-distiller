
# ternary network training
import os
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from utils import config_reference as cfg
import tensorflow.python.keras.backend as K
# tf.compat.v1.disable_eager_execution()

# write weights to file
def save_weights(models_dir, model, epoch, total_epochs, val_acc, ternary=False):
    if ternary:
        weight_filename = f"ternary_model_{epoch}|{total_epochs}_{format(val_acc, '.5f')}.h5"
    else:
        weight_filename = f"model_{epoch}|{total_epochs}_{format(val_acc, '.5f')}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)
    return model_path

def bound_policy(w):
    if w < -1.0:
        return -0.9999
    elif w > 1.0:
        return 0.9999
    else:
        return w

def ternarize(w):
    if w < -0.33:
        return -1
    elif w > 0.33:
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


# create log directory
print("Creating logs directory...")
log_dir = cfg.log_dir
now = datetime.now()
now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
log_dir = os.path.join(log_dir, "ternary_train_" + cfg.dataset + "_" + now_datetime)
os.mkdir(log_dir)
models_dir = os.path.join(log_dir, "model_weights")
os.mkdir(models_dir)

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
optimizer = tf.keras.optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
optimizer2 = tf.keras.optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)

# Build model
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init,
#                                  input_shape=(28, 28, 1)))
# model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init,
#                                  input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1))) # include at front of FC networks
# model.add(tf.keras.layers.Flatten())
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

# Save model json configuration
model_json = model.to_json()
with open(os.path.join(log_dir, "model.json"), "w") as json_file:
    json_file.write(model_json)

print('Before training (FP):\n', model.evaluate(x_test, y_test, verbose=0))
print('Before training (TN):\n', ternary_model.evaluate(x_test, y_test, verbose=0))

# Training loop
bat_per_epoch = math.floor(len(x_train) / batch_size)
for epoch in range(epochs):
    print(f'\n[epoch {epoch+1}]')
    print(f'lr: {format(K.eval(model.optimizer.lr), ".5f")}')
    for i in tqdm(range(bat_per_epoch)):
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
    model.evaluate(x_test, y_test, verbose=0)
    print('\nEvaluating and saving models...')
    tn_results = ternary_model.evaluate(x_test, y_test, verbose=0)
    fp_results = model.evaluate(x_test, y_test, verbose=0)
    save_weights(models_dir, ternary_model, epoch+1, epochs, tn_results[1], True)
    save_weights(models_dir, model, epoch+1, epochs, fp_results[1], False)
    print('32 bit scores:\n', fp_results)
    print('2 bit scores:\n', tn_results)

print('\nAfter training (FP):\n', model.evaluate(x_test, y_test, verbose=0))
print('After training (TN):\n', ternary_model.evaluate(x_test, y_test, verbose=0))

