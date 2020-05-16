import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten
from tensorflow.keras.datasets import fashion_mnist
tf.executing_eagerly()
random_seed = 33
num_classes = 10
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def save_model_as_json_to_disk(filename, model):
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    print("[INFO] Saved model to disk!")


def deterministic_quantization_rule(weight_value):
    if (weight_value > 0.3):
        return 1
    elif (weight_value < -0.3):
        return -1
    else:
        return 0


def quantization_policy(input_weights):
    if len(input_weights.shape) == 1:
        rows = input_weights.shape[0]
        for r in range(rows):
            input_weights[r] = deterministic_quantization_rule(input_weights[r])
    else:
        rows, columns = input_weights.shape
        for r in range(rows):
            for c in range(columns):
                input_weights[r][c] = deterministic_quantization_rule(input_weights[r][c])
    return input_weights


def apply_quantization_policy(quantized_model, policy):
    quantized_model_weights = quantized_model.get_weights()
    # get all weight and bias matrices
    for layer in quantized_model.layers:
        if isinstance(layer, k.layers.Dense):
            print("[INFO] Quantizing Dense")
            layer_wabs = layer.get_weights()
            layer_wabs[0] = policy(layer_wabs[0])
            layer_wabs[1] = policy(layer_wabs[1])
            layer.set_weights(layer_wabs)
        elif isinstance(layer, k.layers.Conv2D):
            print("[INFO] Quantizing Conv2D")
            # todo: add implementation for filter kernels
            layer_wabs = layer.get_weights()
            layer_wabs[0] = policy(layer_wabs[0])
            layer_wabs[1] = policy(layer_wabs[1])
            layer.set_weights(layer_wabs)
        else:
            print("[WARN] Layer instance type not currently supported")
    return quantized_model


# create 32 bit model
model = Sequential()
model = k.Sequential([
    k.layers.Flatten(input_shape=(28, 28)),
    k.layers.Dense(256, activation='relu', kernel_initializer=k.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=random_seed)),
    k.layers.Dense(num_classes, kernel_initializer=k.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=random_seed))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
weights = model.get_weights()

# for layer in model.layers:
#     print(layer.name)
#     print(layer.get_weights())

# get quantized model
quantized_model = tf.keras.models.clone_model(model)
quantized_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

score = quantized_model.evaluate(test_images, test_labels)
print('Quantized Network')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.fit(train_images, train_labels, epochs=1)
score = model.evaluate(test_images, test_labels)
print('Trained Network')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_dir = "/models"
save_model_as_json_to_disk(model_dir+"/model.json", model)
save_model_as_json_to_disk(model_dir+"/q_model.json", model)

quantized_model.set_weights(model.get_weights())
quantized_model = apply_quantization_policy(quantized_model, quantization_policy)
optimizer = Adam(lr=0.1)
quantized_model.compile(optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

score = quantized_model.evaluate(test_images, test_labels)
print('Quantized Network')
print('Test loss:', score[0])
print('Test accuracy:', score[1])




























