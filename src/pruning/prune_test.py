"""
@author: Stephen Baione
@description: Test pruning classes and functionality
# TODO:// After POF, start using VGG16 and ResNet50 for pruning
"""

from pruning.prune_wrapper import PruneWrapper
from pruning.ranker import Ranker

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.layers import Layer, Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.models import Model, Sequential

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from tensorflow.python.keras import datasets


def load_dataset():
    mnist = datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
    (X_train, Y_train), (X_test, Y_test) = (X_train[:10000], Y_train[:10000]), (X_test[:10000], Y_test[:10000])
    return (X_train, Y_train), (X_test, Y_test)


def load_model():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(16, kernel_size=5, activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, Y_train, epochs=15):
    model.fit(X_train, Y_train, epochs=epochs)


def test_wrapper(model, X_train, Y_train, X_test, Y_test):
    original_predictions = model(X_test, training=False)
    wandb = model.get_weights()
    test_layer = model.layers[1]
    wrapped_layer = PruneWrapper(test_layer)
    wrapped_layer.build(test_layer.input_shape)
    mask = wrapped_layer.get_mask()
    new_mask_vals = mask.numpy()
    new_mask_vals[0, :] = 0.0
    wrapped_layer.set_mask(new_mask_vals)
    wrapped_layer.prune()
    layers = [layer for layer in model.layers]
    layers[1] = wrapped_layer
    new_model = rebuild_model(model, layers, X_train, Y_train)
    return model, new_model


def rebuild_model(model, layers, X_train, Y_train):
    new_model = Sequential()
    for layer in layers:
        new_model.add(layer)
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return new_model
    
    
def test():
    (X_train, Y_train), (X_test, Y_test) = load_dataset()
    model = load_model()
    train_model(model, X_train, Y_train)




