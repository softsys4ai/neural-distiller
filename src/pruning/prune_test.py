"""
@author: Stephen Baione
@description: Test pruning classes and functionality
# TODO:// After POF, start using VGG16 and ResNet50 for pruning
"""

from pruning.pruner import Pruner

import tensorflow as tf
from tensorflow.python.keras import datasets

from tensorflow.python.keras.layers import Layer, Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.python.keras.models import Model, Sequential
from models.tf_official_models import official
from official.vision.image_classification import mnist_main

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

import numpy as np


def load_dataset(batch_size=10000):
    mnist_ds = datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist_ds.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
    (X_train, Y_train) = (X_train[:batch_size], Y_train[:batch_size])
    return (X_train, Y_train), (X_test, Y_test)


def load_model():
    model = mnist_main.build_model()
    model.summary()
    return model


def compile_model(model: Model, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_model(model, X_train, Y_train, X_test, Y_test, epochs=5):
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test))


def test():
    (X_train, Y_train), (X_test, Y_test) = load_dataset(batch_size=10000)
    model = load_model()
    compile_model(model)
    train_model(model, X_train, Y_train, X_test, Y_test, epochs=5)
    pruner = Pruner(model, X_train, Y_train, X_test, Y_test, "weight", "low_magnitude")
    pruner.prune()
    evaluation = pruner.evaluate_pruned_model()
    print(evaluation)


if __name__ == "__main__":
    test()
