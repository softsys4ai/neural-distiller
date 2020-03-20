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
from official.r1.mnist import mnist as tf_mnist_model

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

import numpy as np


def load_dataset(batch_size=10000):
    mnist = datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    (X_train, Y_train), (X_test, Y_test) = (X_train[:batch_size], Y_train[:batch_size]), \
                                           (X_test[:batch_size], Y_test[:batch_size])
    X_train, X_test = np.reshape(X_train, (batch_size, 784)), np.reshape(X_test, (batch_size, 784))
    return (X_train, Y_train), (X_test, Y_test)


def load_model():
    model = tf_mnist_model.create_model("channels_last")
    model.summary()
    return model


def compile_model(model: Model, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(optimizer, loss, metrics)


def train_model(model, X_train, Y_train, epochs=15):
    model.fit(X_train, Y_train, epochs=epochs)


def test():
    (X_train, Y_train), (X_test, Y_test) = load_dataset(batch_size=10000)
    model = load_model()
    compile_model(model)
    train_model(model, X_train, Y_train, epochs=5)
    pruner = Pruner(model, X_train, Y_train, X_test, Y_test, "weight", "low_magnitude")
    pruner.prune()
    evaluation = pruner.evaluate_pruned_model()
    print(evaluation)


if __name__ == "__main__":
    test()
