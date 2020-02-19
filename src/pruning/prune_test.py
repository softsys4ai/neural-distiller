import tensorflow.compat.v1 as tf

from tensorflow.python.keras.layers import Layer, Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Model, Sequential

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from tensorflow.python.keras import datasets

from prune_wrapper import PruneWrapper
from pruner import Pruner
from ranker import Ranker

def load_dataset():
    mnist = datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    return (X_train, Y_train), (X_test, Y_test)


def load_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, epochs=10):
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


def test_ranker(model, new_model, X_test, Y_test):
    ranker_old = Ranker(model)
    ranker_new = Ranker(new_model)
    old_grads = ranker_old._get_gradients(model, X_test, Y_test)
    new_grads = ranker_new._get_gradients(new_model, X_test, Y_test)
    print("thing")


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
    model, new_model = test_wrapper(model, X_train, Y_train, X_test, Y_test)
    test_ranker(model, new_model, X_test, Y_test)


    
if __name__ == "__main__":
    test()
