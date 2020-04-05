import tensorflow as tf
from tensorflow.python.keras import datasets

from tensorflow.python.keras.models import Model
from models.tf_official_models import official
from official.vision.image_classification import mnist_main

import numpy as np

import tempfile
import zipfile
import os


def load_dataset(dataset: str, test_size=10000):
    if dataset is None:
        raise ValueError("Must include a string value to indicate dataset")

    ds = None
    if dataset == "mnist":
        ds = datasets.mnist

    if ds is None:
        return None

    (X_train, Y_train), (X_test, Y_test) = ds.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
    (X_train, Y_train) = (X_train[:test_size], Y_train[:test_size])
    return (X_train, Y_train), (X_test, Y_test)


def load_model(model_type: str):
    model = None

    if model_type == "mnist":
        model = mnist_main.build_model()
        model.summary()

    return model


def compile_model(model: Model, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_model(model, X_train, Y_train, X_test, Y_test, callbacks=None, epochs=5, verbose=1):
    if callbacks is not None:
        model.fit(X_train, Y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test, Y_test))
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=verbose)


def save_model_h5(model: Model, file_path=None, include_optimizer=True):
    if file_path is None:
        _, file_path = tempfile.mkstemp(".h5")
    assert ".h5" == file_path[-3:]

    print(f"Saving model to file: {file_path}")
    tf.keras.models.save_model(model, file_path, include_optimizer=include_optimizer)
    return file_path


def compress_model_zip(model: Model, file_path=None):
    uncompressed_path = save_model_h5(model, include_optimizer=False)

    if file_path is None:
        _, file_path = tempfile.mkstemp(".zip")
    assert ".zip" == file_path[-4:]

    print(f"Zipping file to: {file_path}")
    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(uncompressed_path)
    return file_path


def evaluate_model_size(model: Model, name="", uncompressed_path=None, compressed_path=None):
    _uncompressed_path = save_model_h5(model, file_path=uncompressed_path, include_optimizer=False)
    _compressed_path = compress_model_zip(model, file_path=compressed_path)
    return f"Size of {name} model before compression {os.path.getsize(_uncompressed_path) / float(2**20)} MB\n" \
           f"Size of {name} model after compression {os.path.getsize(_compressed_path) / float(2**20)} MB\n"


def format_experiment_name(prune_method: str, prune_level: str, model_type: str, **prune_params):
    params = ""
    for key, value in prune_params.items():
        if key == "pruning_schedule":
            value = value.get_config().get("class_name")
        params += f"{key}_{value}_"
    params = params[:-1]

    return f"method_{prune_method}_level_{prune_level}_model_type_{model_type}_{params}"


def evaluate_percentage_of_zeros(model: Model):
    evaluation = ""
    for i, w in enumerate(model.get_weights()):
        evaluation += f"{model.weights[i].name} -- Total: {w.size}, Zeros: {np.sum(w==0) / w.size * 100}\n"
    evaluation += "\n"
    return evaluation
