import tensorflow as tf

from tensorflow.python.keras.models import Model

import tempfile
import zipfile
import os


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


def evaluate_model_size(models: [Model], name="", uncompressed_path=None, compressed_path=None):
    for model in models:
        _uncompressed_path = save_model_h5(model, file_path=uncompressed_path, include_optimizer=False)
        _compressed_path = compress_model_zip(model, file_path=compressed_path)
        print(f"Size of {name} model before compression {os.path.getsize(_uncompressed_path) / float(2**20)}")
        print(f"Size of {name} model after compression {os.path.getsize(_compressed_path) / float(2**20)}")
