"""
@author: Stephen Baione
@description: Test pruning classes and functionality
# TODO:// After POF, start using VGG16 and ResNet50 for pruning
"""

from pruning.pruner import Pruner
from pruning import prune_util
from pruning import prune_callbacks
from pruning import prune_metrics

import tensorflow as tf

import datetime


# Test of taylor_first_order using pruning framework
def test_prune_taylor_first_order_impl():
    # Loading model and dataset
    model: tf.keras.Model = prune_util.load_model("mnist")
    (X_train, Y_train), (X_test, Y_test) = prune_util.load_dataset(dataset="mnist", train_size=5000, test_size=1000)
    prune_util.compile_model(model)
    prune_util.train_model(model, X_train, Y_train, X_test, Y_test, epochs=8)

    # Evaluate unpruned model
    print(model.evaluate(X_test, Y_test))
    print(model.weights)

    # Prune model
    pruner = Pruner(model, X_train, Y_train, X_test, Y_test,
                    prune_level="filter", prune_method="taylor_first_order")
    prune_params = {
        "sparsity": 0.98
    }
    pruner.prune(**prune_params)
    pruned_model = pruner.pruned_model

    # Evaluate prior to fine tuning
    print(pruned_model.evaluate(X_test, Y_test))

    # Fine tune model
    # Callback needed to maintain sparsity through fine tuning
    callback = [prune_callbacks.MaintainSparsity(), tf.keras.callbacks.TensorBoard(log_dir=f"./pruning/logs/taylor_first_order/{datetime.datetime.now()}")]
    pruned_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=6, callbacks=callback)
    exported_model = pruner.export_pruned_model()
    prune_util.compile_model(exported_model)
    print(exported_model.evaluate(X_train, Y_train))

    tf.keras.models.save_model(exported_model, "./pruning/pruned_models/taylor_first_order_98_percent_filter_sparsity.h5")


if __name__ == "__main__":
    test_prune_taylor_first_order_impl()
