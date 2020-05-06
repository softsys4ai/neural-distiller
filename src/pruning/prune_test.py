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
    model_sparsity = prune_metrics.calculate_sparsity_of_model(pruned_model)
    sparsity_per_layer = prune_metrics.calculate_sparsity_per_layer(pruned_model)
    print(f"Model sparsity: {model_sparsity}\n"
          f"Sparsity per Layer: {sparsity_per_layer}\n")

    # Fine tune model
    # Callback needed to maintain sparsity through fine tuning
    callback = [prune_callbacks.MaintainSparsity(), tf.keras.callbacks.TensorBoard(log_dir=f"./pruning/logs/taylor_first_order/{datetime.datetime.now()}")]
    pruned_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=6, callbacks=callback)
    exported_model = pruner.export_pruned_model()
    prune_util.compile_model(exported_model)
    loss, accuracy = exported_model.evaluate(X_train, Y_train)
    print(f"Loss: {loss}\n"
          f"Accuracy: {accuracy}")
    sparsity_percentage, nonzero_percentage = prune_metrics.calculate_sparsity_of_model(exported_model)
    print(f"Sparse Percentage: {sparsity_percentage}\n"
          f"Nonzero Percentage: {nonzero_percentage}")
    filter_model_sparsity = prune_metrics.calculate_filter_sparsity_of_model(exported_model)
    print(f"Percentage of sparse filters: {filter_model_sparsity}")
    # {layer_name: (total_layer_params, total_layer_nonzero_params, total_layer_zero_params,
    #                             percentage_of_sparse_params, percentage_of_nonzero_params)}
    #
    sparsity_per_layer = prune_metrics.calculate_sparsity_per_layer(exported_model)
    for name, (total_layer_params, total_layer_nonzero_params, total_layer_sparse_params,
               percentage_of_sparse_params, percentage_of_nonzero_params) in sparsity_per_layer.items():
        print(f"Layer: {name}\n"
              f"Total layer params: {total_layer_params}\n"
              f"Total layer nonzero params: {total_layer_nonzero_params}\n"
              f"Total layer sparse params: {total_layer_sparse_params}\n"
              f"Percentage of sparse params: {percentage_of_sparse_params}\n"
              f"Percentage of nonzero params: {percentage_of_nonzero_params}")
    filter_sparsity_per_layer = prune_metrics.calculate_filter_sparsity_per_layer(exported_model)
    for name, (num_filters, num_sparse_filters, percentage_sparse_filters) in filter_sparsity_per_layer.items():
        print(f"Layer: {name}\n"
              f"Number of filters: {num_filters}\n"
              f"Number of sparse filters: {num_sparse_filters}\n"
              f"Percentage of sparse filters: {percentage_sparse_filters}")


if __name__ == "__main__":
    test_prune_taylor_first_order_impl()
