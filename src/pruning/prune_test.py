"""
@author: Stephen Baione
@description: Test pruning classes and functionality
# TODO:// After POF, start using VGG16 and ResNet50 for pruning
"""

from pruning.pruner import Pruner
from pruning import ranker
from pruning import prune_util

import tensorflow as tf


# Test of taylor_first_order using pruning framework
def test_prune_taylor_first_order_impl():
    model = prune_util.load_model("mnist")
    (X_train, Y_train), (X_test, Y_test) = prune_util.load_dataset(dataset="mnist", test_size=800)
    prune_util.compile_model(model)
    prune_util.train_model(model, X_train, Y_train, X_test, Y_test, epochs=4)
    print(prune_util.evaluate_percentage_of_zeros(model))
    original_weights = model.layers[1].get_weights()[0]

    pruner = Pruner(model, X_train, Y_train, X_test, Y_test,
                    prune_level="filter", prune_method="taylor_first_order")
    prune_params = {
        "continue_epochs": 2,
        "prune_n_lowest": 100
    }
    pruner.prune(**prune_params)
    pruned_model = pruner.pruned_model
    print(prune_util.evaluate_percentage_of_zeros(pruned_model))
    print(pruned_model.evaluate(X_test, Y_test))


if __name__ == "__main__":
    test_prune_taylor_first_order_impl()
