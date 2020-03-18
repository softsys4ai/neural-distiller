"""
 * @author Stephen Baione
 * @date 01/26/2020
 * @description Implements pruner object for pruning neural networks
"""

# Implement ranking algorithms for neural network pruning:
# - Taylor: First order
# - Oracle
# - Taylor: Second Order        for
# - layer-wise
# - weight-wise
# - filter-wise


# Implement pruning functionality for neural network
# Package: TensorFlow_model_optimization

# Tie it all together: rank -> prune -> fine tune -> repeat

from pruning.prune_config import get_supported_prune_levels, get_supported_prune_methods
from pruning.prune_wrapper import PruneWrapper
from pruning.ranker import Ranker

import tensorflow as tf

from tensorflow.python.keras.layers import Layer, Conv2D
from tensorflow.python.keras.models import Model

from pprint import pprint as pprint


class Pruner(object):
    def __init__(self, model: Model, X_test, Y_test, prune_level: str, prune_method: str):
        self.model = model

        if prune_level not in get_supported_prune_levels():
            raise ValueError("Unsupported prune_level")
        if prune_method not in get_supported_prune_methods():
            raise ValueError("Unsupported prune_method")

        self.prune_level = prune_level
        self.prune_method = prune_method
        self.ranker = Ranker(model)
        self.X_test = X_test
        self.Y_test = Y_test

    # sparsity is percentage of network params that should remain after prune
    def prune(self, sparsity: float):
        params = self.model.count_params()
        sparsity = int(params * sparsity)
        ranked_weights = self.rank_weights()
        pprint(ranked_weights)
        # TODO:// Prune specified number of weights based on sparsity

    def rank_weights(self):
        ranker = self.ranker
        ranked_weights = ranker.rank(self.X_test, self.Y_test, self.prune_method, self.prune_level)
        return ranked_weights

    def get_model(self):
        return self.model

    def get_prune_level(self):
        return self.prune_level

    def set_prune_level(self, prune_level):
        self.prune_level = prune_level

    def get_prune_method(self):
        return self.prune_method

    def set_prune_method(self, prune_method):
        self.prune_method = prune_method


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = prune_test.load_dataset()
    model = prune_test.load_model()
    prune_test.train_model(model, X_train, Y_train, epochs=2)
    pruner = Pruner(model, X_test, Y_test, prune_level="filter", prune_method="taylor_first_order")
    pruner.prune(sparsity=0.5)
