"""
 * @author Stephen Baione
 * @date 01/26/2020
 * @description Implements pruner object for pruning neural networks
"""

# What do I need?
# Model, X_test, Y_test, X_train, Y_train, Cost function, rank function
#
#
#
#
#
#
#
#
#




# Step 1: Implement ranking algorithms for neural network pruning:
# - Taylor: First order
# - Oracle
# - Taylor: Second Order        for
# - layer-wise
# - weight-wise
# - filter-wise


# Step 2: Implement pruning functionality for neural network
# Package: Tensorflow_model_optimization

# Step 3: Tie it all together: rank -> prune -> fine tune -> repeat


# Class for scheduling and executing pruning  for keras model

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from tensorflow.python.keras.layers import Layer, Conv2D
from tensorflow.python.keras.models import Model
from .prune_config import get_supported_prune_levels, get_supported_prune_methods
from .prune_wrapper import PruneWrapper
from .ranker import Ranker

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
        print(ranked_weights)

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








