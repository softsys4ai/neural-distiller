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

import tensorflow_model_optimization as tfmot

from pprint import pprint as pprint


class Pruner(object):
    def __init__(self, model: Model, X_train, Y_train, X_test, Y_test, prune_level: str, prune_method: str):
        self.model = model
        self.pruned_model = None

        if prune_level not in get_supported_prune_levels():
            raise ValueError("Unsupported prune_level")
        # if prune_method not in get_supported_prune_methods():
        #    raise ValueError("Unsupported prune_method")

        self.prune_level = prune_level
        self.prune_method = prune_method
        self.ranker = Ranker(model)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    # sparsity is percentage of network params that should remain after prune
    def prune(self, **kwargs):
        """
        Executes pruning method on model
        :param kwargs:
            Kwargs will vary depending on prune method chosen. Reference config for supported methods and corresponding
            keyword arguments
        :return:
        """

        def _get_poly_prune_schedule(**kwargs):
            initial_sparsity = kwargs.get("initial_sparsity", 0.2)
            final_sparsity = kwargs.get("final_sparsity", 0.8)
            begin_step = kwargs.get("begin_step", 1000)
            end_step = kwargs.get("end_step", 2000)
            return tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                        final_sparsity=final_sparsity,
                                                        begin_step=begin_step,
                                                        end_step=end_step)

        def _prune_low_magnitude(_model, _X_train, _Y_train, _X_test, _Y_test, epochs=10, **kwargs):
            prune_schedule = None
            pruning_schedule_type = kwargs.get("pruning_schedule_type", "polynomial_decay")

            # Collecting parameters for low magnitude pruning
            if pruning_schedule_type == "polynomial_decay":
                prune_schedule = _get_poly_prune_schedule(**kwargs)

            # Loading model with Prune Wrappers applied
            model_to_prune: Model = tfmot.sparsity.keras.prune_low_magnitude(_model, prune_schedule)
            model_to_prune.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model_to_prune.summary()

            # Fine tune model
            callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
            model_to_prune.fit(_X_train,
                               _Y_train,
                               epochs=epochs,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=(_X_test, _Y_test))

            _pruned_model = tfmot.sparsity.keras.strip_pruning(model_to_prune)
            _pruned_model.summary()
            _pruned_model.compile(optimizer="adam",
                                  loss="sparse_categorical_crossentropy",
                                  metrics=["accuracy"])
            return _pruned_model

        model = self.get_model()
        assert model is not None
        pruned_model = self.pruned_model

        (X_train, Y_train), (X_test, Y_test) = (self.X_train, self.Y_train), (self.X_test, self.Y_test)
        for data in [X_train, Y_train, X_test, Y_test]:
            assert data is not None

        prune_method = self.get_prune_method()
        if prune_method == "low_magnitude":
            pruned_model = _prune_low_magnitude(model, X_train, Y_train, X_test, Y_test, **kwargs)

        self.pruned_model = pruned_model

    def rank_weights(self):
        ranker = self.ranker
        ranked_weights = ranker.rank(self.X_test, self.Y_test, self.prune_method, self.prune_level)
        return ranked_weights

    def evaluate_pruned_model(self, verbose=0):
        pruned_model = self.pruned_model
        X_test = self.X_test
        Y_test = self.Y_test
        return pruned_model.evaluate(X_test, Y_test, verbose=verbose)

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
