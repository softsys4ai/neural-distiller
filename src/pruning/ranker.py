"""
@author: Stephen Baione (sbaione@email.sc.edu)
@desription: Implementation of ranker object, that applies neural network pruning ranking techniques and returns the
results. When results are returned, execution of pruning is controlled by pruner, based on specified sparsity.
"""

from pruning.prune_config import get_supported_prune_levels, get_supported_prune_methods
from pruning.prune_wrapper import PruneWrapper

import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

"""
TODO: Implement ranking methods -
oracle (absdeltacost)
taylor
L1 norm
L2 norm

"""

""" Ranks layers of type layer_type in model by rank of type rank_method

Returns list of ints that represent layer index
"""


class Ranker(object):
    def __init__(self, model: Model):
        super(Ranker, self).__init__()
        self._model = model

    def rank(self, X_test, Y_test, rank_method: str, rank_level: str, **kwargs):
        self._validate_rank_method(rank_method)
        self._validate_rank_level(rank_level)

        rank_func = self.get_method_to_func_mapping().get(rank_method)
        rank_func(X_test, Y_test, rank_level, **kwargs)

    def rank_taylor_first_order(self, X_test, Y_test, rank_level: str, **kwargs):
        taylor_funcs = {
            "filter": self._rank_taylor_first_order_by_filters,
        }
        taylor_func = taylor_funcs.get(rank_level)
        return taylor_func(X_test, Y_test, **kwargs)

    def rank_low_magnitude(self, X_test, Y_test, **kwargs):
        pass

    def _rank_taylor_first_order_by_filters(self, X_test, Y_test, **kwargs):
        model = self._model
        """
        taylor_first_order:
        Delta-Cost is estimated as the absolute product of the gradient of cost function w.r.t activation and the activation
        For multivariate, output:
        Θ(z) = abs(1/m * sum((dc/dAct.) * Act.))
        :return {score: (layer_index, channel_index, conv_filter_index)}
        """
        ranks = {}
        # TODO:// Make loss obj variable to models
        loss_obj = SparseCategoricalCrossentropy(from_logits=True)

        # This naming convention follows ResNet50, which I've been testing with. Must be changed to be more general.
        # Probably should parse layers and populate list based on type(layer)
        inputs = X_test[:100]
        labels = Y_test[:100]
        for index, layer in enumerate(model.layers):
            if isinstance(layer, PruneWrapper) and not isinstance(layer, tf.keras.layers.InputLayer):
                if isinstance(layer.layer, tf.keras.layers.Conv2D):
                    filters = range(layer.get_filters())
                    channels = range(layer.get_channels())
                    mask = layer.get_mask()

                    """
                    For each filter in each channel of each layer:
                    Set the mask over the filter
                    Calculate the estimated loss using the first_order_taylor_expansion
                    Store {estimated_cost: (layer, channel, filter_index)}
                    
                    """
                    print(f"Pruning layer {layer.name} with {filters} filters and {channels} channels")
                    for channel in channels:
                        for conv_filter_index in filters:
                            new_mask_vals = mask.numpy()
                            new_mask_vals[:, :, channel, conv_filter_index] = 0.0
                            layer.set_mask(new_mask_vals)
                            layer.prune()
                            # Calculating gradients and activation
                            with tf.GradientTape() as tape:
                                tape.watch(layer.layer.output)
                                predictions = model(inputs)
                                loss = loss_obj(labels, predictions)
                            # Activation of weights of conv_layer
                            activation = tape.watched_variables()[0]
                            grads = tape.gradient(loss, activation)
                            score = tf.math.reduce_mean(grads * activation).numpy()
                            ranks[score] = (index, channel, conv_filter_index)
                            layer.reset_mask()
                            layer.revert_layer()
        return sorted(ranks.items())

    def rank_taylor_second_order(self, X_test, Y_test, rank_level: str, **kwargs):
        taylor_funcs = {
            "filter": self._rank_taylor_second_order_by_filters,
        }
        taylor_func = taylor_funcs.get(rank_level)
        taylor_func(X_test, Y_test, **kwargs)

    def _rank_taylor_second_order_by_filters(self, X_test, Y_test, **kwargs):
        pass

    def rank_oracle(self, X_test, Y_test, rank_level: str, **kwargs):
        oracle_funcs = {
            "filter": self._rank_oracle_by_filters,
            "weights": self._rank_oracle_by_weights
        }
        oracle_func = oracle_funcs.get(rank_level)
        oracle_func(X_test, Y_test, **kwargs)

    def _rank_oracle_by_layer(self, X_test, Y_test, **kwargs):
        pass

    def _rank_oracle_by_filters(self, X_test, Y_test, **kwargs):
        pass

    def _rank_oracle_by_weights(self, X_test, Y_test, **kwargs):
        pass

    def get_method_to_func_mapping(self):
        return {"taylor_first_order": self.rank_taylor_first_order,
                "taylor_second_order": self.rank_taylor_second_order,
                "oracle": self.rank_oracle}

    @staticmethod
    def _validate_rank_level(rank_level):
        if rank_level not in get_supported_prune_levels():
            raise ValueError("Unsupported rank_level")

    @staticmethod
    def _validate_rank_method(rank_method):
        if rank_method not in get_supported_prune_methods():
            raise ValueError("Unsupported rank_method")
