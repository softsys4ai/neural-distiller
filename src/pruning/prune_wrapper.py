import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Conv2D, Wrapper
from tensorflow.python.keras.models import Model, Sequential

from tensorflow_model_optimization.python.core.sparsity import keras as pruning

import numpy as np
from scipy import stats

import os
import sys
import re


"""
Wrapper implementation notes:
def build(self, input_shape): -> 
"""

class PruneWrapper(Wrapper):
    """
    Tensorflow Layer Wrapper, to add pruning_functionality
    :param
    layer - Tensorflow keras Layer that will be wrapped
    prune_level - The level of the neural network in which the pruning will be applied to
    prune_method - The method that will be used to evaluate and rank model at the prune_level
    """
    def __init__(self, layer: Layer, prune_level: str = "weights", **kwargs):
        super(PruneWrapper, self).__init__(layer, **kwargs)

        # Setting layer variable and saving original weights for possible reversion
        self._layer = layer
        self.original_wandb = layer.get_weights()

        # Validating prune_level
        if prune_level not in ("weights", "filter"):
            raise ValueError("Incompatible prune_level:\n\n" /
                             "Supported Levels:\n" /
                             "weights\nfilter")

        self.prune_level = prune_level

        # Kwargs
        self.scope = kwargs.get("scope", "")

    # Build function to add mask variable to graph
    def build(self, input_shape):
        super(PruneWrapper, self).build(input_shape)

        layer = self._get_layer()

        # Mask to track weight pruning
        wandb = layer.get_weights()
        weights = wandb[0]
        with tf.variable_scope(self.scope):
            self.mask = tf.Variable(initial_value=tf.ones(weights.shape, dtype=tf.float32, name=None),
                               trainable=False,
                               name="mask",
                               dtype=tf.float32,
                               aggregation=tf.VariableAggregation.MEAN,
                               shape=weights.shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def add_update(self, updates, inputs=None):
            pass

        return self.layer.call(inputs)

    def _get_layer(self):
        return self._layer

    def get_filters(self):
        return self._layer.filters

    def get_channels(self):
        return self.get_weights().shape[-2]

    def get_wandb(self):
        return self._layer.get_weights()

    def get_weights(self):
        return self._layer.get_weights()[0]

    def get_biases(self):
        return self._layer.get_weights()[1]

    def get_original_wandb(self):
        return self.original_wandb

    def get_original_weights(self):
        return self.original_wandb[0]

    def get_original_biases(self):
        return self.original_wandb[-1]

    def get_prune_level(self):
        return self.prune_level

    def set_prune_level(self, prune_level):
        if prune_level not in ("weights", "filter", "layer"):
            raise ValueError("Incompatible prune_level:\n\n" /
                             "Supported Levels:\n" /
                             "weights\nfilter\nlayer")
        self.prune_level = prune_level

    def get_mask(self):
        return tf.convert_to_tensor_or_sparse_tensor(self.mask)

    def set_mask(self, new_mask_vals: np.ndarray):
        if new_mask_vals.shape != self.mask.shape:
            raise ValueError("new_mask_vals must be same size as mask")
        self.mask.assign(new_mask_vals)

    def reset_mask(self):
        mask = self.mask
        new_mask = tf.ones_like(mask)
        self.set_mask(new_mask)

    def revert_layer(self):
        self.reset_mask()

        layer = self._get_layer()
        wandb = self.get_wandb()
        original_weights = self.get_original_weights()
        wandb[0] = original_weights
        layer.set_weights(original_weights)
        self._layer = layer

    def prune(self):
        layer = self._layer
        weights = self.get_weights()
        wandb = self.get_wandb()
        mask = self.get_mask()

        new_weights = weights * mask
        wandb[0] = new_weights
        layer.set_weights(wandb)
        self._layer = layer

