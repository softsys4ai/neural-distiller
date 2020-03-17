"""
@author: Stephen Baione (sbaione@email.sc.edu)
@description: TensorFlow Layer Wrapper to extend functionality for neural network pruning.
                Based on pruning Gate Decorator Pattern
"""

import tensorflow as tf

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
        self.original_wandb = layer.get_weights()

        # Validating prune_level
        if prune_level not in ("weights", "filter"):
            raise ValueError("Incompatible prune_level:\n\n"
                             "Supported Levels:\n"
                             "weights\nfilter")

        self.prune_level = prune_level

        # Kwargs
        self.scope = kwargs.get("scope", "")

    # Build function to add mask variable to graph
    def build(self, input_shape):
        super(PruneWrapper, self).build(input_shape)
        if not self.built:
            self.layer.build(input_shape=input_shape)

            layer = self.layer
            # Mask to track weight pruning
            wandb = layer.get_weights()
            weights = wandb[0]
            self.mask = tf.Variable(initial_value=tf.ones(weights.shape, dtype=tf.float32, name=None),
                                    trainable=False,
                                    name=f"{layer.name}_mask",
                                    dtype=tf.float32,
                                    aggregation=tf.VariableAggregation.MEAN,
                                    shape=weights.shape)
        self.built = True

    def call(self, inputs, training=False, **kwargs):
        if training:
            K.learning_phase()

        self.layer.call(inputs, **kwargs)

    def get_filters(self):
        return self.layer.filters

    def get_channels(self):
        return self.get_weights().shape[-2]

    def get_biases(self):
        return self.layer.get_weights()[1]

    def get_original_wandb(self):
        return self.original_wandb

    def get_original_weights(self):
        return self.original_wandb[0]

    def get_original_biases(self):
        return self.original_wandb[1]

    def get_prune_level(self):
        return self.prune_level

    def set_prune_level(self, prune_level):
        if prune_level not in ("weights", "filter", "layer"):
            raise ValueError("Incompatible prune_level:\n\n"
                             "Supported Levels:\n"
                             "weights\nfilter\nlayer")
        self.prune_level = prune_level

    def get_mask(self):
        return tf.convert_to_tensor(self.mask, dtype=self.mask.dtype, name=self.mask.name)

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
        layer = self.layer
        wandb = layer.get_weights()
        original_weights = self.get_original_weights()
        wandb[0] = original_weights
        layer.set_weights(original_weights)
        self.layer = layer

    def prune(self):
        wandb = self.layer.get_weights()
        weights = wandb[0]
        mask = self.get_mask()
        print(weights.shape)

        new_weights = weights * mask
        wandb[0] = new_weights
        self.layer.set_weights(wandb)
