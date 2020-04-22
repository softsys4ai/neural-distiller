"""
@author: Stephen Baione (sbaione@email.sc.edu)
@description: TensorFlow Layer Wrapper to extend functionality for neural network pruning.
                Based on pruning Gate Decorator Pattern
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Conv2D, Wrapper

import numpy as np


class PruneWrapper(Wrapper):
    """
    __init__, build, and call functions required to properly wrap layer
    """
    def __init__(self, layer: Layer, prune_level: str = "weights", **kwargs):
        """
        Tensorflow Layer Wrapper that adds pruning functionality
        :param layer: Layer to be wrapped
        :param prune_level: Level at which pruning will occur
        :param kwargs: scope
        """
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
        """
        Initializes all tensorflow variables and builds layer for training and inference
        :param input_shape: Input shape of layer (typically we won't be calling this function directly)
        :return:
        """
        super(PruneWrapper, self).build(input_shape)
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

    def call(self, inputs, training=None, **kwargs):
        """
        Calls layer's call function
        """
        return self.layer.call(inputs)

    def get_filters(self):
        """
        Number of filters in layer
        :return: integer presenting number of filters in layer (0 if not convolutional)
        """
        if self.layer.__class__ == Conv2D:
            return self.layer.filters
        return 0

    def get_channels(self):
        """
        Number of Channels in layer
        :return: integer representing number of channels in layer
        """
        return self.get_weights()[0].shape[-2]

    def get_biases(self):
        """
        Biases of layer
        """
        return self.layer.get_weights()[1]

    def get_original_wandb(self):
        """
        Original weights and biases of layer before it was wrapped
        :return: original weights and biases of layer
        """
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
        """
        Returns tensor representation of mask
        :return: Tensor
        """
        return tf.convert_to_tensor(self.mask, dtype=self.mask.dtype, name=self.mask.name)

    def set_mask(self, new_mask_vals: np.ndarray):
        """
        Assign new values to the layer Wrapper's mask
        :param new_mask_vals: ndarray containing new values for layer's mask
        :return:
        """
        if new_mask_vals.shape != self.mask.shape:
            raise ValueError("new_mask_vals must be same size as mask")
        self.mask.assign(new_mask_vals)

    def reset_mask(self):
        """
        Resets mask to all ones (A.K.A. no weights are pruned)
        :return:
        """
        mask = self.mask
        new_mask = tf.ones_like(mask)
        self.set_mask(new_mask)

    def revert_layer(self):
        """
        Revert layer to original weights and biases, prior to being Wrapped
        :return:
        """
        self.reset_mask()
        layer = self.layer
        wandb = layer.get_weights()
        original_weights = self.get_original_weights()
        wandb[0] = original_weights
        layer.set_weights(wandb)
        self.layer = layer

    def prune(self):
        """
        Prune layer
        :return:
        """
        # Collect weights and mask of layer
        wandb = self.layer.get_weights()
        weights = wandb[0]
        mask = self.get_mask()

        # Multiply weights by mask to set pruned values to 0
        new_weights = weights * mask
        wandb[0] = new_weights
        self.layer.set_weights(wandb)
