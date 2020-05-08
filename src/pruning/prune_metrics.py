import tensorflow as tf

import numpy as np


def calculate_sparsity_of_model(model: tf.keras.models.Model):
    """
    calculates total sparsity of model
    :param model:
    :return: percentage_of_sparse_params, percentage_of_nonzero_params
    """
    total_params = model.count_params()
    total_zeros = 0
    for layer in model.layers:
        _, layer_zeros, _ = calculate_sparsity_of_layer(layer)
        total_zeros += layer_zeros
    return total_zeros / total_params


def calculate_sparsity_of_layer(layer: tf.keras.layers.Layer):
    layer_params = layer.count_params()
    if layer_params == 0:
        return 0
    weights = layer.get_weights()[0]
    layer_zeros = np.count_nonzero(weights == 0)
    return layer_zeros / layer_params, layer_zeros, layer_params


def calculate_sparsity_per_layer(model: tf.keras.models.Model):
    """
    Calculates weight sparsity for each layer in model
    :param model:
    :return: {layer_name: (total_layer_params, total_layer_nonzero_params, total_layer_zero_params,
                            percentage_of_sparse_params, percentage_of_nonzero_params)}
    """
    layer_metrics = {}
    for layer in model.layers:
        if layer.count_params() != 0:
            layer_params = layer.count_params()
            weights = layer.get_weights()[0]
            layer_zeros = np.count_nonzero(weights == 0)
            layer_metrics[layer.name] = layer_zeros / layer_params
    return layer_metrics


def calculate_filter_sparsity_per_layer(model: tf.keras.models.Model):
    """
    Returns filter sparsity for each Conv2D layer in model
    :param model:
    :return: {layer_name: (number_of_filters, number_of_sparse_filters, percentage_of_sparse_filters)}
    """
    conv_layer = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    layer_metrics = {}
    for layer in conv_layer:
        sparse_filters = 0
        weights = layer.get_weights()[0]
        layer_channels = weights.shape[-2]
        layer_filters = layer.filters * layer_channels
        for channel in range(layer_channels):
            for conv_filter_index in range(layer_filters):
                conv_filter = weights[:, :, channel, conv_filter_index]
                if np.all(conv_filter == 0):
                    sparse_filters += 1
        layer_metrics[layer.name] = (sparse_filters / layer_filters, sparse_filters, layer_filters)
    return layer_metrics


def calculate_filter_sparsity_of_model(model: tf.keras.models.Model):
    """
    Calculates filter sparsity for entire model
    :param model:
    :return: percentage of sparse filters in entire model
    """
    total_filters = 0
    sparse_filters = 0
    layer_metrics = calculate_filter_sparsity_per_layer(model)
    for (layer_name, (layer_sparsity, layer_sparse_filters, layer_filters)) in layer_metrics.items():
        total_filters += layer_filters
        sparse_filters += layer_sparse_filters
    return sparse_filters / total_filters, sparse_filters, total_filters

