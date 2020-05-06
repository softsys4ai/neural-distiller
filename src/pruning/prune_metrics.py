import tensorflow as tf

import numpy as np


def calculate_sparsity_of_model(model: tf.keras.models.Model):
    """
    calculates total sparsity of model
    :param model:
    :return: percentage_of_sparse_params, percentage_of_nonzero_params
    """
    total_params = model.count_params()
    total_nonzeros = 0
    total_zeros = 0
    for layer in model.layers:
        if layer.count_params() != 0:
            layer_params = layer.count_params()
            weights = layer.get_weights()[0]
            total_nonzeros += np.count_nonzero(weights)
            total_zeros += np.count_nonzero(weights == 0)
    return (total_zeros / total_params), (total_nonzeros / total_params)


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
            layer_nonzeros = np.count_nonzero(weights)
            layer_zeros = np.count_nonzero(weights == 0)
            layer_metrics[layer.name] = (layer_params,
                                         layer_nonzeros,
                                         layer_zeros,
                                         layer_zeros / layer_params,
                                         layer_nonzeros / layer_params)
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
        layer_filters = layer.filters
        sparse_filters = 0
        weights = layer.get_weights()[0]
        for conv_filter_index in range(layer_filters):
            conv_filter = weights[:, :, :, conv_filter_index]
            if np.all(conv_filter == 0):
                sparse_filters += 1
        layer_metrics[layer.name] = (layer_filters, sparse_filters, sparse_filters / layer_filters)

    return layer_metrics


def calculate_filter_sparsity_of_model(model: tf.keras.models.Model):
    """
    Calculates filter sparsity for entire model
    :param model:
    :return: percentage of sparse filters in entire model
    """
    conv_layer = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    total_filters = 0
    sparse_filters = 0
    for layer in conv_layer:
        weights = layer.get_weights()[0]
        conv_filters = layer.filters
        channels = weights.shape[-2]
        total_filters += (channels * conv_filters)
        for channel in range(channels):
            for conv_filter in range(conv_filters):
                filter_vals = weights[:, :, channel, conv_filter]
                if np.all(filter_vals == 0):
                    sparse_filters += 1
    return sparse_filters / total_filters

