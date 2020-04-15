"""
@author: Stephen Baione
@description: Test pruning classes and functionality
# TODO:// After POF, start using VGG16 and ResNet50 for pruning
"""

from pruning.pruner import Pruner
from pruning import ranker
from pruning import prune_util

import tensorflow as tf


def test_prune_taylor_first_order_impl():
    model = prune_util.load_model("mnist")
    (X_train, Y_train), (X_test, Y_test) = prune_util.load_dataset(dataset="mnist", test_size=800)
    prune_util.compile_model(model)
    prune_util.train_model(model, X_train, Y_train, X_test, Y_test, epochs=8)
    print(prune_util.evaluate_percentage_of_zeros(model))

    pruner = Pruner(model, X_train, Y_train, X_test, Y_test,
                    prune_level="filter", prune_method="taylor_first_order")
    prune_params = {
        "continue_epochs": 4,
        "fine_tune_epochs": 6,
        "prune_n_lowest": 20
    }
    pruner.prune(**prune_params)
    pruned_model = pruner.pruned_model
    print(prune_util.evaluate_percentage_of_zeros(pruned_model))


# TODO:// Implement into proper classes
# TODO:// Implement Prune Schedule
def test_prune_taylor_first_order():
    model = prune_util.load_model("mnist")
    (X_train, Y_train), (X_test, Y_test) = prune_util.load_dataset(dataset="mnist", test_size=800)
    prune_util.compile_model(model)
    prune_util.train_model(model, X_train, Y_train, X_test, Y_test, epochs=8)
    print(prune_util.evaluate_percentage_of_zeros(model))

    from pruning import prune_wrapper
    # Clone model with PruningWrapper
    def _wrap_model(layer):
        if isinstance(layer, tf.keras.layers.InputLayer) or layer == model.layers[-1] or len(layer.get_weights()) == 0:
            return layer.__class__.from_config(layer.get_config())
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return prune_wrapper.PruneWrapper(layer, prune_level="filter")
        return layer.__class__.from_config(layer.get_config())

    cloned_model = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_wrap_model)
    prune_util.compile_model(cloned_model)
    prune_util.train_model(cloned_model, X_train, Y_train, X_test, Y_test, epochs=4)

    def _rank_taylor_first_order(_model):
        ranks = {}
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        inputs = X_test[:100]
        labels = Y_test[:100]

        for index, layer in enumerate(_model.layers):
            if isinstance(layer, prune_wrapper.PruneWrapper) and not isinstance(layer, tf.keras.layers.InputLayer):
                if isinstance(layer.layer, tf.keras.layers.Conv2D):
                    filters = range(layer.get_filters())
                    channels = range(layer.get_channels())
                    mask = layer.get_mask()

                    for channel in channels:
                        for filter_index in filters:
                            print(index, layer.name, channel, filter_index)
                            new_mask_vals = mask.numpy()
                            new_mask_vals[:, :, channel, filter_index] = 0.0
                            layer.set_mask(new_mask_vals)
                            layer.prune()

                            with tf.GradientTape() as tape:
                                tape.watch(layer.layer.output)
                                predictions = _model(inputs)
                                loss = loss_obj(labels, predictions)
                            activation = tape.watched_variables()[0]
                            grads = tape.gradient(loss, activation)
                            score = tf.math.reduce_mean(grads * activation).numpy()
                            ranks[score] = (index, channel, filter_index)
                            layer.reset_mask()
                            wandb = layer.get_weights()
                            og_weights = layer.get_original_weights()
                            wandb[0] = og_weights
                            layer.set_weights(wandb)

        return sorted(ranks.items())

    ranks = _rank_taylor_first_order(cloned_model)

    # Prune n lowest
    lowest_n_ranks = ranks[:10]
    for (score, (layer, channel, filter_index)) in lowest_n_ranks:
        mask = cloned_model.layers[layer].get_mask()
        new_mask_vals = mask.numpy()
        new_mask_vals[:, :, channel, filter_index] = 0.0
        cloned_model.layers[layer].set_mask(new_mask_vals)
        cloned_model.layers[layer].prune()
    print(prune_util.evaluate_percentage_of_zeros(cloned_model))

    # Strip prune wrappers from model
    def _strip_wrappers(layer):
        if isinstance(layer, prune_wrapper.PruneWrapper):
            return layer.layer
        return layer
    stripped_model = tf.keras.models.clone_model(cloned_model,
                                                 input_tensors=cloned_model.inputs,
                                                 clone_function=_strip_wrappers)
    print(prune_util.evaluate_percentage_of_zeros(stripped_model))


if __name__ == "__main__":
    test_prune_taylor_first_order_impl()
