import tensorflow as tf

from tensorflow.python.keras import backend as K

from pruning.prune_wrapper import PruneWrapper


def _get_prunable_layers(model, wrapper_class):
    prunable_layers = []
    for layer in model.layers:
        if isinstance(layer, wrapper_class):
            prunable_layers.append(layer)
    return prunable_layers


class MaintainSparsity(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MaintainSparsity, self).__init__()
        self.prunable_layers = []

    def on_train_begin(self, logs=None):
        self.step = K.get_value(self.model.optimizer.iterations)

    def on_epoch_end(self, epoch, logs=None):
        self.prunable_layers = _get_prunable_layers(self.model, PruneWrapper)
        for layer in self.prunable_layers:
            if isinstance(layer, PruneWrapper):
                layer.update_weights_op()


