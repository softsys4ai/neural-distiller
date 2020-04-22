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

from pruning.prune_config import get_supported_prune_levels, get_supported_prune_methods
from pruning.prune_wrapper import PruneWrapper
from pruning.ranker import Ranker
from pruning import prune_util

import tensorflow as tf

from tensorflow.python.keras.models import Model

import tensorflow_model_optimization as tfmot


class Pruner(object):
    def __init__(self, model: Model, X_train, Y_train, X_test, Y_test, prune_level: str, prune_method: str):
        """
        Class to implement pruning of Sequential or Functional models
        :param model: Model to be pruned
        :param X_train: Training examples
        :param Y_train: Training labels
        :param X_test: Test examples
        :param Y_test: Test labels
        :param prune_level: Level to prune
        :param prune_method: Method to use while pruning
        """
        assert model is not None
        self.model = model

        if prune_level not in get_supported_prune_levels():
            raise ValueError("Unsupported prune_level")

        self.prune_level = prune_level
        self.prune_method = prune_method
        # Set pruned model as copy of model, with PruningWrapper applied
        self.pruned_model = self.copy_model(model)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def copy_model(self, model):
        """
        Function used to create clone of keras model with PruningWrappers applied to appropriate layers
        :param model: Keras Model
        :return: Keras model, with appropriate layers wrapped
        #TODO:// Extend for more options to fit new pruning algorithms
        """
        def _wrap_model(layer):
            # If layer is input layer, output layer, or layer that has no weights
            if (isinstance(layer, tf.keras.layers.InputLayer)) \
                    or (layer == model.layers[-1]) \
                    or (len(layer.get_weights()) == 0):
                return layer.__class__.from_config(layer.get_config())
            elif self.prune_level == "filters" and isinstance(layer, tf.keras.layers.Conv2D):
                return PruneWrapper(layer)
            return PruneWrapper(layer)
        return tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_wrap_model)

    # sparsity is percentage of network params that should remain after prune
    def prune(self, **kwargs):
        """
        Executes pruning method on model
        :param kwargs:

            Kwargs will vary depending on prune method chosen.
            TODO:// Reference config for supported methods and corresponding keyword arguments
        :return:
        """

        model = self.get_model()
        assert model is not None
        pruned_model = self.pruned_model

        (X_train, Y_train), (X_test, Y_test) = (self.X_train, self.Y_train), (self.X_test, self.Y_test)
        for data in [X_train, Y_train, X_test, Y_test]:
            assert data is not None

        prune_method = self.get_prune_method()
        if prune_method == "low_magnitude":
            pruned_model = self._prune_low_magnitude(model, X_train, Y_train, X_test, Y_test, **kwargs)
        elif prune_method == "taylor_first_order":
            pruned_model = self._prune_taylor_first_order(**kwargs)

        self.pruned_model = pruned_model

    def evaluate_pruned_model(self, verbose=0):
        """
        Evaluate pruned model
        :param verbose: Verbose for evaluation
        :return: Score of model evaluation
        """
        pruned_model = self.pruned_model
        X_test = self.X_test
        Y_test = self.Y_test
        return pruned_model.evaluate(X_test, Y_test, verbose=verbose)

    def _prune_taylor_first_order(self, **kwargs):
        """
        Implementation of Global Filter Taylor First Order Pruning
        :param kwargs:
                    continue_epochs: Epochs to train model after it is wrapped, before pruning
                    prune_n_lowest: Number of weights to prune from model TODO:// Create function for this
        :return:
        """
        def _wrap_model(layer):
            if isinstance(layer, tf.keras.layers.InputLayer) or layer == model.layers[-1] or len(layer.get_weights()) == 0:
                return layer.__class__.from_config(layer.get_config())
            elif isinstance(layer, tf.keras.layers.Conv2D):
                return PruneWrapper(layer, prune_level="filter")
            return layer.__class__.from_config(layer.get_config())

        continue_epochs = kwargs.get("continue_epochs", 5)
        prune_n_lowest = kwargs.get("prune_n_lowest", 10)

        model = self.model
        (X_train, Y_train), (X_test, Y_test) = (self.X_train, self.Y_train), (self.X_test, self.Y_test)

        # Create clone of model, with PruningWrapper applied to relevant layers
        wrapped_model = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_wrap_model)
        prune_util.compile_model(wrapped_model)

        # Train model for continue_epochs after wrappers have been applied
        callbacks = kwargs.get("callbacks", None)
        if callbacks is None:
            prune_util.train_model(wrapped_model, X_train, Y_train, X_test, Y_test,
                                   epochs=continue_epochs)
        else:
            prune_util.train_model(wrapped_model, X_train, Y_train, X_test, Y_test,
                                   epochs=continue_epochs, callbacks=callbacks)

        # Rank filters of model based of taylor first order criteria
        model_ranker = Ranker(wrapped_model)
        sorted_ranks = model_ranker.rank_taylor_first_order(X_test, Y_test, "filter")

        # Collect lowest n ranked filters
        lowest_n_ranks = sorted_ranks[:prune_n_lowest]

        # Prune each of the lowest n ranked filters
        for (score, (layer, channel, filter_index)) in lowest_n_ranks:
            mask = wrapped_model.layers[layer].get_mask()
            new_mask_vals = mask.numpy()
            new_mask_vals[:, :, channel, filter_index] = 0.0
            wrapped_model.layers[layer].set_mask(new_mask_vals)
            wrapped_model.layers[layer].prune()

        # Strip PruningWrappers from model
        stripped_model = tf.keras.models.clone_model(wrapped_model,
                                                     input_tensors=wrapped_model.inputs,
                                                     clone_function=self._strip_wrappers)
        prune_util.compile_model(stripped_model)

        return stripped_model

    @staticmethod
    def _strip_wrappers(layer):
        if isinstance(layer, PruneWrapper):
            return layer.layer
        return layer

    @staticmethod
    def _prune_low_magnitude(_model, _X_train, _Y_train, _X_test, _Y_test, **kwargs):

        prune_schedule = kwargs.get("pruning_schedule")
        continue_epochs = kwargs.get("continue_epochs", 4)
        fine_tune_epochs = kwargs.get("fine_tune_epochs", 2)

        callbacks = kwargs.get("callbacks", [])
        checkpoint_path = kwargs.get("checkpoint_path", None)

        # Wrap model for pruning
        new_pruned_model: Model = tfmot.sparsity.keras.prune_low_magnitude(_model, prune_schedule)
        new_pruned_model.summary()

        # Recompile and fit
        prune_util.compile_model(new_pruned_model)

        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
        prune_util.train_model(new_pruned_model, X_train=_X_train, Y_train=_Y_train,
                               X_test=_X_test, Y_test=_Y_test,
                               callbacks=callbacks, epochs=continue_epochs, verbose=0)
        
        # save and restore model
        checkpoint_file = prune_util.save_model_h5(new_pruned_model, file_path=checkpoint_path, include_optimizer=True)
        with tfmot.sparsity.keras.prune_scope():
            restored_model = tf.keras.models.load_model(checkpoint_file)
            
        # Fine tune restored model
        prune_util.train_model(restored_model, X_train=_X_train,Y_train=_Y_train,
                               X_test=_X_test,Y_test=_Y_test,
                               callbacks=callbacks, epochs=fine_tune_epochs, verbose=0)
        
        # Save final model
        final_model = tfmot.sparsity.keras.strip_pruning(restored_model)
        final_model.summary()
    
        return final_model

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
