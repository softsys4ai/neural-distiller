from pruning.prune_util import load_dataset, load_model, compile_model, train_model, save_model_h5, \
    evaluate_model_size, format_experiment_name, evaluate_percentage_of_zeros

from pruning.pruner import Pruner

import tensorflow as tf
import tensorflow_model_optimization as tfmot

import numpy as np

import datetime


def prune_mnist_pruner(test_size=500, **pruning_params):
    """
    :param test_size: Number of samples to be used for test set
    :keyword
        pruning_params:
        intial_epochs - Epochs to train model before pruning
        prune_schedule - PruningSchedule object to be used
        continue_epochs - Epochs to continue training after wrapping model
        fine_tune_epochs - Epochs to train after pruning model
    :return:
    """
    experiment_name = format_experiment_name("low_magnitude", "weight", "mnist", **pruning_params)
    initial_epochs = pruning_params.get("initial_epochs", 4)

    prune_method = "low_magnitude"
    prune_level = "weight"

    # Load model and dataset, and compile model
    model = load_model("mnist")
    compile_model(model)
    (X_train, Y_train), (X_test, Y_test) = load_dataset("mnist", test_size=test_size)

    # Train model with TensorBoard callback
    log_dir = f"./logs/{experiment_name}/unpruned"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
    )
    train_model(model, X_train, Y_train, X_test, Y_test,
                callbacks=[tensorboard_callback],
                epochs=initial_epochs, verbose=0)

    # Initialize Pruner
    log_dir = f"./logs/{experiment_name}/pruned"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
    )
    model_pruner = Pruner(model, X_train, Y_train, X_test, Y_test,
                          prune_level, prune_method)
    pruning_params["callbacks"] = [tensorboard_callback]
    model_pruner.prune(**pruning_params)

    pruned_model = model_pruner.pruned_model
    print(evaluate_percentage_of_zeros(pruned_model))


if __name__ == "__main__":
    """
    :param test_size: Number of samples to be used for test set 
    :keyword 
        pruning_params:
        intial_epoch - Epochs to train model before pruning
        prune_schedule - PruningSchedule object to be used
        continue_epochs - Epochs to continue training after wrapping model
        fine_tune_epochs - Epochs to train after pruning model 
    :return: 
    """
    # Test 1
    _initial_epochs = 10
    _continue_epochs = 5
    _fine_tune_epochs = 3

    begin_step = 0
    end_step = 4
    initial_sparsity = 0.5
    final_sparsity = 0.9
    _pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(begin_step=begin_step,
                                                             end_step=end_step,
                                                             initial_sparsity=initial_sparsity,
                                                             final_sparsity=final_sparsity)

    pruning_params = {
        "initial_epochs": _initial_epochs,
        "continue_epochs": _continue_epochs,
        "fine_tune_epochs": _fine_tune_epochs,
        "pruning_schedule": _pruning_schedule
    }
    prune_mnist_pruner(**pruning_params)
