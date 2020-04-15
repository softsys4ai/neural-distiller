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
    original_evaluation = evaluate_percentage_of_zeros(model)
    original_size = evaluate_model_size(model)

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
    pruned_evaluation = evaluate_percentage_of_zeros(pruned_model)
    pruned_size = evaluate_model_size(pruned_model)

    print("\n-------- Original Model Sizes --------")
    print(original_size)
    print(original_evaluation)
    print("\n-------- Pruned Model Sizes --------")
    print(pruned_size)
    print(prune_level)


# TODO:// Implement into proper classes
def test_prune_taylor_first_order():
    model = load_model("mnist")
    (X_train, Y_train), (X_test, Y_test) = load_dataset(dataset="mnist", test_size=800)
    compile_model(model)
    train_model(model, X_train, Y_train, X_test, Y_test, epochs=8)
    print(evaluate_percentage_of_zeros(model))

    from pruning import prune_wrapper
    # Clone model with PruningWrapper
    def _wrap_model(layer):
        if isinstance(layer, tf.keras.layers.InputLayer) or layer == model.layers[-1] or len(layer.get_weights()) == 0:
            return layer.__class__.from_config(layer.get_config())
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return prune_wrapper.PruneWrapper(layer, prune_level="filter")
        return layer.__class__.from_config(layer.get_config())

    cloned_model = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_wrap_model)
    compile_model(cloned_model)
    train_model(cloned_model, X_train, Y_train, X_test, Y_test, epochs=4)

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
    print(evaluate_percentage_of_zeros(cloned_model))

    # TODO:// Strip pruning wrappers from model


# IGNORE THIS: Just for reference of how the whole process is run, without having to look between files
def prune_mnist(test_size=500, **pruning_params):
    test_id = pruning_params.get("id")
    experiment_name = format_experiment_name("low_magnitude", "weight", "mnist", **pruning_params)
    initial_epochs = pruning_params.get("initial_epochs", 4)
    fine_tune_epochs = pruning_params.get("fine_tune_epochs", 2)
    pruning_schedule = pruning_params.get("pruning_schedule")

    # Load, compile, and train model
    model = load_model("mnist")
    (X_train, Y_train), (X_test, Y_test) = load_dataset(dataset="mnist", test_size=500)
    compile_model(model)
    train_model(model, X_train, Y_train, X_test, Y_test, epochs=initial_epochs)

    # Initial Evaluation
    unpruned_score = model.evaluate(X_test, Y_test, verbose=0)
    unpruned_score = f"\nLoss: {unpruned_score[0]}\n" \
                     f"Validation: {unpruned_score[-1]}\n"
    unpruned_keras_file = save_model_h5(model, include_optimizer=False)
    initial_evaluation = ""
    for i, w in enumerate(model.get_weights()):
        initial_evaluation += f"{model.weights[i].name} -- Total: {w.size}, Zeros:{np.sum(w == 0)}\n"
    initial_evaluation += "\n"
    initial_evaluation += evaluate_model_size([model], uncompressed_path=unpruned_keras_file)
    initial_evaluation += "\n"

    # Wrap model for pruning
    new_pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
    new_pruned_model.summary()

    # Recompile and fit for specified epochs
    compile_model(new_pruned_model)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    new_pruned_model.fit(X_train, Y_train, epochs=initial_epochs, verbose=1,
                         callbacks=callbacks, validation_data=(X_test, Y_test))

    # Get score of prune wrapped model
    pruned_score = new_pruned_model.evaluate(X_test, Y_test, verbose=0)
    pruned_score = f"\nLoss: {pruned_score[0]}\n" \
                   f"Validation: {pruned_score[-1]}\n"
    # Save and restore model
    checkpoint_file = save_model_h5(new_pruned_model, include_optimizer=True)
    with tfmot.sparsity.keras.prune_scope():
        restored_model = tf.keras.models.load_model(checkpoint_file)

    # Fine tune restored model
    restored_model.fit(X_train, Y_train, epochs=fine_tune_epochs, verbose=1,
                       callbacks=callbacks, validation_data=(X_test, Y_test))
    pruned_fine_tune_score = restored_model.evaluate(X_test, Y_test, verbose=0)
    pruned_fine_tune_score = f"\nLoss: {pruned_fine_tune_score[0]}\n" \
                             f"Validation: {pruned_fine_tune_score[-1]}\n"

    # Strip wrapper
    final_model = tfmot.sparsity.keras.strip_pruning(restored_model)
    final_model.summary()

    # Save final model
    pruned_keras_file = save_model_h5(final_model, include_optimizer=False)

    # Evaluate pruning compression and zeros
    pruned_evaluation = "\n"
    for i, w in enumerate(final_model.get_weights()):
        pruned_evaluation += f"{final_model.weights[i].name} -- Total: {w.size}, Zeros:{np.sum(w == 0)}\n"
    pruned_evaluation += "\n"
    pruned_evaluation += evaluate_model_size([final_model], uncompressed_path=pruned_keras_file)

    # TODO:// Fix to relative path
    with open(f"/Users/stephenbaione/Desktop/neural-distiller/src/pruning/experiments/{experiment_name}.txt",
              "w") as file:
        file.write(
            "-------- Scores --------\n"
            f"unpruned_score: {unpruned_score}\n"
            f"pruned_score: {pruned_score}\n"
            f"pruned_fine_tune_score: {pruned_fine_tune_score}\n"
            "-------- Evaluatioins --------\n"
            f"Initial Evaluation:\n{initial_evaluation}"
            f"Pruned_evaluation:\n{pruned_evaluation}"
        )


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
