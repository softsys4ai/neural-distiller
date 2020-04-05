from pruning.prune_util import load_dataset, load_model, compile_model, train_model, save_model_h5,\
    evaluate_model_size, format_experiment_name

import tensorflow as tf
import tensorflow_model_optimization as tfmot

import numpy as np


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
        pruned_evaluation += f"{final_model.weights[i].name} -- Total: {w.size}, Zeros:{np.sum(w==0)}\n"
    pruned_evaluation += "\n"
    pruned_evaluation += evaluate_model_size([final_model], uncompressed_path=pruned_keras_file)

    # TODO:// Fix to relative path
    with open(f"/Users/stephenbaione/Desktop/neural-distiller/src/pruning/experiments/{experiment_name}.txt", "w") as file:
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
    # Test 1
    test_id = 1
    _initial_epochs = 4
    _fine_tune_epochs = 2

    begin_step = 0
    end_step = 4
    initial_sparsity = 0.5
    final_sparsity = 0.9
    _pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(begin_step=begin_step,
                                                            end_step=end_step,
                                                            initial_sparsity=initial_sparsity,
                                                            final_sparsity=final_sparsity)

    pruning_params = {
        "test_id": test_id,
        "initial_epochs": _initial_epochs,
        "fine_tune_epochs": _fine_tune_epochs,
        "pruning_schedule": _pruning_schedule
    }
    prune_mnist(**pruning_params)

    # Test 2
    test_id = 2
    _initial_epochs = 8
    _fine_tune_epochs = 4

    begin_step = 0
    end_step = 4
    initial_sparsity = 0.5
    final_sparsity = 0.9
    _pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(begin_step=begin_step,
                                                             end_step=end_step,
                                                             initial_sparsity=initial_sparsity,
                                                             final_sparsity=final_sparsity)

    pruning_params = {
        "test_id": test_id,
        "initial_epochs": _initial_epochs,
        "fine_tune_epochs": _fine_tune_epochs,
        "pruning_schedule": _pruning_schedule
    }
    prune_mnist(**pruning_params)

    # Test 3
    test_id = 3
    _initial_epochs = 4
    _fine_tune_epochs = 2

    begin_step = 0
    end_step = 4
    initial_sparsity = 0.5
    final_sparsity = 0.95
    _pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(begin_step=begin_step,
                                                             end_step=end_step,
                                                             initial_sparsity=initial_sparsity,
                                                             final_sparsity=final_sparsity)

    pruning_params = {
        "test_id": test_id,
        "initial_epochs": _initial_epochs,
        "fine_tune_epochs": _fine_tune_epochs,
        "pruning_schedule": _pruning_schedule
    }
    prune_mnist(**pruning_params)

    # Test 4
    test_id = 4
    _initial_epochs = 8
    _fine_tune_epochs = 4

    begin_step = 0
    end_step = 4
    initial_sparsity = 0.5
    final_sparsity = 0.95
    _pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(begin_step=begin_step,
                                                             end_step=end_step,
                                                             initial_sparsity=initial_sparsity,
                                                             final_sparsity=final_sparsity)

    pruning_params = {
        "test_id": test_id,
        "initial_epochs": _initial_epochs,
        "fine_tune_epochs": _fine_tune_epochs,
        "pruning_schedule": _pruning_schedule
    }
    prune_mnist(**pruning_params)
