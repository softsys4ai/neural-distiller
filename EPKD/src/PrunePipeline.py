import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from HelperUtil import HelpfulFunctions
helpful = HelpfulFunctions()


def prune(model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, logdir):
    print('[info] applying pruning techniques to the provided model')
    # TODO use all pruning techniques on the teacher model
    studentOne = sparsePrune(model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, logdir)
    return studentOne

def sparsePrune(model, X_train, Y_train, X_test, Y_test, num_train_samples, batch_size, epochs, logdir):
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
    # TODO determine how to limit this pruning to retain 90% of the network weights / size
    new_pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.01,
                                                     final_sparsity=0.1,
                                                     begin_step=0,
                                                     end_step=end_step,
                                                     frequency=100)
    }
    new_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
    new_model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])
    callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
    ]
    new_model.fit(X_train, Y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         callbacks=callbacks,
                         validation_data=(X_test, Y_test))

    score = new_model.evaluate(X_test, Y_test, verbose=0)
    print('Sparsely Pruned Network Results')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return new_model

# TODO add more pruning techniques