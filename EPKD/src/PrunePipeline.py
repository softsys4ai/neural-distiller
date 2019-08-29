import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from HelperUtil import HelpfulFunctions
helpful = HelpfulFunctions()


def prune(model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, initSparse, endSparse, logdir):
    print('[info] applying pruning techniques to the provided model')
    # TODO use all pruning techniques on the teacher model
    studentOne = sparsePrune(model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, initSparse, endSparse, logdir)
    return studentOne

def sparsePrune(model, X_train, Y_train, X_test, Y_test, num_train_samples, batch_size, epochs, initSparse, endSparse, logdir):
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
    # TODO determine how to limit this pruning to retain 90% of the network weights / size
    new_pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initSparse,
                                                     final_sparsity=endSparse,
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
    print('[INFO] Sparsely Pruned Network Results')
    print('[INFO] Test loss:', score[0])
    print('[INFO] Test accuracy:', score[1])
    return new_model

def L1RankPrune():
    print("[INFO] Using L1 norm of the weights in each filter to rank them")

def AbsDeltaCostPrune():
    print("[INFO] Using the absolute change to the cost of the network to rank the importance of each filter")

# TODO add more pruning techniques