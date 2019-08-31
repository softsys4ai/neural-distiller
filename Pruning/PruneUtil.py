import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from Utils import HelperUtil

def prune(model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, initSparse, endSparse):
    print('[info] applying pruning techniques to the provided model')
    # TODO use all pruning techniques on the teacher model
    studentOne = sparsePrune(model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, initSparse, endSparse)
    return studentOne

# weight removal method for all layer types
def sparsePrune(logger, model, X_train, Y_train, X_test, Y_test, num_train_samples, batch_size, epochs, initSparse, endSparse):
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
        sparsity.PruningSummaries(log_dir=None, profile_batch=0)
    ]
    new_model.fit(X_train, Y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         callbacks=callbacks,
                         validation_data=(X_test, Y_test))

    score = new_model.evaluate(X_test, Y_test, verbose=0)
    logger.info('Sparsely Pruned Network Results')
    logger.info('Test loss:', score[0])
    logger.info('Test accuracy:', score[1])
    return new_model

# filter removal method for conv layers
def L1RankPrune(logger, model, layersToPrune):
    logger.info("Using L1 norm of the weights in each filter to rank them")
    # TODO use HelperUtil.find_layer_of_type() to find layers of interest for pruning

# filter removal method for conv layers
def AbsDeltaCostPrune(logger):
    logger.info('Using the absolute change to the cost of the network to rank the importance of each filter')

# filter removal method for conv layers
def TaylorPruning(logger):
    # https://arxiv.org/pdf/1906.10771v1.pdf
    logger.info("Filter importance estimation through highly efficient taylor expansion method")

# filter removal method for conv layers
def FilterAgentPrune(logger):
    logger.info("Using a CNN pruning agent to determine which filters to remove (filter weights) --> CNN Agent --> (binary removal decision)")

# TODO add more pruning techniques