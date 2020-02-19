import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from utils import helper_util


def prune(logger, model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, initSparse, endSparse):
    print('[info] applying pruning techniques to the provided model')
    student_nets = []
    student_nets.append(sparsePrune(logger, model, X_train, Y_train, X_test, Y_test, numTrainingSamples, batchSize, epochs, initSparse, endSparse))
    student_nets.append(L1RankPrune(logger, model, HelperUtil.find_layers_of_type(logger, model, "conv"), X_train, Y_train, X_test, Y_test, batchSize, epochs))
    # TODO use all pruning techniques on the teacher model
    return student_nets

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
    logger.info('Sparsely Pruned Network Experiment-Results')
    logger.info('Test loss:', score[0])
    logger.info('Test accuracy:', score[1])
    return new_model


def rank_filters_l1(logger, model):
    logger.info("Ranking filters by L1 norm")
    layersofinterest = HelperUtil.find_layers_of_type(logger, model, "conv")
    conv_layers_weights = [model.layers[layer].get_weights()[0] for layer in layersofinterest]
    rankedFiltersPerLayer = [] * len(layersofinterest)
    # rank filters in eah layer
    for i in range(len(conv_layers_weights)):
        weight = conv_layers_weights[i]
        weights_dict = {}
        num_filters = len(weight[0,0,0,:])
        for j in range(num_filters):
            w_s = np.sum(abs(weight[:,:,:,j]))
            filt ='{}'.format(j)
            weights_dict[filt]=w_s
        # sorting the weights
        weights_dict_sort=sorted(weights_dict.items(), key=lambda kv: kv[1])
        filter_num = []
        weights_value=[]
        for elem in weights_dict_sort:
            filter_num.append(elem[0])
            weights_value.append(elem[1])
        layer_rankings = zip(filter_num, weights_value)
        rankedFiltersPerLayer.append(layer_rankings)
    return rankedFiltersPerLayer

# filter removal method for conv layers
def L1RankPrune(logger, model, layersToPrune, X_train, Y_train, X_test, Y_test, batch_size, epochs):
    logger.info("L1RankPrune: Using L1 norm of the weights in each filter to rank them")
    rankedFilters = rank_filters_l1(logger, model)

    # TODO remove 10% of global filters

# integrate L1 rank pruning with keras surgeon
def KerasSurgeonExample(logger, teacher, X_train, Y_train, X_test, Y_test):
    from tfkerassurgeon.operations import delete_channels
    layersofinterest = HelperUtil.find_layers_of_type(logger, teacher, "conv")
    layer_0 = teacher.layers[0]
    model_new = delete_channels(teacher, layer_0, [4,15,6,26,9,24,21,17,3,23,14,10,13])
    layer_1 = model_new.layers[1]
    model_new_two = delete_channels(model_new, layer_1, [11,7,37,13,48,56,41,23,16,35])
    model_new_two.compile(loss='categorical_crossentropy',
                      optimizer=cfg.student_optimizer,
                      metrics=['accuracy'])
    prunedLoss, prunedAcc = HelperUtil.calculate_weighted_score(logger, model_new_two, X_train, Y_train, X_test, Y_test)
    logger.info('Teacher weighted score: (acc, loss) --> (%s, %s)' % (prunedLoss, prunedAcc))

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