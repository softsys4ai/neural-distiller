import numpy as np
from Configuration import Config as cfg
from tensorflow.python.keras.losses import categorical_crossentropy as logloss
from tensorflow.python.keras.losses import KLDivergence as KL
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.python.keras.layers import Lambda, concatenate, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import kullback_leibler_divergence
import tensorflow as tf

def apply_knowledge_distillation_modifications(logger, model, temp):
    # modifying student network for KD
    model.layers.pop()
    logits = model.get_layer('logits').output
    probs = Activation('softmax')(logits)
    logits_T = Lambda(lambda x: x / temp)(logits)
    probs_T = Activation('softmax')(logits_T)
    output = concatenate([probs, probs_T])
    model = Model(model.input, output)  # modified student model
    return model


def revert_knowledge_distillation_modifications(logger, model):
    logits = model.get_layer('logits').output
    output = Activation('softmax')(logits)
    model2 = Model(model.input, output)  # reverted student model
    return model2


def softmax(x):
    return np.exp(x) / (np.exp(x).sum())


def knowledge_distillation_loss(logger, y_true, y_pred, alpha=cfg.alpha):
    logger.info("compiling and training student with alpha: %s" % alpha)
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[:, :cfg.dataset_num_classes], y_true[:, cfg.dataset_num_classes:]
    y_pred, y_pred_softs = y_pred[:, :cfg.dataset_num_classes], y_pred[:, cfg.dataset_num_classes:]
    # loss = (1-alpha)*logloss(y_true, y_pred) + alpha*logloss(y_true_softs, y_pred_softs)
    loss = logloss(y_true, y_pred) + alpha * kullback_leibler_divergence(y_true_softs, y_pred_softs)
    return loss

def knowledge_distillation_loss_KL(logger, y_true, y_pred, alpha=cfg.alpha):
    logger.info("compiling and training student with alpha: %s" % alpha)
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[:, :cfg.dataset_num_classes], y_true[:, cfg.dataset_num_classes:]
    y_pred, y_pred_softs = y_pred[:, :cfg.dataset_num_classes], y_pred[:, cfg.dataset_num_classes:]
    # loss = (1-alpha)*logloss(y_true, y_pred) + alpha*logloss(y_true_softs, y_pred_softs)
    loss = logloss(y_true, y_pred) + alpha * KL(y_true_softs, y_pred_softs)
    return loss


def custom_logloss_loss(y_true, y_pred):
    loss = logloss(y_true, y_pred)
    return loss


# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :cfg.dataset_num_classes]
    y_pred = y_pred[:, :cfg.dataset_num_classes]
    return categorical_accuracy(y_true, y_pred)

def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return top_k_categorical_accuracy(y_true, y_pred)

# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):
    logits = y_true[:, 256:]
    y_soft = softmax(logits/temperature)
    y_pred_soft = y_pred[:, 256:]
    return logloss(y_soft, y_pred_soft)

def calculate_unweighted_score(logger, model, X_train, Y_train, X_test, Y_test):
    # with tf.Graph().as_default():
    model.compile(optimizer=cfg.student_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    val_score = model.evaluate(X_test, Y_test, verbose=0)
    del model
    return train_score, val_score


def calculate_weighted_score(logger, model, X_train, Y_train, X_test, Y_test):
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    val_score = model.evaluate(X_test, Y_test, verbose=0)
    logger.info("Model training results (raw): %s, %s" % (train_score, val_score))
    return ((train_score[0] + 3 * val_score[0]) / 4), ((train_score[1] + 3 * val_score[1]) / 4)


def calculate_score(logger, model, X_train, Y_train, X_test, Y_test):
    train_acc = model.evaluate(X_train, Y_train, verbose=0)
    val_acc = model.evaluate(X_test, Y_test, verbose=0)
    return ((train_acc[0] + val_acc[0]) / 2), ((train_acc[1] + val_acc[1]) / 2)


def find_layers_of_type(logger, model, layertype):
    layerNames = [layer.name for layer in model.layers]
    convLayers = []
    for i in range(len(layerNames)):
        if layertype in layerNames[i]:
            convLayers.append(i)
    return convLayers


def find_trainable_layers(logger, model):
    modelWeights = [x.get_weights() for x in model.layers]
    trainableLayers = []
    for i in range(len(modelWeights)):
        if len(modelWeights[i]) != 0:
            trainableLayers.append(i)
    return trainableLayers


# def WriteDictToCSV(csv_file,csv_columns,dict_data):
#     try:
#         with open(csv_file, 'w') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#             writer.writeheader()
#             for data in dict_data:
#                 writer.writerow(data)
#     except IOError as (errnum, strerror):
#             print("I/O error({0}): {1}".format(errnum, strerror))
#     return
