import numpy as np
import keras
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy

nb_classes = 10
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

def knowledge_distillation_loss(y_true, y_pred, alpha):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    loss = alpha*logloss(y_true, y_pred) + logloss(y_true_softs, y_pred_softs)
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)

def calculate_weighted_score(logger, model, X_train, Y_train, X_test, Y_test):
    logger.info('Calculating weighted model score')
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    val_score = model.evaluate(X_test, Y_test, verbose=0)
    logger.info('Raw model scores: (train_acc, train_loss, val_acc, val_loss) --> (%s, %s, %s, %s)'
                % (train_score[1], train_score[0], val_score[1], val_score[0]))
    return ((train_score[0] + 3*val_score[0])/4), ((train_score[1] + 3*val_score[1])/4)

def calculate_score(logger, model, X_train, Y_train, X_test, Y_test):
    logger.info('Calculating model score')
    train_acc = model.evaluate(X_train, Y_train, verbose=0)
    val_acc = model.evaluate(X_test, Y_test, verbose=0)
    return ((train_acc[0] + val_acc[0])/2), ((train_acc[1] + val_acc[1])/2)