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