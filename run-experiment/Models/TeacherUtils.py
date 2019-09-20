# TODO createStudentTrainingData(model, X_train, Y_train, X_test, Y_test)
from tensorflow.python.keras.models import Model
import numpy as np
from Utils import HelperUtil


def createStudentTrainingData(model, temp, X_train, Y_train, X_test, Y_test):
    teacher_WO_Softmax = Model(model.input, model.get_layer('logits').output) # TODO test this method of getting the last layer output
    teacher_train_logits = teacher_WO_Softmax.predict(X_train)
    # directly retrieve the logits
    teacher_test_logits = teacher_WO_Softmax.predict(X_test)
    # softmax at raised temperature
    train_logits_T = teacher_train_logits / temp
    test_logits_T = teacher_test_logits / temp
    Y_train_soft = HelperUtil.softmax(train_logits_T)
    Y_test_soft = HelperUtil.softmax(test_logits_T)
    # Concatenate so that this becomes a (num_classes + num_classes) dimensional vector
    Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
    Y_test_new = np.concatenate([Y_test, Y_test_soft], axis=1)
    return Y_train_new, Y_test_new