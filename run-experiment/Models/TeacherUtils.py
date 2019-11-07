# TODO createStudentTrainingData(model, X_train, Y_train, X_test, Y_test)
from tensorflow.python.keras.models import Model
import numpy as np
from Utils import HelperUtil

def find_largest_value(output_distribution):
    pos = 0
    max_val = output_distribution[pos]
    for i in range(1, len(output_distribution)):
        if output_distribution[i] > max_val:
            pos = i
            max_val = output_distribution[i]
    return max_val

def createStudentTrainingData(model, temp, X_train, Y_train, X_test, Y_test):
    teacher_WO_Softmax = Model(model.input, model.get_layer('logits').output) # TODO test this method of getting the last layer output
    teacher_WO_Softmax.summary()
    teacher_train_logits = teacher_WO_Softmax.predict(X_train)
    # directly retrieve the logits
    teacher_test_logits = teacher_WO_Softmax.predict(X_test)
    # softmax at raised temperature
    train_logits_T = teacher_train_logits / temp
    test_logits_T = teacher_test_logits / temp
    train_logits_T = abs(train_logits_T) # TODO remove
    test_logits_T = abs(test_logits_T) # TODO remove
    Y_train_soft = HelperUtil.softmax(train_logits_T)
    Y_test_soft = HelperUtil.softmax(test_logits_T)
    Y_train_soft, Y_test_soft = normalizeStudentSoftTargets(Y_train_soft, Y_test_soft)
    for i in range(0, len(Y_train_soft)):
        Y_train_soft[i] = (1/find_largest_value(Y_train_soft[i]))*Y_train_soft[i]
    for i in range(0, len(Y_test_soft)):
        Y_test_soft[i] = (1/find_largest_value(Y_test_soft[i]))*Y_test_soft[i]
    # Concatenate so that this becomes a (num_classes + num_classes) dimensional vector
    Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
    Y_test_new = np.concatenate([Y_test, Y_test_soft], axis=1)
    return Y_train_new, Y_test_new


def normalizeStudentSoftTargets(Y_train_soft, Y_test_soft):
    for i in range(len(Y_train_soft)):
        sum = 0
        for val in Y_train_soft[i]:
            sum += val
        Y_train_soft[i] = [x/sum for x in Y_train_soft[i]]
    for i in range(len(Y_test_soft)):
        sum = 0
        for val in Y_test_soft[i]:
            sum += val
        Y_test_soft[i] = [x/sum for x in Y_test_soft[i]]
    return Y_train_soft, Y_test_soft

