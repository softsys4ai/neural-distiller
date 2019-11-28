import pickle
from keras import backend as K
from keras.datasets import cifar100
import numpy as np
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt 

def convert_logits_to_soft_targets(temp, normalize, teacher_train_logits, teacher_test_logits, Y_train, Y_test):
    # softmax at raised temperature
    train_logits_T = teacher_train_logits / temp
    test_logits_T = teacher_test_logits / temp
    Y_train_soft = K.softmax(train_logits_T)
    Y_test_soft = K.softmax(test_logits_T)
    sess = K.get_session()
    Y_train_soft = sess.run(Y_train_soft)
    Y_test_soft = sess.run(Y_test_soft)

    if normalize is True:
        Y_train_soft, Y_test_soft = normalizeStudentSoftTargets(Y_train_soft, Y_test_soft)
        for i in range(0, len(Y_train_soft)):
            Y_train_soft[i] = (1 / find_largest_value(Y_train_soft[i])) * Y_train_soft[i]
        for i in range(0, len(Y_test_soft)):
            Y_test_soft[i] = (1 / find_largest_value(Y_test_soft[i])) * Y_test_soft[i]

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

def find_largest_value(output_distribution):
    pos = 0
    max_val = output_distribution[pos]
    for i in range(1, len(output_distribution)):
        if output_distribution[i] > max_val:
            pos = i
            max_val = output_distribution[i]
    return max_val

def load_cifar_100():
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    # X_train = X_train.reshape(50000, 32, 32, 3)
    # X_test = X_test.reshape(10000, 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, Y_train, X_test, Y_test

# load dataset
X_train, Y_train, X_test, Y_test = load_cifar_100()

# load teacher logits
filehandler = open("cifar100_10_soft_targets.pkl", 'rb')
teacher_train_logits = pickle.load(filehandler)
teacher_test_logits = pickle.load(filehandler)


Y_train_new_1, Y_test_new_1 = convert_logits_to_soft_targets(1, True, teacher_train_logits, teacher_test_logits, Y_train, Y_test)
Y_train_new_2, Y_test_new_2 = convert_logits_to_soft_targets(2, True, teacher_train_logits, teacher_test_logits, Y_train, Y_test)
Y_train_new_5, Y_test_new_5 = convert_logits_to_soft_targets(5, True, teacher_train_logits, teacher_test_logits, Y_train, Y_test)
Y_train_new_10, Y_test_new_10 = convert_logits_to_soft_targets(10, True, teacher_train_logits, teacher_test_logits, Y_train, Y_test)
Y_train_new_20, Y_test_new_20 = convert_logits_to_soft_targets(20, True, teacher_train_logits, teacher_test_logits, Y_train, Y_test)

x = np.arange(100)
#y_one_hot = Y_train_new_1[0][:100]
y_1 = Y_train_new_1[0][100:]
y_2 = Y_train_new_2[0][100:]
y_5 = Y_train_new_5[0][100:]
y_10 = Y_train_new_10[0][100:]
y_20 = Y_train_new_20[0][100:]
#plt.plot(x, y_one_hot)
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_5)
plt.plot(x, y_10)
plt.plot(x, y_20)
#plt.legend(['one hot label', 'temperature 1', 'temperature 2', 'temperature 5', 'temperature 10', 'temperature 20'], loc='upper left')
plt.legend(['temperature 1', 'temperature 2', 'temperature 5', 'temperature 10', 'temperature 20'], loc='upper left')  
plt.title('Size 10 Teacher Output Distribution')  
plt.show()

