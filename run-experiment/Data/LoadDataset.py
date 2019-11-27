from keras.datasets import mnist, cifar10, cifar100
from tensorflow.python.keras.utils import np_utils
import numpy as np



def load_mnist(logger):
    # preparing the MNIST dataset for training teacher and student models
    nb_classes = 10
    input_shape = (28, 28, 1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # convert y_train and y_test to categorical binary values
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    # convert to float32 type
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # Normalize the values
    # if using CIFAR-100 or CIFAR-10, do not need division b/c already done
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    return X_train, Y_train, X_test, Y_test

def load_cifar_10(logger):
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, Y_train, X_test, Y_test

def load_cifar_100(logger):
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

def load_preprocessed_cifar100(logger):
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    del X_train
    del y_train
    X_train = np.load("/local/second-neur-dist/neural-distiller/pre-experiment/preprocess/data/x_train_60.npy")
    y_train = np.load("/local/second-neur-dist/neural-distiller/pre-experiment/preprocess/data/y_train_60.npy")
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, Y_train, X_test, Y_test

def load_dataset_by_name(logger, datasetname):
    if datasetname is "mnist":
       return load_mnist(logger)
    elif datasetname is "cifar10":
        return load_cifar_10(logger)
    elif datasetname is "cifar100":
        return load_cifar_100(logger)
    elif datasetname is "cifar100-static-transform":
        # dataset of preprocessed aka transformed cifar100 images
        return load_preprocessed_cifar100(logger)
    else:
        logger.error("provided dataset name is not supported!")