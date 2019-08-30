from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils

def load_mnist():
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
    X_train /= 255
    X_test /= 255
    return X_train, Y_train, X_test, Y_test