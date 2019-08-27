import keras
from keras.datasets import mnist
from keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.constraints import max_norm
from keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.models import Sequential
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
from HelperUtil import HelpfulFunctions

class StudentModel:
    def __init__(self):
        self.nb_classes = 10
        self.input_shape = (28, 28, 1) # Input shape of each image
        self.nb_filters = 64 # number of convolutional filters to use
        self.dropoutVal = 0.2
        self.student = None
        self.epochs = 4
        self.batchSize = 256
        self.alpha = 0.1
        self.temp = 7
    
    def printSummary(self):
        student.summary()

    def getModel(self):
        return student

    def buildAndCompile(self):
        input_shape = (28, 28, 1)
        student = Sequential()
        student.add(Flatten(input_shape=input_shape))
        student.add(Dense(32, activation='relu'))
        student.add(Dropout(dropoutVal))
        student.add(Dense(nb_classes))
        student.add(Activation('softmax'))
        #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        student.compile(
            #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
            optimizer='adadelta',
            loss=lambda y_true, y_pred: HelpfulFunctions.knowledge_distillation_loss(y_true, y_pred, alpha),
            #loss='categorical_crossentropy',
            metrics=[HelpfulFunctions.acc])

    def train(self, X_train, Y_train_new, X_test, Y_test_new):
        student.fit(X_train, Y_train_new,
            batch_size=batchSize,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test_new))