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

# teacher model class
class TeacherModel:
    # TODO add CL arguments to change the teacher's hyperparameters
    def __init__(self):
        self.nb_classes = 10
        self.input_shape = (28, 28, 1) # Input shape of each image
        self.nb_filters = 64 # number of convolutional filters to use
        self.pool_size = (2, 2) # size of pooling area for max pooling
        self.kernel_size = (3, 3) # convolution kernel size
        self.teacher = None
        self.teacher_WO_Softmax = None
        self.epochs = 4
        self.batch_size = 256
        self.temp = 7
	
    def printSummary(self):
        teacher.summary()

    def getModel(self):
        return teacher

    def createTeacherWOSoftmax(self):
        teacher_WO_Softmax = Model(teacher.input, teacher.get_layer('dense_6').output)

    def buildAndCompile(self):
        nb_classes = 10
        input_shape = (28, 28, 1)
        teacher = Sequential()
        teacher.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        teacher.add(Conv2D(64, (3, 3), activation='relu'))
        teacher.add(MaxPooling2D(pool_size=(2, 2)))

        teacher.add(Dropout(0.25)) # For reguralization

        teacher.add(Flatten())
        teacher.add(Dense(128, activation='relu'))
        teacher.add(Dropout(0.5)) # For reguralization

        teacher.add(Dense(nb_classes))
        teacher.add(Activation('softmax')) # Note that we add a normal softmax layer to begin with

        teacher.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
        self.createTeacherWOSoftmax()
        return t
        
    def train(self, model, X_train, Y_train, X_test, Y_test):
        teacher.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

    def createStudentTrainingData(self, X_train, Y_train, X_test, Y_test):
        teacher_train_logits = teacher_WO_Softmax.predict(X_train)
        teacher_test_logits = teacher_WO_Softmax.predict(X_test) # This model directly gives the logits ( see the teacher_WO_softmax model above)
        # Perform a manual softmax at raised temperature
        train_logits_T = teacher_train_logits/temp
        test_logits_T = teacher_test_logits / temp 
        Y_train_soft = HelpfulFunctions.softmax(train_logits_T)
        Y_test_soft = HelpfulFunctions.softmax(test_logits_T)
        # Concatenate so that this becomes a 10 + 10 dimensional vector
        Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
        Y_test_new =  np.concatenate([Y_test, Y_test_soft], axis =1)
        return Y_train_new, Y_test_new
