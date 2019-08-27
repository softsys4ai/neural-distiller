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
from keras.models import model_from_json
from HelperUtil2 import acc
from HelperUtil2 import knowledge_distillation_loss
import datetime

class StudentModel:
    def __init__(self):
        self.nb_classes = 10
        self.input_shape = (28, 28, 1) # Input shape of each image
        self.nb_filters = 64 # number of convolutional filters to use
        self.dropoutVal = 0.2
        self.student = Sequential()
        self.epochs = 4
        self.batchSize = 256
        self.alpha = 0.1
        self.temp = 50
        self.name = "StudentDense"
    
    def printSummary(self):
        self.student.summary()

    def getModel(self):
        return self.student

    def buildAndCompile(self):
        self.student.add(Flatten(input_shape=self.input_shape))
        self.student.add(Dense(32, activation='relu'))
        self.student.add(Dropout(self.dropoutVal))
        self.student.add(Dense(self.nb_classes))
        self.student.add(Activation('softmax'))
        # Remove the softmax layer from the student network
        self.student.layers.pop()
        logits = self.student.layers[-1].output
        probs = Activation('softmax')(logits)
        # softed probabilities at raised temperature
        logits_T = Lambda(lambda x: x / self.temp)(logits)
        probs_T = Activation('softmax')(logits_T)
        output = concatenate([probs, probs_T])
        self.student = Model(self.student.input, output) # final student model
        #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.student.compile(
            #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
            optimizer='adadelta',
            loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, self.alpha),
            #loss='categorical_crossentropy',
            metrics=[acc])

    def train(self, X_train, Y_train_new, X_test, Y_test_new):
        self.student.fit(X_train, Y_train_new,
            batch_size=self.batchSize,
            epochs=self.epochs,
            verbose=1,
            validation_data=(X_test, Y_test_new))

    def save(self):
        now = datetime.datetime.now()
        # serialize model to JSON
        model_json = self.student.to_json()
        with open("{}.json".format(now.strftime("%Y-%m-%d_%H:%M:%S")), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.student.save_weights("{}_{}.h5".format(now.strftime(self.name, "%Y-%m-%d_%H:%M:%S")))
        print("[INFO] Saved model to disk")
        
        # later...

    def load(self, model_filename, weights_filename):  
        # load json and create model
        json_file = open(model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_filename)
        print("[INFO] Loaded model from disk")
        
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))