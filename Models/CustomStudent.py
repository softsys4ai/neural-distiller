import os
from tensorflow.python.keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate, MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from tensorflow.python.keras.models import Model, Sequential, model_from_json
from tensorflow.python.keras.optimizers import adadelta
from Utils.HelperUtil import acc
from Utils.HelperUtil import knowledge_distillation_loss
import datetime

class StudentDense:
    def __init__(self, callback=None):
        self.callbacks = callback
        self.nb_classes = 10
        self.input_shape = (28, 28, 1) # Input shape of each image
        self.nb_filters = 64 # number of convolutional filters to use
        self.dropoutVal = 0.2
        self.student = Sequential()
        self.epochs = 100
        self.batchSize = 256
        self.alpha = 0.1
        self.temp = 75
        self.name = "StudentDense"
    
    def printSummary(self):
        self.student.summary()

    def getModel(self):
        return self.student

    def buildAndCompile(self):
        # cannot reference self in lambda, or else error in model save
        tempAlpha = self.alpha
        tempTemp = self.temp
        # constructing the student network
        self.student.add(Flatten(input_shape=self.input_shape))
        self.student.add(Dense(32, activation='relu'))
        self.student.add(Dropout(self.dropoutVal))
        self.student.add(Dense(self.nb_classes))
        self.student.add(Activation('softmax'))
        # modifying student network for KD
        self.student.layers.pop()
        logits = self.student.layers[-1].output
        probs = Activation('softmax')(logits)
        logits_T = Lambda(lambda x: x / tempTemp)(logits)
        probs_T = Activation('softmax')(logits_T)
        output = concatenate([probs, probs_T])
        self.student = Model(self.student.input, output) # final student model
        #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.student.compile(
            #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
            optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
            loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, tempAlpha),
            #loss='categorical_crossentropy',
            metrics=[acc])

    def train(self, X_train, Y_train_new, X_test, Y_test_new):
        self.student.fit(X_train, Y_train_new,
            batch_size=self.batchSize,
            epochs=self.epochs,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=(X_test, Y_test_new))
        score = self.student.evaluate(X_test, Y_test_new, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save(self, X_test, Y_test):
        now = datetime.datetime.now()
        # evaluating model
        score = self.student.evaluate(X_test, Y_test, verbose=0)
        # serialize model to JSON
        model_json = self.student.to_json()
        with open("ModelConfigs/{}_{}_{}.json".format(round(score[1] * 100, 2), self.name, now.strftime("%Y-%m-%d_%H-%M-%S")), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.student.save_weights("ModelConfigs/{}_{}_{}.h5".format(round(score[1] * 100, 2), self.name, now.strftime("%Y-%m-%d_%H-%M-%S")))
        print("[INFO] Saved student model to disk")

    def load(self, model_filename, weights_filename):
        print('[INFO] creating custom student model')
        # load json and create model
        model_filename = os.path.join("ModelConfigs", model_filename)
        weights_filename = os.path.join("ModelConfigs", weights_filename)
        with open(model_filename, 'rb') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            self.student = model_from_json(loaded_model_json)
            # load weights into new model
            self.student.load_weights(weights_filename)
            self.student.compile(
                optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, self.alpha),
                metrics=[acc])
            print("[INFO] Loaded student model from disk w/ weights")

    def load(self, model_filename):
        model_filename = os.path.join("ModelConfigs", model_filename)
        with open(model_filename, 'rb') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            self.student = model_from_json(loaded_model_json)
            self.student.compile(
                optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, self.alpha),
                metrics=[acc])
            print("[INFO] Loaded student model from disk w/o weights")