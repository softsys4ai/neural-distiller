import os
from tensorflow.python.keras.layers import Dropout, Dense, Flatten, Activation
from tensorflow.python.keras.models import Sequential, model_from_json
from Utils.HelperUtil import acc
from Utils.HelperUtil import knowledge_distillation_loss
import datetime
from Configuration import Config as cfg

class StudentDense32:
    def __init__(self, callback=None):
        self.callbacks = callback
        self.nb_classes = 10
        self.input_shape = (28, 28, 1) # Input shape of each image
        self.nb_filters = 64 # number of convolutional filters to use
        self.dropoutVal = 0.0
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
        # constructing the student network
        self.student.add(Flatten(input_shape=self.input_shape))
        self.student.add(Dense(cfg.base_dense_student_size, activation='relu'))
        self.student.add(Dropout(self.dropoutVal))
        self.student.add(Dense(self.nb_classes))
        self.student.add(Activation('softmax'))
        # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.student.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
            loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, tempAlpha),
            # loss='categorical_crossentropy',
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
        with open(os.path.join(cfg.teacher_model_dir, "{}_{}_{}.json".format(round(score[1] * 100, 2), self.name, now.strftime("%Y-%m-%d_%H-%M-%S"))), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.student.save_weights(os.path.join(cfg.teacher_model_dir, "{}_{}_{}.h5".format(round(score[1] * 100, 2), self.name, now.strftime("%Y-%m-%d_%H-%M-%S"))))
        print("[INFO] Saved student model to disk")

    def load(self, model_filename, weights_filename):
        print('[INFO] creating custom student model')
        # load json and create model
        model_filename = os.path.join(cfg.teacher_model_dir, model_filename)
        weights_filename = os.path.join(cfg.teacher_model_dir, weights_filename)
        with open(model_filename, 'rb') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            self.student = model_from_json(loaded_model_json)
            # load weights into new model
            self.student.load_weights(weights_filename)
            self.student.compile(
                optimizer=cfg.student_optimizer,
                loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, self.alpha),
                metrics=[acc])
            print("[INFO] Loaded student model from disk w/ weights")

    def load(self, model_filename):
        model_filename = os.path.join(cfg.teacher_model_dir, model_filename)
        with open(model_filename, 'rb') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            self.student = model_from_json(loaded_model_json)
            self.student.compile(
                optimizer=cfg.student_optimizer,
                loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, self.alpha),
                metrics=[acc])
            print("[INFO] Loaded student model from disk w/o weights")