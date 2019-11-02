import os
from tensorflow.python.keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from tensorflow.python.keras.models import Model, Sequential, model_from_json
import numpy as np
from Utils import HelperUtil
import datetime
from Configuration import Config as cfg

# teacher model class
class TeacherCNN:
    # TODO add CL arguments to change the teacher's hyperparameters
    def __init__(self, callback=None):
        self.callbacks = callback
        self.nb_classes = 10
        self.input_shape = (28, 28, 1) # Input shape of each image
        self.nb_filters = 64 # number of convolutional filters to use
        self.pool_size = (2, 2) # size of pooling area for max pooling
        self.kernel_size = (3, 3) # convolution kernel size
        self.teacher = Sequential()
        self.teacher_WO_Softmax = None
        self.epochs = 10
        self.batch_size = 256
        self.temp = 40
        self.name = "TeacherCNN"

    def printSummary(self):
        self.teacher.summary()

    def getModel(self):
        return self.teacher

    def createTeacherWOSoftmax(self):
        self.teacher_WO_Softmax = Model(self.teacher.input, self.teacher.get_layer('dense', 7).output)

    def buildAndCompile(self):
        self.teacher.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=self.input_shape))
        self.teacher.add(Conv2D(64, (3, 3), activation='relu'))
        self.teacher.add(MaxPooling2D(pool_size=(2, 2)))

        self.teacher.add(Dropout(0.25)) # For reguralization

        self.teacher.add(Flatten())
        self.teacher.add(Dense(128, activation='relu'))
        self.teacher.add(Dropout(0.5)) # For reguralization

        self.teacher.add(Dense(self.nb_classes))
        self.teacher.add(Activation('softmax')) # Note that we add a normal softmax layer to begin with

        self.teacher.compile(loss='categorical_crossentropy',
                    optimizer=cfg.student_optimizer,
                    metrics=['accuracy'])
        self.createTeacherWOSoftmax()
        return self.teacher
        
    def train(self, X_train, Y_train, X_test, Y_test):
        self.teacher.fit(X_train, Y_train,
            batch_size=self.batch_size,
            epochs=cfg.teacher_epochs,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=(X_test, Y_test))
        score = self.teacher.evaluate(X_test, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    def createStudentTrainingData(self, X_train, Y_train, X_test, Y_test):
        teacher_train_logits = self.teacher_WO_Softmax.predict(X_train)
        teacher_test_logits = self.teacher_WO_Softmax.predict(X_test) # This model directly gives the logits ( see the teacher_WO_softmax model above)
        # Perform a manual softmax at raised temperature
        train_logits_T = teacher_train_logits / self.temp
        test_logits_T = teacher_test_logits / self.temp
        Y_train_soft = HelperUtil.softmax(train_logits_T)
        Y_test_soft = HelperUtil.softmax(test_logits_T)
        # Concatenate so that this becomes a 10 + 10 dimensional vector
        Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
        Y_test_new = np.concatenate([Y_test, Y_test_soft], axis=1)
        return Y_train_new, Y_test_new

    def save(self, X_test, Y_test):
        now = datetime.datetime.now()
        # evaluating the model
        score = self.teacher.evaluate(X_test, Y_test, verbose=0)
        # serialize model to JSON
        model_json = self.teacher.to_json()
        with open(os.path.join(cfg.teacher_model_dir, "{}_{}_{}.json".format(round(score[1] * 100, 2), self.name, now.strftime("%Y-%m-%d_%H-%M-%S"))), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.teacher.save_weights(os.path.join(cfg.teacher_model_dir, "{}_{}_{}.h5".format(round(score[1] * 100, 2), self.name, now.strftime("%Y-%m-%d_%H-%M-%S"))))
        print("[INFO] Saved model to disk")
        
        # later...

    def load(self, model_filename, weights_filename):  
        # load json and create model
        model_filename = os.path.join(cfg.teacher_model_dir, model_filename)
        weights_filename = os.path.join(cfg.teacher_model_dir, weights_filename)
        with open(model_filename, 'rb') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            self.teacher = model_from_json(loaded_model_json)
            # load weights into new model
            self.teacher.load_weights(weights_filename)
            print("[INFO] Loaded model from disk")
            # evaluate loaded model on test data
            self.teacher.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])
            self.createTeacherWOSoftmax()