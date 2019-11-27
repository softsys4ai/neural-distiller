import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.datasets import cifar100
from keras.utils import np_utils
import numpy as np

def load_cifar_100(nb_classes, train_x, train_y):
    (x_del, y_del), (test_x, test_y) = cifar100.load_data()
    del x_del
    del y_del
    test_y = np_utils.to_categorical(test_y, nb_classes)
    test_x = test_x.astype('float32')
    test_x /= 255
    return train_x, train_y, test_x, test_y

# load dataset here
numClasses = 100
print("loading preprocessed training data")
x_train_unprocessed = np.load("data/x_train_60.npy")
Y_train_unprocessed = np.load("data/y_train_60.npy")
print("loading test data and scaling down")
x_train, y_train, x_test, y_test = load_cifar_100(numClasses, x_train_unprocessed, Y_train_unprocessed)

print("creating model and training")
# create and compile model
model = Sequential([
            Conv2D(32, kernel_size=3, input_shape=x_train_unprocessed.shape[1:], strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])

model.compile(optimizer=SGD(lr=0.01, momentum=0.5, nesterov=True),
              loss=categorical_crossentropy,
              metrics=['acc'])

# fit the model on the training set
model.fit(x_train, y_train,
          batch_size=128,
          epochs=150,
          verbose=1,
          callbacks=None,
          validation_data=(x_test, y_test))