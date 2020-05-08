import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras import optimizers
from keras import regularizers

# For testing right now
def get_model():
    weight_decay = 0.0005
    num_classes = 10
    x_shape = [32, 32, 3]

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape, kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model









