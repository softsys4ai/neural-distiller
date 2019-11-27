from keras import backend as K
import numpy as np
from numpy import save
from keras.datasets import cifar100
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_cifar_100():
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

# load dataset
X_train, Y_train, X_test, Y_test = load_cifar_100()

datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=True,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
datagen.fit(X_train)
data_x = []
data_y = []
i = 0
multFactor = 3
generated_batches = 600*multFactor
for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=100):
    i += 1
    print("Parsing batch " + str(i))
    data_x.append(x_batch)
    data_y.append(y_batch)
    if i >= generated_batches:
        break
x_train = np.vstack(data_x)
y_train = np.vstack(data_y)
print(x_train.shape)
print(y_train.shape)
save("data/x_train_60.npy", x_train)
save("data/y_train_60.npy", y_train)
print("complete")
















