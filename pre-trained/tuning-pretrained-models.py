# mobile net pre-trained network
import numpy as np
import keras
from keras.applications.mobilenet import MobileNet
from keras.datasets.fashion_mnist import load_data# Load the fashion-mnist train data and test data
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import argparse

# consuming command line arguments
parser = argparse.ArgumentParser(description='pre-trained model arguments.')
parser.add_argument('-t', action="store_true", default=False, dest='is_trainable', help='boolean indicating if the network is trainable')
args = parser.parse_args()

# loading the pre-trained MobileNet model
mobile_net = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# MobileNet V2  accepts one of the following formats: (96, 96), (128, 128), (160, 160),(192, 192), (224, 224)
# trainable has to be false in order to freeze the layers
for layer in base_model.layers:
  layer.trainable = args.is_trainable 
model = base_model

# method to prepare an input image
def prepare_image(file):
    img_path = 'MobileNet-inference-images/'
    img = image.load_img(img_path + file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mopbilenet.preprocess_input(img_array_expanded_dims)

preprocessed_image = prepare_image('1.PNG')
predictions = mobile_net.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)


# # loading the training data
# (x_train, y_train), (x_test, y_test) = load_data()
# # normalizing the data
# norm_x_train = x_train.astype('float32') / 255
# norm_x_test = x_test.astype('float32') / 255
# # one hot encoding the data
# encoded_y_train = to_categorical(y_train, num_classes=10, dtype='float32')
# encoded_y_test = to_categorical(y_test, num_classes=10, dtype='float32')

# # compilation of the model
# model.compile(optimizer=Adam(),
#               loss='categorical_crossentropy',
#               metrics=['categorical_accuracy'])

# # saving the model
# model_name = "keras_mobilenetv2"
# model.save(f"models/{model_name}.h5")