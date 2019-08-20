import keras
import numpy as np
from keras.applications import mobilenet
#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
#import matplotlib.pyplot as plt
import numpy as np
filename = 'MobileNet-inference-images/cat.jpg'
# load an image in PIL format
original_image = load_img(filename, target_size=(224, 224))
numpy_image = img_to_array(original_image)
input_image = np.expand_dims(numpy_image, axis=0)
processed_image_mobilenet = mobilenet.preprocess_input(input_image.copy())
predictions_mobilenet = mobilenet_model.predict(processed_image_mobilenet)
label_mobilenet = decode_predictions(predictions_mobilenet)
print ('label_mobilenet = %s' % label_mobilenet)