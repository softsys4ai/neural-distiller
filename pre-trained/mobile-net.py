import keras
import numpy as np
from keras.applications import mobilenet
from keras.preprocessing import image as image_utils
#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
#import matplotlib.pyplot as plt
import numpy as np
filename = 'MobileNet-inference-images/cat.jpg'
def pre_process_image(name):
    image = image_utils.load_img(name, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image
print("[INFO] preprocessing image...")
image = pre_process_image('MobileNet-inference-images/fox.jpg')
processed_image_mobilenet = mobilenet.preprocess_input(image.copy())
predictions_mobilenet = mobilenet_model.predict(processed_image_mobilenet)
label_mobilenet = decode_predictions(predictions_mobilenet)
print ('label_mobilenet = %s' % label_mobilenet)