import numpy as np
#from keras.datasets import imagenet
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.utils import to_categorical
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from efficientnet import EfficientNetB0
print("[INFO] loading model...")
model = EfficientNetB0(weights='imagenet')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
# testing efficientnet
def pre_process_image(name):
    image = image_utils.load_img(name, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
print("[INFO] preprocessing image...")
image = pre_process_image('MobileNet-inference-images/golden.jpg')
# classify the given image
print("[INFO] classifying image...")
preds = model.predict(image)
P = decode_predictions(preds)
print(P)



# (x_train, y_train), (x_test, y_test) = imagenet.load_data()
# NUM_CLASSES = 100
# y_train = to_categorical(y_train, NUM_CLASSES)
# y_test = to_categorical(y_test, NUM_CLASSES)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255.0
# x_test /= 255.0