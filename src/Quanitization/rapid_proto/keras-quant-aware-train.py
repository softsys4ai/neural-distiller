import numpy as np
import tensorflow as tf
from keras import backend as K

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],  1) / 255.   # normalize
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) / 255.      # normalize
y_train = to_categorical(y_train, num_classes=10)    #one hot
y_test = to_categorical(y_test, num_classes=10)    #one hot

# Create model
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax', name='pred'))

# Quantization aware training
sess = K.get_session()
tf.contrib.quantize.create_training_graph(sess.graph)
sess.run(tf.global_variables_initializer())

# You can plot the quantize training graph on tensorboard
# tf.summary.FileWriter('/workspace/tensorboard', graph=sess.graph)

# Define optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=256)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

# Print the min max in fakequant
for node in sess.graph.as_graph_def().node:
    if 'weights_quant/AssignMaxLast' in node.name \
        or 'weights_quant/AssignMinLast' in node.name:
        tensor = sess.graph.get_tensor_by_name(node.name + ':0')
        print('{} = {}'.format(node.name, sess.run(tensor)))