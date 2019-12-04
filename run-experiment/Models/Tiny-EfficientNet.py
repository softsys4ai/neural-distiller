import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Activation, MaxPool2D, \
    GlobalAveragePooling2D, BatchNormalization, Add, multiply, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.backend import sigmoid

def relu6_Activation(x):
  return min(max(0, x), 6)

def Swish_Activation(input, beta=1.0):
    return input * sigmoid(beta * input)

def DropConnect(input, prob):
    return tf.nn.Dropout(input, keep_prob=prob) * prob

def SE_block(in_block, channels, ratio):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(channels//ratio, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return multiply()([in_block, x])

def basic_inverted_residual_block(x, expand=64, squeeze=16):
  m = Conv2D(expand, (1,1), activation='relu')(x)
  m = DepthwiseConv2D((3,3), activation='relu')(m)
  m = Conv2D(squeeze, (1,1), activation='relu')(m)
  return Add()([m, x])

def inverted_residual_block(x, expand=64, squeeze=16):
  m = Conv2D(expand, (1,1))(x)
  m = BatchNormalization()(m)
  m = Activation('relu6')(m)
  m = DepthwiseConv2D((3,3))(m)
  m = BatchNormalization()(m)
  m = Activation('relu6')(m)
  m = Conv2D(squeeze, (1,1))(m)
  m = BatchNormalization()(m)
  return Add()([m, x])



# building TinyEfficientNet
numClasses = 100


input_shape = (32, 32, 3)
inputs = Input(shape=input_shape)
# stem of network
x = Conv2D(32, kernel_size=3, strides=2, use_bias=False)(inputs)
x = BatchNormalization(momentum=0.01, epsilon=1e-3)(x)
x = inverted_residual_block(x)
x = inverted_residual_block(x)
x = inverted_residual_block(x)
# head of network
x = Conv2D(32, kernel_size=1, use_bias=False)(x)
x = BatchNormalization(momentum=0.01, epsilon=1e-3)(x)
# final layers
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(numClasses, name='logits')(x)
output = Activation(Swish_Activation, name='SwishActivation')(x)

optimizer = get_optimizer(cfg.start_teacher_optimizer)
model.compile(optimizer=optimizer,
              loss=logloss,  # the same as the custom loss function
              metrics=['accuracy'])

















