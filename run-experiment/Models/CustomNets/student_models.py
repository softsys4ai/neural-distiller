# import os
# # from tensorflow.python.keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate, MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
# # from tensorflow.python.keras.models import Sequential, model_from_json
# #
# # # TODO create an easy method to generate and return a student model of a specified size
# CNN_10 = Sequential([
#     Conv2D(32, kernel_size=(3, 3),
#                             activation='relu',
#                             input_shape=self.input_shape),
#     Conv2D(64, (3, 3), activation='relu')
#     MaxPooling2D(pool_size=(2, 2))
#     Dropout(0.25))  # For reguralization
#     Flatten()
#     Dense(128, activation='relu')
#     Dropout(0.5)  # For reguralization
#     Dense(self.nb_classes)
#     Activation('softmax') # Note that we add a normal softmax layer to begin with
# ])
# #
# # CNN_8 = Sequential([
# #     Dense(32, input_shape=(784,)),
# #     Activation('relu'),
# #     Dense(10),
# #     Activation('softmax'),
# # ])
# #
# # CNN_6 = Sequential([
# #     Dense(32, input_shape=(784,)),
# #     Activation('relu'),
# #     Dense(10),
# #     Activation('softmax'),
# # ])
# #
# # CNN_4 = Sequential([
# #     Dense(32, input_shape=(784,)),
# #     Activation('relu'),
# #     Dense(10),
# #     Activation('softmax'),
# # ])
# #
# # CNN_3 = Sequential([
# #     Dense(32, input_shape=(784,)),
# #     Activation('relu'),
# #     Dense(10),
# #     Activation('softmax'),
# #

# TODO implement multi-layer perceptron networks of varying capacity
# # constructing the student network
# self.student.add(Flatten(input_shape=self.input_shape))
# self.student.add(Dense(cfg.student_dense_256_size, activation='relu'))
# self.student.add(Dropout(self.dropoutVal))
# self.student.add(Dense(self.nb_classes))
# self.student.add(Activation('softmax'))