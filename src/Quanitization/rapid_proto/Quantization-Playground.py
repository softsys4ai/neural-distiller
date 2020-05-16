import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# iris = load_iris()
# X = iris['data']
# y = iris['target']
# names = iris['target_names']
# feature_names = iris['feature_names']
#
# # One hot encoding
# enc = OneHotEncoder()
# Y = enc.fit_transform(y[:, np.newaxis]).toarray()
#
# # Scale data to have mean 0 and variance 1
# # which is importance for convergence of the neural network
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Split the data set into training and testing
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_scaled, Y, test_size=0.5, random_state=2)


def deterministic_quantization_rule(weight_value):
    if (weight_value > 0.3):
        return 1
    elif (weight_value < -0.3):
        return -1
    else:
        return 0

def quantization_policy(input_weights):
    q_weights = np.copy(input_weights)
    rows = len(input_weights)
    columns = len(input_weights[0])
    for r in range(rows):
        for c in range(columns):
            q_weights[r][c] = deterministic_quantization_rule(input_weights[r][c])
    return q_weights

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model2 = Sequential([
#             Conv2D(16, kernel_size=(3, 3),
#                    activation='relu',
#                    input_shape=X_train.shape[1:]),
#             # MaxPooling2D(pool_size=(2, 2)),
#             Conv2D(32, (3, 3), activation='relu'),
#             # Conv2D(8, (3, 3), activation='relu'),
#             # MaxPooling2D(pool_size=(2, 2)),
#             Flatten(),
#             Dense(32, activation='relu'),
#             Dense(10, name='logits'),
#             Activation('softmax')  # Note that we add a normal softmax layer to begin with
#         ])

print(model.get_weights())

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))


model_q = Sequential()
model_q.add(Dense(32, activation='relu', input_dim=100,
                  kernel_initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=33)))
model_q.add(Dense(1, activation='sigmoid'))
model_q.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

q_model_weights_ref = model_q.get_weights()
q_model_weights = model_q.get_weights()
for i in range(len(q_model_weights)):
    if i == 0:
        q_model_weights[i] = quantization_policy(q_model_weights[i])
    if i == 2:
        q_model_weights[i] = quantization_policy(q_model_weights[i])

# sync model weights
model.set_weights(model_q.get_weights())
model_q.set_weights(q_model_weights)

print("\nFP Model Summary:")
model.summary()
print("\nFP Model Weights:")
print(model.get_weights())
print("\n-----------------------------------\n")
print("\nTernary Model Summary:")
model_q.summary()
print("\nTernary Model Weights:")
print(model_q.get_weights())

# q_predictions = model_q.predict(data)
# print(round(q_predictions))
sess = K.get_session()
trainingExample = np.random.random((1,100))
for epoch in range(1, 101, 1):
    print(f"[INFO] Training Epoch {epoch}")
    model_q.fit(data, labels, epochs=1, batch_size=32, verbose=1)
    gradients = K.gradients(model_q.output, model_q.trainable_weights)
    evaluated_gradients = sess.run(gradients, feed_dict={model.input: trainingExample})
    # # batch_size = 50
    # # for i in range(0, 1000, batch_size): # batching
    # model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
    # q_model_weights = model.get_weights()
    # for i in range(len(q_model_weights)):
    #     if i == 0:
    #         q_model_weights[i] = quantization_policy(q_model_weights[i])
    #     if i == 2:
    #         q_model_weights[i] = quantization_policy(q_model_weights[i])
    # model_q.set_weights(q_model_weights)
    #
    # print("\nFP Model Weights:")
    # print(model.get_weights())
    # print("\nTernary Model Weights:")
    # print(model_q.get_weights())
    #
    # # evaluate quantized model
    # _, accuracy = model.evaluate(data, labels)
    # print('FP Accuracy: %.2f' % (accuracy * 100))
    # _, accuracy = model_q.evaluate(data, labels)
    # print('Q Accuracy: %.2f' % (accuracy * 100))
    #
    # # fp_predictions = model.predict(data[i:i + batch_size])
    # # q_predictions = model_q.predict(data[i:i+batch_size])
    # # fp_loss = np.sum(keras.losses.binary_crossentropy(labels[i:i+batch_size], fp_predictions, from_logits=False, label_smoothing=0)) / batch_size
    # # q_loss = np.sum(keras.losses.binary_crossentropy(labels[i:i+batch_size], q_predictions, from_logits=False, label_smoothing=0)) / batch_size
    # # print(fp_loss)
    # # todo: forward propogate
    # # todo: calculate loss
    # # todo: change FP model weights
    # # todo: apply quantization policy to create quantized network
    #



print("\n[INFO] Starting training loop...")
# model.fit(data, labels, epochs=200, batch_size=32)





