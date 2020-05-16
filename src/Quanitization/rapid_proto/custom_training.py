import tensorflow as tf
import numpy as np
import keras.backend as K
from tqdm import tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.losses import binary_crossentropy

# Setting seeds for reproducibility
np.random.seed(0)
tf.set_random_seed(0)

# Dataset: given 2 numbers, predict the sum
# Sum 2 numbers from 0 to 10 dataset
samples = np.random.randint(0, 9, size=(10, 2))
targets = np.sum(samples, axis=-1)

# Samples for testing
samples_test = np.random.randint(0, 9, size=(10, 2))
targets_test = np.sum(samples_test, axis=-1)

# Model
x = Input(shape=[2])
y = Dense(units=1)(x)
model = Model(x, y)


# Loss
def loss_fn(y_true, y_pred):
    # You can get all the crazy and twisted you
    # want here no Keras restrictions this time :)
    loss_value = K.sum(K.pow((y_true - y_pred), 2))
    return loss_value


# Optimizer to run the gradients
optimizer = Adam(lr=1e-4)

# Graph creation
# Creating training flow
# Ground truth input, samples or X_t
y_true = Input(shape=[0])

# Prediction
y_pred = model(x)

# Loss
loss = loss_fn(y_true, y_pred)

# Operation for getting
# gradients and updating weights
updates_op = optimizer.get_updates(
    params=model.trainable_weights,
    loss=loss)

# The graph is created, now we need to call it
# this would be similar to tf session.run()
train = K.function(
    inputs=[x, y_true],
    outputs=[loss],
    updates=updates_op)

test = K.function(
    inputs=[x, y_true],
    outputs=[loss])

# Training loop
epochs = 200

for epoch in range(epochs):
    print('Epoch %s:' % epoch)

    # Fancy progress bar
    pbar = tqdm(range(len(samples)))

    # Storing losses for computing mean
    losses_train = []

    # Batch loop: batch size=1
    for idx in pbar:
        sample = samples[idx]
        target = targets[idx]

        # Adding batch dim since batch=1
        sample = np.expand_dims(sample, axis=0)
        target = np.expand_dims(target, axis=0)

        # To tensors, input of
        # K.function must be tensors
        sample = K.constant(sample)
        target = K.constant(target)

        # Running the train graph
        loss_train = train([sample, target])

        # Compute loss mean
        losses_train.append(loss_train[0])
        loss_train_mean = np.mean(losses_train)

        # Update progress bar
        pbar.set_description('Train Loss: %.3f' % loss_train_mean)

    # Testing
    losses_test = []
    for idx in range(len(samples_test)):
        sample_test = samples_test[idx]
        target_test = targets_test[idx]

        # Adding batch dim since batch=1
        sample_test = np.expand_dims(sample_test, axis=0)
        target_test = np.expand_dims(target_test, axis=0)

        # To tensors
        sample_test = K.constant(sample_test)
        target_test = K.constant(target_test)

        # Evaluation test graph
        loss_test = test([sample_test, target_test])

        # Compute test loss mean
        losses_test.append(loss_test[0])

    loss_test_mean = np.mean(losses_test)
    print('Test Loss: %.3f' % loss_test_mean)