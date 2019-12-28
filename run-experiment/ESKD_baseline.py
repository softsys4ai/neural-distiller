# Blake + Stephen ESKD Experiment

# TODO make this experiment script runnable from the train.py script

# STEP 1
# train the size 10 teacher in 5 epoch intervals
# harvest logits and save the model weights at each interval

# external dependencies
import os
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
# project imports
from Data import LoadDataset
from Utils import TeacherUtils
from Models import KnowledgeDistillationModels

# add high priority
os.nice(1)


# write teacher weights to file
def save_weights(models_dir, model, model_size, model_num, total_epochs, val_acc, train_acc):
    weight_filename = f"model_{model_size}_{model_num}|{total_epochs}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)
    return model_path


# write teacher logits to file
def save_logits(logits_dir, model_size, curr_epochs, total_epochs, train_logits, test_logits):
    logits_filename = os.path.join(logits_dir, f"logits_{model_size}_{curr_epochs}|{total_epochs}.pkl")
    with open(logits_filename, "wb") as file:
        pickle.dump(train_logits, file)
        pickle.dump(test_logits, file)

def reset_model_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean
min = np.min(X_train)
max = np.max(X_train)

# # use when debugging
# X_train = X_train[:128]
# Y_train = Y_train[:128]

# Set explicit starting model path
EXPLICIT_START_WEIGHT_PATH = "/home/blakete/model_10_0|200_0.01_0.008.h5"
USE_EXPLICIT_START = False
USE_SAME_STARTING_WEIGHTS = False

# ESKD experiment hyperparameters
num_models = 100
epochs = 150
learning_rate = 0.01
dataset = "cifar100"
teacher_model_size = 2

# experiment directory structure
# ESKD_experiment_{datetime}
# saved_model_weights
# saved_weights

# create experiment run directory for each session's model weights and logits
log_dir = os.getcwd()
now = datetime.now()
now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
log_dir = os.path.join(log_dir, "ESKD_baseline_" + dataset + f"_{teacher_model_size}_" + now_datetime)
os.mkdir(log_dir)
models_dir = os.path.join(log_dir, "models")
os.mkdir(models_dir)

# initialize and save starting network state
if (USE_SAME_STARTING_WEIGHTS):
    teacher_model = KnowledgeDistillationModels.get_model_cifar100(100, X_train, teacher_model_size)
    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    teacher_model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
    if (USE_EXPLICIT_START):
        teacher_model.load_weights(EXPLICIT_START_WEIGHT_PATH)
    train_acc = teacher_model.evaluate(X_train, Y_train, verbose=0)
    val_acc = teacher_model.evaluate(X_test, Y_test, verbose=0)
    prev_model_path = save_weights(models_dir, teacher_model, teacher_model_size, 0, num_models,
                                   format(val_acc[1], '.2f'), format(train_acc[1], '.3f'))

# intermittent training and harvesting of logits for ESKD experiment
teacher_model = KnowledgeDistillationModels.get_model_cifar100(100, X_train, teacher_model_size)
for i in range(1, num_models):

    # setup current iteration params and load model
    print(f"\nTraining size {teacher_model_size} network, {i}/{num_models}")
    # load model for current iteration
    if (USE_SAME_STARTING_WEIGHTS):
        teacher_model.load_weights(prev_model_path)
    else:
        reset_model_weights(teacher_model)

    # compile and train network
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    teacher_model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=20, min_delta=0.00007),
        ModelCheckpoint("checkpoint-model.hf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    ]
    teacher_model.fit(X_train, Y_train,
                      validation_data=(X_test, Y_test),
                      batch_size=128,
                      epochs=epochs,
                      verbose=1,
                      callbacks=callbacks)
    teacher_model.load_weights("checkpoint-model.hf5")
    # evaluate and save model weights
    train_acc = teacher_model.evaluate(X_train, Y_train, verbose=0)
    val_acc = teacher_model.evaluate(X_test, Y_test, verbose=0)
    save_weights(models_dir, teacher_model, teacher_model_size, i, num_models,
                 format(val_acc[1], '.3f'), format(train_acc[1], '.3f'))
    os.remove("checkpoint-model.hf5")  # remove previous checkpoint


















