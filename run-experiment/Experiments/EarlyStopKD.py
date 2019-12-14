# Blake + Stephen ESKD Experiment

# STEP 1
# train the size 10 teacher in 5 epoch intervals
# harvest logits and save the model weights at each interval

# STEP2
# Distill knowledge to student model for each set of logits

# Step 3
# Evaluate student models for robustness and accuracy


# external dependencies
import os
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from keras.optimizers import SGD
import tensorflow.python.keras as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# project imports
import LoadDataset
from Utils import TeacherUtils
import KnowledgeDistillationModels
from Configuration import Config as cfg

# add high priority
os.nice(1)

# write teacher weights to file
def save_weights(models_dir, model, model_size, curr_epochs, total_epochs, val_acc, train_acc):
    weight_filename = f"{model_size}_{curr_epochs}|{total_epochs}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)

# write teacher logits to file
def save_logits(logits_dir, model, model_size, curr_epochs, total_epochs, train_logits, test_logits):
    logits_filename = os.path.join(logits_dir, f"{model_size}_{curr_epochs}|{total_epochs}.pkl")
    with open(logits_filename, "wb") as file:
        pickle.dump(train_logits, file)
    pickle.dump(test_logits, file)

X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean
min = np.min(X_train)
max = np.max(X_train)

# ESKD experiment hyperparameters
dataset = "cifar100"
teacher_model_size = 10
epoch_min = 0
epoch_max = 200
interval_size = 5
epoch_intervals = np.arange(epoch_min, epoch_max, interval_size)

# experiment directory structure
# ESKD_experiment_{datetime}
# saved_model_weights
# saved_weights

# create experiment run directory for each session's model weights and logits
log_dir = os.getcwd()
now = datetime.now()
now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
log_dir = os.path(log_dir, "ESKD_" + dataset + f"_{teacher_model_size}_" + now_datetime)
os.mkdir(log_dir)
logits_dir = os.path(log_dir, "logits")
os.mkdir(logits_dir)
models_dir = os.path(log_dir, "models")
os.mkdir(logits_dir)

teacher_model = KnowledgeDistillationModels.get_model_cifar100(100, X_train, teacher_model_size)
teacher_model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                      loss="categorical_crossentropy",
                      metrics=["acc"])

# intermittent training and harvesting of logits for ESKD experiment
for i in range(len(epoch_intervals)):

    # setup current iteration params and load model
    curr_epochs = epoch_intervals[i]
    print(f"Training size {teacher_model_size} teacher network {curr_epochs}|{epoch_max}")

    # load model for current iteration
    teacher_model = KnowledgeDistillationModels.get_model_cifar100(100, X_train, teacher_model_size)

    # clear current session to free memory
    if (i != 0):
        tf.keras.backend.clear_session()
        K.clear_session()
    teacher_model.load_weights()

    # compile and train network
    teacher_model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                          loss="categorical_crossentropy",
                          metrics=["acc"])
    teacher_model.fit(X_train, Y_train,
                      validation_data=(X_test, Y_test),
                      batch_szie=128,
                      epochs=interval_size,
                      verbose=1,
                      callbacks=None)
    # evaluate and save model weights
    train_acc = teacher_model.evaluate(X_train, Y_train, verbose=0)
    val_acc = teacher_model.evaluate(X_test, Y_test, verbose=0)
    save_weights(models_dir, teacher_model, curr_epochs, val_acc, train_acc)
    # create and save logits from model
    train_logits, test_logits = TeacherUtils.createStudentTrainingData(teacher_model, None, X_train, None, X_test, None)
    save_logits(logits_dir, train_logits, test_logits, curr_epochs)
    # delete the model for the next training iteration
    del teacher_model




















