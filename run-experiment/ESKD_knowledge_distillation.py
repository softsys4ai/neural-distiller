
# TODO test!

# STEP 1
# train the size 10 teacher in 5 epoch intervals
# harvest logits and save the model weights at each interval

# STEP 2
# Distill knowledge to student model for each set of logits

# Step 3
# Evaluate student models for robustness and accuracy

# external imports
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.losses import KLDivergence as KL
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.losses import categorical_crossentropy as logloss
# project imports
from Models import KnowledgeDistillationModels
from Data import LoadDataset
from Utils import HelperUtil



# setting up parameters for loading distillation logits
experiment_dir = "/home/blakete/neural-distiller/run-experiment/ESKD_cifar100_10_20-12-19_17:44:50"
dataset = "cifar100"
dataset_num_classes = 100
alpha = 1.0  # TODO test different values for KL loss
logits_dir = os.path.join(experiment_dir, "logits")
model_size = 2
student_epochs = 150
logit_model_size = 6
epoch_interval = 1  # TODO make the harvesting experiment directory name contain the epoch information
min_epochs = 1
total_epochs = 150
arr_epochs = np.arange(min_epochs, total_epochs + epoch_interval-1e-2, epoch_interval)
min_temp = 1
max_temp = 10
temp_interval = 0.5
arr_temps = np.arange(min_temp, max_temp + temp_interval, temp_interval)

# write student weights to file
def save_weights(models_dir, model, model_size, curr_epochs, total_epochs, train_temp, val_acc, train_acc):
    weight_filename = f"model_{model_size}_{curr_epochs}|{total_epochs}_{train_temp}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)
    return model_path


def knowledge_distillation_loss(y_true, y_pred, alpha=1.0):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[:, :dataset_num_classes], y_true[:, dataset_num_classes:]
    y_pred, y_pred_softs = y_pred[:, :dataset_num_classes], y_pred[:, dataset_num_classes:]
    # loss = (1-alpha)*logloss(y_true, y_pred) + alpha*logloss(y_true_softs, y_pred_softs)
    # original loss function that works for us
    # loss = logloss(y_true, y_pred) + alpha * logloss(y_true_softs, y_pred_softs)
    # testing this loss function, used by other works
    loss = logloss(y_true, y_pred) + alpha * logloss(y_true_softs, y_pred_softs)
    return loss


def knowledge_distillation_loss_KL(y_true, y_pred, alpha=1.0):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[:, :dataset_num_classes], y_true[:, dataset_num_classes:]
    y_pred, y_pred_softs = y_pred[:, :dataset_num_classes], y_pred[:, dataset_num_classes:]
    # loss = (1-alpha)*logloss(y_true, y_pred) + alpha*logloss(y_true_softs, y_pred_softs)
    loss = logloss(y_true, y_pred) + alpha * KL(y_true_softs, y_pred_softs)
    return loss


def acc(y_true, y_pred):
    y_true = y_true[:, :dataset_num_classes]
    y_pred = y_pred[:, :dataset_num_classes]
    return categorical_accuracy(y_true, y_pred)


# convert loaded logits to soft targets at a specified temperature
def modified_kd_targets_from_logits(Y_train, Y_test, train_logits, test_logits, temp):
    # create soft targets from loaded logits
    train_logits_t = train_logits / temp
    test_logits_t = test_logits / temp
    Y_train_soft = K.softmax(train_logits_t)
    Y_test_soft = K.softmax(test_logits_t)
    sess = K.get_session()
    Y_train_soft = sess.run(Y_train_soft)
    Y_test_soft = sess.run(Y_test_soft)
    # concatenate hard and soft targets to create the knowledge distillation targets
    Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
    Y_test_new = np.concatenate([Y_test, Y_test_soft], axis=1)
    return Y_train_new, Y_test_new

# TODO use this after preliminary run to see if results improve!
def normalizeStudentSoftTargets(Y_train_soft, Y_test_soft):
    for i in range(len(Y_train_soft)):
        sum = 0
        for val in Y_train_soft[i]:
            sum += val
        Y_train_soft[i] = [x/sum for x in Y_train_soft[i]]
    for i in range(len(Y_test_soft)):
        sum = 0
        for val in Y_test_soft[i]:
            sum += val
        Y_test_soft[i] = [x/sum for x in Y_test_soft[i]]
    return Y_train_soft, Y_test_soft


def load_and_compile_student(X_train, model_size):
    student_model = KnowledgeDistillationModels.get_model(dataset, dataset_num_classes, X_train, model_size, "resnet")
    return compile_student(student_model)


def compile_student(student_model, KD=False, alpha=1.0):
    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    if (KD):
        # todo resolve slicing bug with KL loss function
        student_model.compile(optimizer=optimizer,
                              loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, alpha),
                              metrics=[acc])
    else:
        student_model.compile(optimizer=optimizer,
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])
    return student_model


# method to load logits from the specified experiment directory
def load_logits(logits_dir, model_size, curr_epochs, total_epochs):
    logits_filename = os.path.join(logits_dir, f"logits_{model_size}_{curr_epochs}|{total_epochs}.pkl")
    with open(logits_filename, "rb") as file:
        teacher_train_logits = pickle.load(file)
        teacher_test_logits = pickle.load(file)
    return teacher_train_logits, teacher_test_logits


# create experiment run directory for each session's model weights and logits
log_dir = os.getcwd()
now = datetime.now()
now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
log_dir = os.path.join(log_dir, "ESKD_Knowledge_Distillation_" + dataset + f"_{model_size}_" + now_datetime)
os.mkdir(log_dir)
models_dir = os.path.join(log_dir, "models")
os.mkdir(models_dir)

# load and shift data by train mean
X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean
min = np.min(X_train)
max = np.max(X_train)

# load and save model starting weights to be used for each experiment
student_model = load_and_compile_student(X_train, model_size)
train_acc = student_model.evaluate(X_train, Y_train, verbose=0)
val_acc = student_model.evaluate(X_test, Y_test, verbose=0)
starting_model_path = save_weights(models_dir, student_model, model_size, 0, total_epochs, 0,
                               format(val_acc[1], '.3f'), format(train_acc[1], '.3f'))

# iterate logits stored for each interval and distill a student model with them
for i in range(len(arr_epochs)):
    # load logits
    train_logits, test_logits = load_logits(logits_dir, logit_model_size, arr_epochs[i], total_epochs)
    for j in range(len(arr_temps)):
        print("--------------------------Starting new KD step--------------------------")
        print(f"teacher network logits {arr_epochs[i]}|{total_epochs}, student trained at temperature {arr_temps[j]}")
        # clear current session to free memory
        tf.keras.backend.clear_session()
        # apply temperature to logits and create modified targets for knowledge distillation
        Y_train_new, Y_test_new = modified_kd_targets_from_logits(Y_train, Y_test, train_logits, test_logits, arr_temps[j])
        # load student model
        student_model = load_and_compile_student(X_train, model_size)
        # modify student model for knowledge distillation
        student_model = HelperUtil.apply_knowledge_distillation_modifications(None, student_model, arr_temps[j])
        student_model = compile_student(student_model, True, alpha)
        # load starting model weights
        student_model.load_weights(starting_model_path)
        # train student model on hard and soft targets
        checkpoint_filename = f"checkpoint_model_{model_size}_{arr_epochs[i]}|{total_epochs}.h5"
        callbacks = [
            EarlyStopping(monitor='val_acc', patience=25, min_delta=0.001),
            ModelCheckpoint(checkpoint_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        ]
        student_model.fit(X_train, Y_train_new,
                          batch_size=128,
                          epochs=student_epochs,
                          verbose=1,
                          callbacks=callbacks,
                          validation_data=(X_test, Y_test_new))
        # delete modified and reload unmodified student network
        del student_model
        student_model = load_and_compile_student(X_train, model_size)
        # load model from checkpoint and save it proper location with save weights method
        student_model.load_weights(checkpoint_filename)
        student_model = compile_student(student_model)
        # evaluate student model after training
        train_acc = student_model.evaluate(X_train, Y_train, verbose=0)
        val_acc = student_model.evaluate(X_test, Y_test, verbose=0)
        save_weights(models_dir, student_model, model_size, arr_epochs[i], total_epochs, arr_temps[j],
                     format(val_acc[1], '.5f'), format(train_acc[1], '.5f'))
        # upon completion, delete the checkpoint file
        os.remove(checkpoint_filename)
        del student_model
