
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
import csv
import time
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.python.keras.losses import KLDivergence as KL
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.losses import categorical_crossentropy as logloss
# project imports
from models import knowledge_distillation_models
from data import load_dataset
from utils import helper_util
from utils import config_reference as cfg


# write student weights to file
def save_weights(models_dir, model, model_size, curr_epochs, total_epochs, train_temp, val_acc, train_acc):
    weight_filename = f"model_{model_size}_{curr_epochs}|{total_epochs}_{train_temp}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)
    return model_path

def save_model(models_dir, model, model_size, curr_epochs, total_epochs, train_temp, val_acc, train_acc):
    model_filename = f"model_{model_size}_{curr_epochs}|{total_epochs}_{train_temp}_{val_acc}_{train_acc}.h5"
    weight_filename = f"model_weights_{model_size}_{curr_epochs}|{total_epochs}_{train_temp}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, model_filename)
    model_weight_path = os.path.join(models_dir, weight_filename)
    model.save(model_path)
    model.save_weights(model_weight_path)
    del model
    return model_path, model_weight_path

def knowledge_distillation_loss(y_true, y_pred, alpha=1.0):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[:, :cfg.dataset_num_classes], y_true[:, cfg.dataset_num_classes:]
    y_pred, y_pred_softs = y_pred[:, :cfg.dataset_num_classes], y_pred[:, cfg.dataset_num_classes:]
    # loss = (1-alpha)*logloss(y_true, y_pred) + alpha*logloss(y_true_softs, y_pred_softs)
    # original loss function that works for us
    # loss = logloss(y_true, y_pred) + alpha * logloss(y_true_softs, y_pred_softs)
    # testing this loss function, used by other works
    loss = logloss(y_true, y_pred) + alpha * logloss(y_true_softs, y_pred_softs)
    return loss


def knowledge_distillation_loss_KL(y_true, y_pred, alpha=1.0):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[:, :cfg.dataset_num_classes], y_true[:, cfg.dataset_num_classes:]
    y_pred, y_pred_softs = y_pred[:, :cfg.dataset_num_classes], y_pred[:, cfg.dataset_num_classes:]
    # loss = (1-alpha)*logloss(y_true, y_pred) + alpha*logloss(y_true_softs, y_pred_softs)
    loss = logloss(y_true, y_pred) + alpha * KL(y_true_softs, y_pred_softs)
    return loss


def acc(y_true, y_pred):
    y_true = y_true[:, :cfg.dataset_num_classes]
    y_pred = y_pred[:, :cfg.dataset_num_classes]
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
    student_model = knowledge_distillation_models.get_model(cfg.dataset, cfg.dataset_num_classes, X_train, model_size, cfg.model_type)
    return compile_student(student_model)


def compile_student(student_model, KD=False, alpha=1.0):
    optimizer = SGD(lr=cfg.learning_rate, momentum=0.9, nesterov=True)
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

def run(training_gpu):
    # create experiment run directory for each session's model weights and logits
    log_dir = cfg.log_dir
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
    log_dir = os.path.join(log_dir, "ESKD_Knowledge_Distillation_" + cfg.dataset + f"_{cfg.student_model_size}_" + now_datetime)
    os.mkdir(log_dir)
    models_dir = os.path.join(log_dir, "models")
    os.mkdir(models_dir)

    # load and normalize
    X_train, Y_train, X_test, Y_test = load_dataset.load_dataset_by_name(None, cfg.dataset)
    X_train, X_test = load_dataset.z_standardization(X_train, X_test)

    if cfg.debug1 is True:
        X_train = X_train[:128]
        Y_train = Y_train[:128]
        X_test = X_test[:128]
        Y_test = Y_test[:128]

    # load and save model starting weights to be used for each experiment
    if not cfg.USE_EXPLICIT_START_MODEL:
        student_model = load_and_compile_student(X_train, cfg.student_model_size)
        train_acc = student_model.evaluate(X_train, Y_train, verbose=0)
        val_acc = student_model.evaluate(X_test, Y_test, verbose=0)
        starting_model_path, starting_model_weight_path = save_model(models_dir, student_model, cfg.student_model_size, 0, cfg.total_teacher_logit_epochs, 0,
                                       format(val_acc[1], '.5f'), format(train_acc[1], '.5f'))
    else:
        starting_model_path = cfg.EXPLICIT_START_MODEL_PATH
        starting_model_weight_path = cfg.EXPLICIT_START_MODEL_WEIGHT_PATH


    # create results list to store test generated results
    #                   student model size,      teacher training epoch used,           distillation temp applied,          test accuracy         training accuracy,         distil. test accuracy,   distil. training accuracy,    distil. step time
    test_results = [["student model size", "teacher model training epoch used", "applied temperature", "test accuracy", "train accuracy", "distillation test accuracy", "distillation train accuracy", "distillation step cumulative time"]]

    # load student model
    student_model = load_and_compile_student(X_train, cfg.student_model_size)
    # iterate logits stored for each interval and distill a student model with them
    for i in range(len(cfg.arr_of_distillation_epochs)):
        # load logits
        train_logits, test_logits = load_logits(cfg.logits_dir, cfg.teacher_logit_model_size, int(cfg.arr_of_distillation_epochs[i]), cfg.total_teacher_logit_epochs)
        for j in range(len(cfg.arr_of_distillation_temps)):
            time_start = time.time()
            print("\n--------------------------Starting new KD step--------------------------")
            print(f"teacher network logits {int(cfg.arr_of_distillation_epochs[i])}|{cfg.total_teacher_logit_epochs}, {cfg.model_type} student trained at temperature {cfg.arr_of_distillation_temps[j]}")
            print("[INFO] Creating knowledge distillation targets...")
            # apply temperature to logits and create modified targets for knowledge distillation
            Y_train_new, Y_test_new = modified_kd_targets_from_logits(Y_train, Y_test, train_logits, test_logits, cfg.arr_of_distillation_temps[j])
            # print("[INFO] Cleaning up previous training session...")
            # tensorflow.python.keras.backend.clear_session()
            print("[INFO] Loading starting model...")
            # load starting model
            student_model = load_model(starting_model_path)
            # student_model.summary()
            print("[INFO] Preparing starting model for knowledge distillation...")
            # modify student model for knowledge distillation
            student_model = helper_util.apply_knowledge_distillation_modifications(None, student_model,
                                                                                   cfg.arr_of_distillation_temps[j])
            studentmodel = compile_student(student_model, True, cfg.alpha)
            # load starting model weights
            student_model.load_weights(starting_model_weight_path)
            # train student model on hard and soft targets
            checkpoint_filename = f"checkpoint_model_gpu_{int(training_gpu)}_size_{cfg.student_model_size}_temp_{cfg.arr_of_distillation_temps[j]}.h5"
            callbacks = [
                EarlyStopping(monitor='val_acc', patience=30, min_delta=0.001),
                ModelCheckpoint(checkpoint_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
            ]
            student_model.fit(X_train, Y_train_new,
                              batch_size=cfg.train_batch_size,
                              epochs=cfg.student_epochs,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=(X_test, Y_test_new))
            # calculation of distillation accuracies
            distillation_train_acc = student_model.evaluate(X_train, Y_train_new, verbose=0)
            distillation_val_acc = student_model.evaluate(X_test, Y_test_new, verbose=0)
            print("[INFO] Loading starting model...")
            # load starting model
            student_model = load_model(starting_model_path)
            print("[INFO] Loading student knowledge distillation weights...")
            # load model from checkpoint and save it proper location with save weights method
            student_model.load_weights(checkpoint_filename)
            print("[INFO] Evaluating student model...")
            # evaluate student model after training
            train_acc = student_model.evaluate(X_train, Y_train, verbose=0)
            val_acc = student_model.evaluate(X_test, Y_test, verbose=0)
            print("[INFO] Saving student model and cleaning up training session...")
            save_model(models_dir, student_model, cfg.student_model_size, int(cfg.arr_of_distillation_epochs[i]), cfg.total_teacher_logit_epochs, cfg.arr_of_distillation_temps[j],
                         format(val_acc[1], '.5f'), format(train_acc[1], '.5f'))
            # upon completion, delete the checkpoint file
            os.remove(checkpoint_filename)
            time_end = time.time()
            print(f"Iteration total time (minutes): {(time_end - time_start) / 60}")
            # write result to csv file
            #                   student model size,      teacher training epoch used,           distillation temp applied,          test accuracy         training accuracy,         distil. test accuracy,   distil. training accuracy,    distil. step time
            generated_row = [cfg.student_model_size, int(cfg.arr_of_distillation_epochs[i]), cfg.arr_of_distillation_temps[j], format(val_acc[1], '.5f'), format(train_acc[1], '.5f'), distillation_val_acc, distillation_train_acc, (time_end-time_start)]
            test_results.append(generated_row)
            # opening the csv file in 'w+' mode
            file = open('g4g.csv', 'w+', newline='')
            # writing the data into the file
            with open(os.path.join(log_dir, "results.csv"), "w+") as result_file:
                write = csv.writer(result_file)
                write.writerows(test_results)
            # print(test_results)
            K.clear_session()


