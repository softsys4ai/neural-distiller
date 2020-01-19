# Blake + Stephen ESKD Experiment

# TODO make this experiment script runnable from the run_experiment.py script

# STEP 1
# train the size 10 teacher in 5 epoch intervals
# harvest logits and save the model weights at each interval

# external dependencies
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import SGD
# from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# project imports
from data import load_dataset
from utils import teacher_utils
from utils import config_reference as cfg
from models import knowledge_distillation_models

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

def run():
    # load and normalize
    X_train, Y_train, X_test, Y_test = load_dataset.load_cifar_100(None)
    X_train, X_test = load_dataset.z_standardization(X_train, X_test)

    # use when debugging
    if cfg.debug3 is True:
        X_train = X_train[:128]
        Y_train = Y_train[:128]

    # create experiment run directory for each session's model weights and logits
    log_dir = cfg.log_dir
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
    log_dir = os.path.join(log_dir, "ESKD_baseline_" + cfg.dataset + f"_{cfg.student_model_size}_" + now_datetime)
    os.mkdir(log_dir)
    models_dir = os.path.join(log_dir, "models")
    os.mkdir(models_dir)

    # initialize and save starting network state
    if (cfg.USE_SAME_STARTING_WEIGHTS):
        baseline_student_model = knowledge_distillation_models.get_model(cfg.dataset, 100, X_train, cfg.student_model_size, cfg.model_type)
        optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
        baseline_student_model.compile(optimizer=optimizer,
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])
        if (cfg.USE_EXPLICIT_START):
            baseline_student_model.load_weights(cfg.EXPLICIT_START_WEIGHT_PATH)
        train_acc = baseline_student_model.evaluate(X_train, Y_train, verbose=0)
        val_acc = baseline_student_model.evaluate(X_test, Y_test, verbose=0)
        prev_model_path = save_weights(models_dir, baseline_student_model, cfg.student_model_size, 0, cfg.num_models_to_train,
                                       format(val_acc[1], '.4f'), format(train_acc[1], '.4f'))

    # intermittent training and harvesting of logits for ESKD experiment
    baseline_student_model = knowledge_distillation_models.get_model(cfg.dataset, 100, X_train, cfg.student_model_size, cfg.model_type)
    for i in range(1, cfg.num_models_to_train):

        # setup current iteration params and load model
        print(f"\nTraining size {cfg.student_model_size} network, {i}/{cfg.num_models_to_train}")
        # load model for current iteration
        if (cfg.USE_SAME_STARTING_WEIGHTS):
            baseline_student_model.load_weights(prev_model_path)
        else:
            reset_model_weights(baseline_student_model)

        # compile and train network
        optimizer = SGD(lr=cfg.learning_rate, momentum=0.9, nesterov=True)
        baseline_student_model.compile(optimizer=optimizer,
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])
        chckpnt = log_dir + "/baseline-checkpoint-model.hf5"
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=30, min_delta=0.00007),
            ModelCheckpoint(chckpnt, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        ]
        baseline_student_model.fit(X_train, Y_train,
                          validation_data=(X_test, Y_test),
                          batch_size=128,
                          epochs=cfg.baseline_models_train_epochs,
                          verbose=1,
                          callbacks=callbacks)
        baseline_student_model.load_weights(chckpnt)
        # evaluate and save model weights
        train_acc = baseline_student_model.evaluate(X_train, Y_train, verbose=0)
        val_acc = baseline_student_model.evaluate(X_test, Y_test, verbose=0)
        save_weights(models_dir, baseline_student_model, cfg.student_model_size, i, cfg.num_models_to_train,
                     format(val_acc[1], '.4f'), format(train_acc[1], '.4f'))
        # os.remove(chckpnt)  # remove previous checkpoint


















