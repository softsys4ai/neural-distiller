
# STEP 1
# train the size 10 teacher in 5 epoch intervals
# harvest logits and save the model weights at each interval

# STEP 2
# Distill knowledge to student model for each set of logits

# Step 3
# Evaluate student models for robustness and accuracy

# external dependencies
import os
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# project imports
from data import load_dataset
from utils import teacher_utils, config_reference as cfg
from models import knowledge_distillation_models


# write teacher weights to file
def save_weights(models_dir, model, model_size, curr_epochs, total_epochs, val_acc, train_acc):
    weight_filename = f"model_{model_size}_{curr_epochs}|{total_epochs}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)
    return model_path


# write teacher logits to file
def save_logits(logits_dir, model_size, curr_epochs, total_epochs, train_logits, test_logits):
    logits_filename = os.path.join(logits_dir, f"logits_{model_size}_{curr_epochs}|{total_epochs}.pkl")
    with open(logits_filename, "wb") as file:
        pickle.dump(train_logits, file)
        pickle.dump(test_logits, file)

def run():
    X_train, Y_train, X_test, Y_test = load_dataset.load_dataset_by_name(None, cfg.dataset)
    X_train, X_test = load_dataset.z_standardization(X_train, X_test)

    # use when debugging
    if cfg.debug0 is True:
        X_train = X_train[:128]
        Y_train = Y_train[:128]
        X_test = X_test[:128]
        Y_test = Y_test[:128]

    # experiment directory structure
    # ESKD_experiment_{datetime}
    # saved_model_weights
    # saved_weights

    # create experiment run directory for each session's model weights and logits
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
    cfg.log_dir = os.path.join(cfg.log_dir, "ESKD_Logit_Harvesting_" + cfg.dataset + f"_{cfg.teacher_model_size}_" + now_datetime)
    os.mkdir(cfg.log_dir)
    logits_dir = os.path.join(cfg.log_dir, "logits")
    os.mkdir(logits_dir)
    models_dir = os.path.join(cfg.log_dir, "models")
    os.mkdir(models_dir)

    if (cfg.use_datagen0):
        datagen = ImageDataGenerator(
            # set input mean to 0 over the cfg.dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of cfg.dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        datagen.fit(X_train)

    # initialize and save starting network state
    teacher_model = knowledge_distillation_models.get_model(cfg.dataset, cfg.dataset_num_classes, X_train, cfg.teacher_model_size, cfg.model_type)
    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    teacher_model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
    train_acc = teacher_model.evaluate(X_train, Y_train, verbose=0)
    val_acc = teacher_model.evaluate(X_test, Y_test, verbose=0)
    prev_model_path = save_weights(models_dir, teacher_model, cfg.teacher_model_size, 0, cfg.epoch_max0,
                                   format(val_acc[1], '.4f'), format(train_acc[1], '.4f'))
    train_logits, test_logits = teacher_utils.createStudentTrainingData(teacher_model, None, X_train, None, X_test, None)
    save_logits(logits_dir, cfg.teacher_model_size, 0, cfg.epoch_max0, train_logits, test_logits)

    # load model for current iteration
    teacher_model = knowledge_distillation_models.get_model(cfg.dataset, cfg.dataset_num_classes, X_train, cfg.teacher_model_size, cfg.model_type)
    # compile network for training
    learning_rates = [0.1, 0.2, 0.004, 0.0008]
    learning_rate_epochs = [0, 60, 120, 160]
    optimizer = SGD(lr=learning_rates[0], momentum=0.9, nesterov=True, decay=5e-4)
    teacher_model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

    # intermittent training and harvesting of logits for ESKD experiment
    for i in range(1, len(cfg.epoch_intervals0)):

        # setup current iteration params and load model
        curr_epochs = cfg.epoch_intervals0[i]
        print(f"Training size {cfg.teacher_model_size} teacher network {curr_epochs}|{cfg.epoch_max0}")

        # # clear current session to free memory
        try:
            idx = learning_rate_epochs.index(curr_epochs)
            print("[INFO] Updating learning rate!")
            tf.keras.backend.clear_session()
            print("[INFO] Re-building network!")
            teacher_model = knowledge_distillation_models.get_model(cfg.dataset, cfg.dataset_num_classes, X_train, cfg.teacher_model_size, cfg.model_type)
            optimizer = SGD(lr=learning_rates[idx], momentum=0.9, nesterov=True, decay=5e-4)
            teacher_model.compile(optimizer=optimizer,
                                  loss="categorical_crossentropy",
                                  metrics=["accuracy"])
            print("[INFO] done!")
        except:
            print("[INFO] lr unchanged")


        teacher_model.load_weights(prev_model_path)

        # train network
        if (cfg.use_datagen0):
            teacher_model.fit(datagen.flow(X_train, Y_train, batch_size=128),
                              validation_data=(X_test, Y_test),
                              epochs=cfg.interval_size0,
                              verbose=1,
                              callbacks=None)
        else:
            teacher_model.fit(X_train, Y_train,
                              validation_data=(X_test, Y_test),
                              batch_size=128,
                              epochs=cfg.interval_size0,
                              verbose=1,
                              callbacks=None)

        # evaluate and save model weights
        train_acc = teacher_model.evaluate(X_train, Y_train, verbose=0)
        val_acc = teacher_model.evaluate(X_test, Y_test, verbose=0)
        prev_model_path = save_weights(models_dir, teacher_model, cfg.teacher_model_size, curr_epochs, cfg.epoch_max0,
                                       format(val_acc[1], '.4f'), format(train_acc[1], '.4f'))
        # create and save logits from model
        train_logits, test_logits = teacher_utils.createStudentTrainingData(teacher_model, None, X_train, None, X_test, None)
        save_logits(logits_dir, cfg.teacher_model_size, curr_epochs, cfg.epoch_max0, train_logits, test_logits)

        # delete the model for the next training iteration
        # del teacher_model
        # del optimizer

