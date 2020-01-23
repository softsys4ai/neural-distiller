
# STEP 1
# train the size 10 teacher in 5 epoch intervals
# harvest logits and save the model weights at each interval

# STEP 2
# Distill knowledge to student model for each set of logits

# Step 3
# Evaluate student models for robustness and accuracy

# external dependencies
import os
import glob
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# project imports
from data import load_dataset
from utils import teacher_utils, config_reference as cfg
from models import knowledge_distillation_models


# write teacher weights to file
def save_weights(models_dir, model, curr_epochs, val_acc, train_acc):
    weight_filename = f"model_{curr_epochs}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, weight_filename)
    model.save_weights(model_path)
    return model_path

def save_model(models_dir, model, model_size, curr_epochs, total_epochs, val_acc, train_acc):
    model_filename = f"model_{curr_epochs}_{val_acc}_{train_acc}.h5"
    model_path = os.path.join(models_dir, model_filename)
    model.save(model_path)
    del model
    return model_path

# write teacher logits to file
def save_logits(logits_dir, model_size, curr_epochs, total_epochs, train_logits, test_logits):
    logits_filename = os.path.join(logits_dir, f"logits_{model_size}_{curr_epochs}|{total_epochs}.pkl")
    with open(logits_filename, "wb") as file:
        pickle.dump(train_logits, file)
        pickle.dump(test_logits, file)

def step_decay(epoch):
    lrate = 0.1
    if epoch >= 60:
        lrate /= 5
    if epoch >= 120:
        lrate /= 5
    if epoch >= 160:
        lrate /= 5
    print(f"[INFO] Current learning rate is {lrate}...")
    return lrate

def run():
    print("[INFO] Loading dataset...")
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

    print("[INFO] Creating teacher model...")
    # initialize and save starting network state
    teacher_model = knowledge_distillation_models.get_model(cfg.dataset, cfg.dataset_num_classes, X_train, cfg.teacher_model_size, cfg.model_type)
    optimizer = SGD(lr=cfg.learning_rate, momentum=0.9, nesterov=True)
    teacher_model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
    train_acc = teacher_model.evaluate(X_train, Y_train, verbose=0)[1]
    val_acc = teacher_model.evaluate(X_test, Y_test, verbose=0)[1]
    save_weights(models_dir, teacher_model, 0, val_acc, train_acc)
    # setup training callbacks and train model
    chckpnt = os.path.join(models_dir, "model_{epoch}_{val_accuracy:.5f}_{accuracy:.5f}.h5")
    callbacks = [
        ModelCheckpoint(chckpnt, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True,
                        mode='max')
    ]
    if cfg.USE_TEACHER_LR_SCHEDULER:
        print("[INFO] Using learning rate scheduler...")
        lrate = LearningRateScheduler(step_decay)
        callbacks.append(lrate)
    # train network
    print("[INFO] Training loaded model...")
    if (cfg.use_datagen0):
        teacher_model.fit(datagen.flow(X_train, Y_train, batch_size=128),
                          validation_data=(X_test, Y_test),
                          epochs=cfg.epoch_max0,
                          verbose=1,
                          callbacks=callbacks)
    else:
        teacher_model.fit(X_train, Y_train,
                          validation_data=(X_test, Y_test),
                          batch_size=128,
                          epochs=cfg.epoch_max0,
                          verbose=1,
                          callbacks=callbacks)
    # get all teacher model file names
    TEACHER_DIR_QUERY = os.path.join(models_dir, "*.h5")
    TEACHER_MODEL_PATHS = glob.glob(TEACHER_DIR_QUERY)
    # remove file path to parse information from model names
    teacher_rm_path = models_dir + "/"
    TEACHER_MODEL_NAMES = [x[len(teacher_rm_path):] for x in TEACHER_MODEL_PATHS]
    epochs, val_accs, train_accs = teacher_utils.parse_info_from_teacher_names(TEACHER_MODEL_NAMES)
    print("[INFO] Producing logits with teacher models...")
    # collect logits from all of the teacher models
    print("[INFO] Creating and saving model logits...")
    for i in range(len(TEACHER_MODEL_PATHS)):
        teacher_model.load_weights(TEACHER_MODEL_PATHS[i])
        print(f"[INFO] {i+1}/{len(TEACHER_MODEL_PATHS)}...")
        train_logits, test_logits = teacher_utils.createStudentTrainingData(teacher_model, None, X_train, None, X_test,
                                                                            None)
        save_logits(logits_dir, cfg.teacher_model_size, epochs[i], cfg.epoch_max0, train_logits, test_logits)
    print("[COMPLETE]")



