import sys
import traceback
import logging
sys.path.append("..")  # Adds higher directory to python modules path.
from Configuration import Config as cfg
from Data import LoadDataset
from Models import ModelLoader
from Models import TeacherUtils
from Utils import HelperUtil
import datetime
import json
import ast
import os
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPooling2D, Dense, Flatten, Activation, Conv2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.losses import categorical_crossentropy as logloss
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import adadelta
from numpy.random import seed
from tensorflow import set_random_seed

set_random_seed(cfg.random_seed)
seed(cfg.random_seed)
nb_classes = 10


def import_config(config_file_path):
    with open(config_file_path, 'r') as f:
        configuration = json.load(f)
    return configuration


# TODO include best train and validation accuracies, may be more telling
def create_result(netSize, temp, alpha, train_score, val_score):
    result_obj = {"result": []}  # empty space for our experiment1's data
    result = {}
    result["date_time"] = str(datetime.datetime.now())
    result["net_size"] = str(netSize)
    result["temp"] = str(temp)
    result["alpha"] = str(alpha)
    result["val_acc"] = str(val_score[1])
    result["acc"] = str(train_score[1])
    result["val_loss"] = str(val_score[0])
    result["loss"] = str(train_score[0])
    result_obj['result'].append(result)
    return result_obj


def create_meta(teacher_name, epochs, temp, alpha, order):
    metadata_obj = {"metadata": []}  # empty space for our experiment1's data
    metadata = {}
    metadata["date_time"] = str(datetime.datetime.now())
    metadata["teacher_name"] = str(teacher_name)
    metadata["epochs"] = str(epochs)
    metadata["temp"] = str(temp)
    metadata["alpha"] = str(alpha)
    metadata['order'] = str(order)
    metadata_obj['metadata'].append(metadata)
    return metadata_obj


def run(logger, options):
    logger.info(cfg.student_train_spacer + "GENERIC MULTISTAGE" + cfg.student_train_spacer)

    # session file setup
    session_file_name = "experiments_" + datetime.datetime.now().isoformat() + ".log"
    log_path = ".." + cfg.log_dir
    session_file_relative_path = log_path + session_file_name
    my_path = os.path.abspath(os.path.dirname(__file__))
    session_log_file = os.path.join(my_path, session_file_relative_path)
    with open(session_log_file, "w") as f:
        f.write("begin test: " + datetime.datetime.now().isoformat() + "\n")
        f.close()
    temporary_teacher_model_file = cfg.temp_experiment_configs_dir + "/" + cfg.temp_serialized_net

    # load configuration file
    configuration = import_config(options.config_file_path)
    teacher_name = configuration['teacher_name']
    epochs = configuration['epochs']
    temperatures = configuration['temp_config']
    alphas = configuration['alpha_config']
    order_combinations = configuration['size_combinations']

    # loading training data
    X_train, Y_train, X_test, Y_test = LoadDataset.load_mnist(logger)
    try:
        for order in order_combinations:
            for alpha in alphas:
                for temp in temperatures:
                    logger.info("Clearing tensorflow/keras backend session...")
                    tf.keras.backend.clear_session()  # must clear the current session to free memory!
                    # with tf.Graph().as_default():
                    previousModel = None
                    if teacher_name is not None:
                        ssm = ModelLoader(logger, options.teacherModel)
                        previousModel = ssm.get_loaded_model()
                        teacher_name = options.teacherModel
                    # creating experiment1 metadata
                    experiment_result = {"experiment_results": []}  # empty space for our experiment1's data
                    experiment_metadata = create_meta(teacher_name, epochs, temp, alpha, order)
                    # experiment_result["experiment_results"].append(experiment_metadata) # TODO add back after resolving memory leak
                    # performing experiment1 on given size, alpha, and temperature combination
                    for net_size in order:
                        model = None
                        # setting up model based on size
                        if net_size == 10:
                            # model = Sequential([
                            #     Conv2D(256, kernel_size=(3, 3),
                            #            activation='relu',
                            #            input_shape=X_train.shape[1:]),
                            #     Conv2D(128, (3, 3), activation='relu'),
                            #     MaxPooling2D(pool_size=(2, 2)),
                            #     Conv2D(64, (3, 3), activation='relu'),
                            #     Conv2D(32, (3, 3), activation='relu'),
                            #     MaxPooling2D(pool_size=(2, 2)),
                            #     Conv2D(16, (3, 3), activation='relu'),
                            #     MaxPooling2D(pool_size=(2, 2)),
                            #     Flatten(),
                            #     Dense(128, activation='relu'),
                            #     Dense(cfg.mnist_number_classes, name='logits'),
                            #     Activation('softmax')  # Note that we add a normal softmax layer to begin with
                            # ])
                            logger.info("--------------------------------------------------------------------------------")
                            logger.info("loading pre-trained teacher")
                            previousModel = load_model('size_10_teacher.h5')
                            continue
                        elif net_size == 8:
                            model = Sequential([
                                Conv2D(128, kernel_size=(3, 3),
                                       activation='relu',
                                       input_shape=X_train.shape[1:]),
                                MaxPooling2D(pool_size=(2, 2)),
                                Conv2D(64, (3, 3), activation='relu'),
                                Conv2D(32, (3, 3), activation='relu'),
                                MaxPooling2D(pool_size=(2, 2)),
                                Conv2D(16, (3, 3), activation='relu'),
                                # Conv2D(8, (3, 3), activation='relu'),
                                MaxPooling2D(pool_size=(2, 2)),
                                Flatten(),
                                Dense(96, activation='relu'),
                                Dense(cfg.mnist_number_classes, name='logits'),
                                Activation('softmax')  # Note that we add a normal softmax layer to begin with
                            ])
                        elif net_size == 6:
                            model = Sequential([
                                Conv2D(64, kernel_size=(3, 3),
                                       activation='relu',
                                       input_shape=X_train.shape[1:]),
                                Conv2D(32, (3, 3), activation='relu'),
                                MaxPooling2D(pool_size=(2, 2)),
                                Conv2D(16, (3, 3), activation='relu'),
                                # Conv2D(8, (3, 3), activation='relu'),
                                MaxPooling2D(pool_size=(2, 2)),
                                Flatten(),
                                Dense(64, activation='relu'),
                                Dense(cfg.mnist_number_classes, name='logits'),
                                Activation('softmax')  # Note that we add a normal softmax layer to begin with
                            ])
                            # model = load_model(cfg.teacher_model_dir + "/best_size_6_model.hdf5")
                            # previousModel = model
                            # continue
                        elif net_size == 4:
                            model = Sequential([
                                Conv2D(32, kernel_size=(3, 3),
                                       activation='relu',
                                       input_shape=X_train.shape[1:]),
                                MaxPooling2D(pool_size=(2, 2)),
                                Conv2D(16, (3, 3), activation='relu'),
                                # Conv2D(8, (3, 3), activation='relu'),
                                MaxPooling2D(pool_size=(2, 2)),
                                Flatten(),
                                Dense(128, activation='relu'),
                                Dense(cfg.mnist_number_classes, name='logits'),
                                Activation('softmax')  # Note that we add a normal softmax layer to begin with
                            ])
                            # model = load_model(cfg.teacher_model_dir + "/best_size_4_model.hdf5")
                            # previousModel = model
                            # continue
                        elif net_size == 2:
                            model = Sequential([
                                Conv2D(16, kernel_size=(3, 3),
                                       activation='relu',
                                       input_shape=X_train.shape[1:]),
                                MaxPooling2D(pool_size=(2, 2)),
                                Flatten(),
                                Dense(16, activation='relu'),
                                Dense(cfg.mnist_number_classes, name='logits'),
                                Activation('softmax')  # Note that we add a normal softmax layer to begin with
                            ])
                        else:
                            raise Exception(
                                'The given net size is not a possible network. Given net size was: {}'.format(net_size))

                        # perform KD if there is a previously trained model to work with
                        if previousModel is not None:
                            # load model config from disc to avoid any weird errors
                            # previousModel = load_model(temporary_teacher_model_file)
                            # logger.info("evaluating teacher...")
                            # train_score, val_score = HelperUtil.calculate_unweighted_score(logger, previousModel, X_train,
                            #                                                                Y_train, X_test, Y_test)
                            # logger.info("Teacher scores: %s, %s" % (val_score, train_score))
                            # train with KD
                            logger.info("creating soft targets for student...")
                            Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(previousModel, temp, X_train,
                                                                                             Y_train, X_test, Y_test)
                            logger.info("completed")
                            model = HelperUtil.apply_knowledge_distillation_modifications(logger, model, temp)
                            # model = multi_gpu_model(model, gpus=4)
                            model.compile(
                                optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                                loss=lambda y_true, y_pred: HelperUtil.knowledge_distillation_loss(logger, y_true, y_pred, alpha),
                                metrics=[HelperUtil.acc])
                            logger.info("training model...\norder:%s\nsize:%d\ntemp:%d\nalpha:%d" % (order, net_size, temp, alpha))
                            model.fit(X_train, Y_train_new,
                                      batch_size=cfg.student_batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=cfg.student_callbacks,
                                      validation_data=(X_test, Y_test_new))
                            model = HelperUtil.revert_knowledge_distillation_modifications(logger, model)
                            # train_score, val_score = HelperUtil.calculate_unweighted_score(logger, model, X_train, Y_train,
                            #                                                                X_test, Y_test)
                            model.compile(optimizer=cfg.student_optimizer,
                                          loss='categorical_crossentropy',
                                          metrics=['accuracy'])
                            train_score = model.evaluate(X_train, Y_train, verbose=0)
                            val_score = model.evaluate(X_test, Y_test, verbose=0)
                            result = create_result(net_size, temp, alpha, train_score, val_score)
                            logger.info(result)
                            experiment_result["experiment_results"].append(result)

                        # if no previously trained model, train the network
                        else:
                            model.compile(optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                                          loss=logloss,  # the same as the custom loss function
                                          metrics=['accuracy'])
                            # callbacks = cfg.student_callbacks
                            # callbacks.append(ModelCheckpoint(filepath=cfg.teacher_model_dir + "/best_size_" + str(netSize) + "_model.hdf5", save_best_only=True))
                            model.fit(X_train, Y_train,
                                      batch_size=cfg.student_batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=cfg.student_callbacks,
                                      validation_data=(X_test, Y_test))
                            train_score, val_score = HelperUtil.calculate_unweighted_score(logger, model, X_train, Y_train,
                                                                                           X_test, Y_test)
                            # append current trained network result to current experiment1 result object
                            result = create_result(net_size, temp, alpha, train_score, val_score)
                            logger.info(result)
                            # experiment_result["experiment_results"].append(result) # TODO add back after resolving memory leak
                            # model.save('size_10_teacher.h5')  # creates a HDF5 file 'my_model.h5'

                            # returns a compiled model
                            # identical to the previous one
                            # model = load_model('my_model.h5')

                        # temporarily serialize model to load as teacher in following KD training to avoid errors
                        del previousModel # free memory
                        previousModel = model  # previously trained model becomes teacher
                        # model.save(temporary_teacher_model_file)
                    # appending experiment1 result to log file
                    if os.path.exists(session_log_file):
                        open_type = 'a'
                    else:
                        open_type = 'w'
                    with open(session_log_file, open_type) as f:
                        f.write(json.dumps(experiment_result))
                        f.write("\n")
                        f.close()
                    # printing the results of training
                    logger.info(cfg.student_train_spacer)
    except Exception:
        traceback.print_exc()
        error = traceback.format_exc()
        # error.upper()
        logging.error('Error encountered: ', exc_info=True)