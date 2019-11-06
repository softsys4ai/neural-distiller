import sys
import traceback
import logging
import pickle
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
from tensorflow.python.keras.backend import clear_session
import numpy as np

tf.set_random_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)

def import_config(config_file_path):
    with open(config_file_path, 'r') as f:
        configuration = json.load(f)
    return configuration


# TODO include best train and validation accuracies, may be more telling
def create_result(netSize, temp, alpha, train_score, val_score):
    result = {}
    result["date_time"] = str(datetime.datetime.now())
    result["net_size"] = str(netSize)
    result["temp"] = str(temp)
    result["alpha"] = str(alpha)
    result["val_acc"] = str(val_score[1])
    result["acc"] = str(train_score[1])
    result["val_loss"] = str(val_score[0])
    result["loss"] = str(train_score[0])
    return result


def create_meta(dataset, teacher_name, epochs, temp, alpha, order):
    metadata = {}
    metadata["date_time"] = str(datetime.datetime.now())
    metadata["dataset"] = str(dataset)
    metadata["teacher_name"] = str(teacher_name)
    metadata["epochs"] = str(epochs)
    metadata["temp"] = str(temp)
    metadata["alpha"] = str(alpha)
    metadata['order'] = str(order)
    return metadata


def get_model(dataset, numClasses, X_train, net_size):
    if dataset is "mnist":
        return get_model_mnist(numClasses, X_train, net_size)
    elif dataset is "cifar100":
        return get_model_cifar100(numClasses, X_train, net_size)

def get_model_mnist(numClasses, X_train, net_size):
    # setting up model based on size
    if net_size == 10:
        model = Sequential([
            Conv2D(256, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
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
            Dense(numClasses, name='logits'),
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
            Dense(numClasses, name='logits'),
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
            Dense(32, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_4_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 2:
        # model = Sequential([
        #     Conv2D(8, kernel_size=(3, 3),
        #            activation='relu',
        #            input_shape=X_train.shape[1:]),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(4, (3, 3), activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Flatten(),
        #     Dense(16, input_shape=X_train.shape[1:]),
        #     Activation('relu'),
        #     Dense(numClasses, name='logits'),
        #     Activation('softmax'),
        # ])
        model = Sequential([
            Dense(16, activation='relu', input_shape=X_train.shape[1:]),
            Dense(16, activation='relu', input_shape=X_train.shape[1:]),
            Activation('relu'),
            Flatten(),
            Dense(cfg.mnist_number_classes, name='logits'),
            Activation('softmax'),
        ])
    else:
        print('no model available for given size!')
    return model

def get_model_cifar100(numClasses, X_train, net_size):
    # setting up model based on size
    if net_size == 10:
        model = Sequential([
            Conv2D(256, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
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
            Dense(numClasses, name='logits'),
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
            Dense(numClasses, name='logits'),
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
            Dense(32, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_4_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 2:
        # model = Sequential([
        #     Conv2D(8, kernel_size=(3, 3),
        #            activation='relu',
        #            input_shape=X_train.shape[1:]),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(4, (3, 3), activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Flatten(),
        #     Dense(16, input_shape=X_train.shape[1:]),
        #     Activation('relu'),
        #     Dense(numClasses, name='logits'),
        #     Activation('softmax'),
        # ])
        model = Sequential([
            Dense(16, activation='relu', input_shape=X_train.shape[1:]),
            Dense(16, activation='relu', input_shape=X_train.shape[1:]),
            Activation('relu'),
            Flatten(),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
    else:
        print('no model available for given size!')
    return model

# method to check for already saved copy of teacher knowledge
def get_pretrained_teacher_logits(netSize, dataset):
    # load pre-created soft targets for teacher
    logitFileName = os.path.join(cfg.soft_targets_dir, str(dataset)+str(netSize)+"_soft_targets.pkl")
    if os.path.exists(logitFileName): # check for logit file existence
        filehandler = open(logitFileName, 'rb')
        Y_train_new = pickle.load(filehandler)
        Y_test_new = pickle.load(filehandler)
        return Y_train_new, Y_test_new
    else:
        return None, None

def save_pretrained_teacher_logits(netSize, Y_train_new, Y_test_new, dataset):
    logitFileName = os.path.join(cfg.soft_targets_dir, str(dataset)+str(netSize)+"_soft_targets.pkl")
    filehandler = open(logitFileName, 'wb')
    pickle.dump(Y_train_new, filehandler)
    pickle.dump(Y_test_new, filehandler)

def run(logger, options):
    logger.info(cfg.student_train_spacer + "GENERIC MULTISTAGE" + cfg.student_train_spacer)

    # session file setup
    session_file_name = cfg.dataset + "_grid_search_experiment_" + datetime.datetime.now().isoformat() + ".log"
    log_path = ".." + cfg.log_dir
    session_file_relative_path = log_path + session_file_name
    my_path = os.path.abspath(os.path.dirname(__file__))
    session_log_file = os.path.join(my_path, session_file_relative_path)
    with open(session_log_file, "w") as f:
        f.write("begin test: " + datetime.datetime.now().isoformat() + "\n")
        f.close()

    # load configuration file
    configuration = import_config(options.config_file_path)
    teacher_name = configuration['teacher_name']
    epochs = configuration['epochs']
    temperatures = configuration['temp_config']
    alphas = configuration['alpha_config']
    order_combinations = configuration['size_combinations']

    # loading training data
    X_train, Y_train, X_test, Y_test = LoadDataset.load_dataset_by_name(logger, cfg.dataset)
    try:
        for order in order_combinations:
            for alpha in alphas:
                for temp in temperatures:
                    tf.keras.backend.clear_session()  # must clear the current session to free memory!
                    logger.info("Clearing tensorflow/keras backend session and de-allocating remaining models...")
                    model = None
                    previousModel = None
                    if teacher_name is not None:
                        ssm = ModelLoader(logger, options.teacherModel)
                        previousModel = ssm.get_loaded_model()
                        teacher_name = options.teacherModel
                    # creating experiment1 metadata
                    experiment_result = {"experiment_results": []}  # empty space for our experiment1's data
                    experiment_metadata = create_meta(cfg.dataset, teacher_name, epochs, temp, alpha, order)
                    experiment_result['metadata'] = experiment_metadata
                    # performing experiment on given size, alpha, and temperature combination
                    for net_size in order:
                        model = None
                        # perform KD if there is a previously trained model to work with
                        if previousModel is not None:
                            model = get_model(cfg.dataset, cfg.dataset_num_classes, X_train, net_size)
                            logger.info("loading soft targets for student training...")
                            Y_train_new, Y_test_new = get_pretrained_teacher_logits(previousModel, cfg.dataset)
                            logger.info("completed")
                            # filehandler = open("mnist_10_soft_targets.pkl", 'wb')
                            # pickle.dump(Y_train_new, filehandler)
                            # pickle.dump(Y_test_new, filehandler)
                            model = HelperUtil.apply_knowledge_distillation_modifications(logger, model, temp)
                            # model.summary()
                            # model = multi_gpu_model(model, gpus=4)
                            model.compile(
                                optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                                loss=lambda y_true, y_pred: HelperUtil.knowledge_distillation_loss(logger, y_true, y_pred, alpha),
                                metrics=[HelperUtil.acc])
                            logger.info("training model...\norder:%s\nsize:%d\ntemp:%d\nalpha:%f" % (order, net_size, temp, alpha))
                            model.fit(X_train, Y_train_new,
                                      batch_size=cfg.student_batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=cfg.student_callbacks,
                                      validation_data=(X_test, Y_test_new))
                            # model = HelperUtil.revert_knowledge_distillation_modifications(logger, model)
                            del model
                            # train_score, val_score = HelperUtil.calculate_unweighted_score(logger, model, X_train, Y_train,
                            #                                                                X_test, Y_test)
                            model = get_model(cfg.dataset, cfg.dataset_num_classes, X_train, net_size)
                            # load best model from checkpoint for evaluation
                            model.load_weights(cfg.checkpoint_path)
                            model.compile(optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                                          loss=logloss,  # the same as the custom loss function
                                          metrics=['accuracy'])
                            train_score = model.evaluate(X_train, Y_train, verbose=0)
                            val_score = model.evaluate(X_test, Y_test, verbose=0)
                            result = create_result(net_size, temp, alpha, train_score, val_score)
                            logger.info(result)
                            experiment_result["experiment_results"].append(result)
                            # # remove checkpoint of best model for new checkpoint
                            # os.remove(cfg.checkpoint_path)
                            # save soft targets
                            Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(model, temp, X_train, Y_train, X_test, Y_test)
                            save_pretrained_teacher_logits(net_size, Y_train_new, Y_test_new, cfg.dataset)
                            # clear soft targets
                            Y_train_new = None
                            Y_test_new = None
                            # set model to current net size to preserve in previousModel
                            model = net_size
                        # if no previously trained model, train the network
                        else:
                            # load the already created soft targets
                            Y_train_new, Y_test_new = get_pretrained_teacher_logits(net_size, cfg.dataset)
                            # train network if not previously created logits
                            if Y_train_new is None or Y_test_new is None:
                                logger.info("training teacher model...\norder:%s\nsize:%d\ntemp:%d\nalpha:%f" % (
                                order, net_size, temp, alpha))
                                model = get_model(cfg.dataset, cfg.dataset_num_classes, X_train, net_size)
                                model.compile(optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                                              loss=logloss,  # the same as the custom loss function
                                              metrics=['accuracy'])
                                # train network and save model with bet validation accuracy to cfg.checkpoint_path
                                model.fit(X_train, Y_train,
                                          validation_data=(X_test, Y_test),
                                          batch_size=cfg.student_batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          callbacks=cfg.student_callbacks)
                                # load best model from checkpoint for evaluation
                                del model
                                model = get_model(cfg.dataset, cfg.dataset_num_classes, X_train, net_size)
                                model.load_weights(cfg.checkpoint_path)
                                model.compile(optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                                              loss=logloss,  # the same as the custom loss function
                                              metrics=['accuracy'])
                                # evaluate network
                                train_score, val_score = HelperUtil.calculate_unweighted_score(logger, model, X_train,
                                                                                               Y_train,
                                                                                               X_test, Y_test)
                                # save evaluation
                                result = create_result(net_size, temp, alpha, train_score, val_score)
                                logger.info(result)
                                experiment_result["experiment_results"].append(result)
                                Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(model, temp, X_train,
                                                                                                 Y_train, X_test,
                                                                                                 Y_test)
                                save_pretrained_teacher_logits(net_size, Y_train_new, Y_test_new, cfg.dataset)
                                # # remove checkpoint of best model for new checkpoint
                                # os.remove(cfg.checkpoint_path)
                            else:
                                model = net_size
                        # temporarily serialize model to load as teacher in following KD training to avoid errors
                        del previousModel # free memory
                        previousModel = model  # previously trained model becomes teacher

                    # appending experiment result to log file
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
                    # free model variables for next configuration iteration
                    del model
                    del previousModel

    except Exception:
        traceback.print_exc()
        error = traceback.format_exc()
        # error.upper()
        logging.error('Error encountered: %s' % error, exc_info=True)