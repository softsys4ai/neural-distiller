import os
import socket
import logging
import sys
import numpy as np
from optparse import OptionParser
from datetime import datetime
from Configuration import Config as cfg
from Data import LoadDataset
from Models.ModelLoader import ModelLoader
from Models import TeacherUtils
from Utils import HelperUtil
from Pruning import PruneUtil
from numpy.random import seed
seed(cfg.random_seed)
from tensorflow import set_random_seed
set_random_seed(cfg.random_seed)

def run(logger, options):
    teacher_model_name = str(options.model)
    # loading training data
    X_train, Y_train, X_test, Y_test = LoadDataset.load_mnist(logger)
    # config callbacks
    callbacks = []
    # setting up teacher model
    stm = ModelLoader(logger, teacher_model_name)
    stm.compile_loaded_model()
    teacher = stm.get_loaded_model()
    # evaluate teacher accuracy and performance
    teacherLoss, teacherAcc = HelperUtil.calculate_weighted_score(logger, teacher, X_train, Y_train, X_test, Y_test)
    logger.info('Teacher weighted score: (acc, loss) --> (%s, %s)' % (teacherAcc, teacherLoss))
    # retrieving soft targets for student model training
    Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(teacher, X_train, Y_train, X_test, Y_test)

    # TODO implement student network in a variety of sizes
    # TODO perform simple incremental learning

    logger.info(cfg.student_train_spacer + "NEW STUDENT TRAINING SESSION" + cfg.student_train_spacer)
    cfg.temp = temp
    # setting up custom student network
    ssm = ModelLoader(logger, "custom_student")
    student = ssm.get_loaded_model()
    # generic pre-KD modification to student network
    student = HelperUtil.apply_knowledge_distillation_modifications(logger, student)
    # training and evaluating the student model
    logger.info('Training student network')
    logger.info('Student params: (temperature, epochs, batch_size) --> (%s, %s, %s)' % (
    cfg.temp, cfg.student_epochs, cfg.student_batch_size))
    student.fit(X_train, Y_train_new,
                batch_size=cfg.student_batch_size,
                epochs=cfg.student_epochs,
                verbose=1,
                callbacks=[],
                validation_data=(X_test, Y_test_new))
    logger.info('Completed student network training')
    # generic reversal of pre-KD modification to student network
    student = HelperUtil.revert_knowledge_distillation_modifications(logger, student)
    # evaluating student performance
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, student, X_train, Y_train,
                                                                  X_test, Y_test)
    logger.info('Student weighted score: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))