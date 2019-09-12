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


from Models.LeNet5 import LeNet5Teacher

def config_option_parser(logger):
    # reading command line input
    usage = """USAGE: %python train.py -m [model]
                 Model ResNet50:                  python train.py -m resnet50
                 Model VGG16:                     python train.py -m vgg16
                 Model LeNet-5:                   python train.py -m lenet5
                 Model AlexNet:                   python train.py -m alexnet
                 Model Xception:                  python train.py -m xception
                 Model InceptionV3:               python train.py -m inceptionv3
                 Model CustomTeacher:             python train.py -m custom_teacher
                 Model CustomStudent:             python train.py -m custom_student
        """
    parser = OptionParser(usage=usage)
    parser.add_option('-m', "--model",
                      action="store",
                      type="string",
                      dest="model",
                      help="Type of Model")
    parser.add_option("-g",
                      "--gpus",
                      default=1,
                      type=int,
                      dest="num_gpus",
                      help="# of GPUs to use for training")
    (options, args) = parser.parse_args()
    logger.info("Parsed command line options")
    return (options, usage)

def config_logger():
    """This function is used to configure logging information
    @returns:
        logger: logging object
    """
    # get log directory
    log_dir = os.getcwd() + cfg.log_dir
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%Y_%H:%M:%S")
    log_file_name = "logfile_{0}".format(str(now_datetime))
    log_file = os.path.join(log_dir, log_file_name)

    # get logger object
    ip = socket.gethostbyname(socket.gethostname())
    extra = {"ip_address": ip}
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler(log_file)

    # define log format
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(ip_address)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # define log level
    logger.setLevel(logging.INFO)
    logger = logging.LoggerAdapter(logger, extra)
    now = datetime.now()
    logger.info(cfg.spacer+"NEW TRAINING SESSION STARTED"+cfg.spacer)
    logger.info("Initialized logger")
    return logger

def main():
    # TODO take test type as an input parameter and create switch for different tests
    # logging
    logger = config_logger()
    # command line input
    (options, usage) = config_option_parser(logger)
    teacher_model_name = str(options.model)
    # TODO call experiments to run here based on a switch statement and command line option passed to the config parser
    from Experiments import SimpleStudentTeacher
    SimpleStudentTeacher.run(logger, options)
    logger.info('-- COMPLETE')

    # # loading training data
    # X_train, Y_train, X_test, Y_test = LoadDataset.load_mnist(logger)
    # # config callbacks
    # callbacks = []
    # # setting up teacher model
    # stm = ModelLoader(logger, teacher_model_name)
    # stm.compile_loaded_model()
    # teacher = stm.get_loaded_model()
    # # evaluate teacher accuracy and performance
    # teacherLoss, teacherAcc = HelperUtil.calculate_weighted_score(logger, teacher, X_train, Y_train, X_test, Y_test)
    # logger.info('Teacher weighted score: (acc, loss) --> (%s, %s)' % (teacherAcc, teacherLoss))
    # # retrieving soft targets for student model training
    # Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(teacher, X_train, Y_train, X_test, Y_test)
    #
    # # TODO measure power consumption and inference time
    # # TODO perform all pruning operations here
    # # logger.info("Pruning the teacher network")
    # # studentModels = PruneUtil.prune(logger, teacher.getModel(), X_train, Y_train, X_test, Y_test, len(X_train), cfg.pruning_batch_size, cfg.pruning_epochs, 0.01, 0.1)
    #
    # # training student network at a range of temperatures
    # for temp in cfg.test_temperatures:
    #     logger.info(cfg.student_train_spacer + "NEW STUDENT TRAINING SESSION" + cfg.student_train_spacer)
    #     cfg.temp = temp
    #     # setting up custom student network
    #     ssm = ModelLoader(logger, "custom_student")
    #     student = ssm.get_loaded_model()
    #     # generic pre-KD modification to student network
    #     student = HelperUtil.apply_knowledge_distillation_modifications(logger, student)
    #     # training and evaluating the student model
    #     logger.info('Training student network')
    #     logger.info('Student params: (temperature, epochs, batch_size) --> (%s, %s, %s)' % (cfg.temp, cfg.student_epochs, cfg.student_batch_size))
    #     student.fit(X_train, Y_train_new,
    #                      batch_size=cfg.student_batch_size,
    #                      epochs=cfg.student_epochs,
    #                      verbose=1,
    #                      callbacks=[],
    #                      validation_data=(X_test, Y_test_new))
    #     logger.info('Completed student network training')
    #     # generic reversal of pre-KD modification to student network
    #     finalStudent = HelperUtil.revert_knowledge_distillation_modifications(logger, student)
    #     # evaluating student performance
    #     studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, finalStudent, X_train, Y_train, X_test, Y_test)
    #     logger.info('Student weighted score: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    #
    #
    # # training student network at a range of temperatures
    # for temp in cfg.test_temperatures:
    #     logger.info(cfg.student_train_spacer + "NEW STUDENT TRAINING SESSION" + cfg.student_train_spacer)
    #     cfg.temp = temp
    #     # setting up custom student network
    #     ssm = ModelLoader(logger, "custom_student")
    #     student = ssm.get_loaded_model()
    #     # generic pre-KD modification to student network
    #     student = HelperUtil.apply_knowledge_distillation_modifications(logger, student)
    #     # training and evaluating the student model
    #     logger.info('Training student network')
    #     logger.info('Student params: (temperature, epochs, batch_size) --> (%s, %s, %s)' % (cfg.temp, cfg.student_epochs, cfg.student_batch_size))
    #     student.fit(X_train, Y_train_new,
    #                      batch_size=cfg.student_batch_size,
    #                      epochs=cfg.student_epochs,
    #                      verbose=1,
    #                      callbacks=[],
    #                      validation_data=(X_test, Y_test_new))
    #     logger.info('Completed student network training')
    #     # generic reversal of pre-KD modification to student network
    #     finalStudent = HelperUtil.revert_knowledge_distillation_modifications(logger, student)
    #     # evaluating student performance
    #     studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, finalStudent, X_train, Y_train, X_test, Y_test)
    #     logger.info('Student weighted score: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    #
    #     logger.info('-- COMPLETE')
if __name__ == "__main__":
    main()