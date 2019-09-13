import os
import socket
import logging
import sys
from optparse import OptionParser
from datetime import datetime
from Configuration import Config as cfg
from numpy.random import seed
seed(cfg.random_seed)
from tensorflow import set_random_seed
from Experiments import NoStepStudent, OneStepStudentTeacher, TwoStepIntermediate, ThreeStepIntermediate, ComplexIntermediateStudent
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
    parser.add_option('-e', "--experiment",
                      action="store",
                      type="string",
                      dest="experiment",
                      help="Type of experiment to execute")
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
    # logging
    logger = config_logger()
    # command line input
    (options, usage) = config_option_parser(logger)
    if options.experiment == "no-step-student":
        NoStepStudent.run(logger, options)
    elif options.experiment == "one-step-student-teacher":
        OneStepStudentTeacher.run(logger, options)
    elif options.experiment == "two-step-student-teacher":
        TwoStepIntermediate.run(logger, options)
    elif options.experiment == "three-step-student-teacher":
        ThreeStepIntermediate.run(logger, options)
    elif options.experiment == "complex-intermediate":
        logger.error("Provided xperiment type not implemented!")
        #ComplexIntermediateStudent(logger, options)
    else:
        logger.error("Provided experiment type not supported!")
        return
    logger.info('-- COMPLETE')

    # TODO measure power consumption and inference time
    # TODO perform all pruning operations here
    # logger.info("Pruning the teacher network")
    # studentModels = PruneUtil.prune(logger, teacher.getModel(), X_train, Y_train, X_test, Y_test, len(X_train), cfg.pruning_batch_size, cfg.pruning_epochs, 0.01, 0.1)


if __name__ == "__main__":
    main()