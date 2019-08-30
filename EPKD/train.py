import os
import socket
import logging
import sys
from optparse import OptionParser
from datetime import datetime
from Configuration import Config as cfg
from Data import LoadDataset
from Models.ModelLoader import ModelLoader
from Models import TeacherUtils
from Pruning import PruneUtil

def config_option_parser():
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
    return (options, usage)

def config_logger():
    """This function is used to configure logging information
    @returns:
        logger: logging object
    """
    # get log directory
    log_dir = os.getcwd() + cfg.log_dir
    log_file_name = "logfile_{0}".format(str(datetime.now().date()))
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
    logger.info("[STATUS]: Start RunTest")
    return logger

def main():
    # command line input
    (options, usage) = config_option_parser()
    teacher_model_name = str(options.model)
    # logging
    logger = config_logger()
    # loading training data
    X_train, Y_train, X_test, Y_test = LoadDataset.load_mnist()
    # config callbacks
    callbacks = []
    # setting up teacher model
    stm = ModelLoader(logger, teacher_model_name)
    teacher = stm.get_loaded_model()

    # TODO perform all pruning operations here
    # print("[INFO] Attempting to prune teacher network")
    # studentModel = PruneUtil.prune(teacher.getModel(), X_train, Y_train, X_test, Y_test, len(X_train), cfg.pruning_batch_size, cfg.pruning_epochs, 0.01, 0.1)

    # retreiving soft targets for student model training
    Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(teacher, X_train, Y_train, X_test, Y_test)
    # setting up custom student network
    ssm = ModelLoader(logger, "custom_student")
    student = ssm.get_loaded_model()
    # training and evaluating the student model
    student.fit(X_train, Y_train_new,
                     batch_size=cfg.student_batch_size,
                     epochs=cfg.student_epochs,
                     verbose=1,
                     callbacks=[],
                     validation_data=(X_test, Y_test_new))
    score = student.evaluate(X_test, Y_test_new, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('-- done')

if __name__ == "__main__":
    main()