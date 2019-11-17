import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="2";
import socket
import logging
import sys
from optparse import OptionParser
from datetime import datetime
from Configuration import Config as cfg
from Experiments import GenericMultistageKD_CNN

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
    parser.add_option('-e', "--experiment",
                      action="store",
                      type="string",
                      dest="experiment",
                      help="Type of experiment1 to execute")
    parser.add_option('-t', "--teacher-model",
                      action="store",
                      type="string",
                      default=None,
                      dest="teacherModel",
                      help="Type of Model")
    parser.add_option('-c', "--config-file-path",
                      action="store",
                      type="string",
                      default=None,
                      dest="config_file_path",
                      help="Type of Model")
    (options, args) = parser.parse_args()
    return (options, usage)

# TODO add separate logger path for experimental results
def config_logger(log_file):
    """This function is used to configure logging information
    @returns:
        logger: logging object
    """

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
    logger.info(cfg.spacer + "NEW TRAINING SESSION STARTED" + cfg.spacer)
    logger.info("Initialized logger")
    return logger


def main():
    # command line input
    (options, usage) = config_option_parser()

    # get log directory
    log_dir = os.getcwd() + cfg.log_dir
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%Y_%H:%M:%S")
    if options.experiment == "search-alpha-temp-configurations":
        log_dir = log_dir + "/" + cfg.dataset + "_grid_search_" + now_datetime
        os.mkdir(log_dir)
    log_file_name = "experiment_log"
    log_file = os.path.join(log_dir, log_file_name)

    # add high priority
    os.nice(1)
    # logging
    logger = config_logger(log_file)

    try:
        if options.experiment == "search-alpha-temp-configurations":
            session_log_file = log_dir + "/training_session.log"
            GenericMultistageKD_CNN.run(logger, options, session_log_file)
        elif options.experiment == "full-PaKD-compression":
            logger.error("Provided experiment1 type not yet implemented!")
            # TODO measure power consumption and inference time
            # TODO perform all pruning operations
            # TODO pruning, ex: studentModels = PruneUtil.prune(logger, teacher.getModel(), X_train, Y_train, X_test, Y_test, len(X_train), cfg.pruning_batch_size, cfg.pruning_epochs, 0.01, 0.1)
        else:
            logger.error("Provided experiment1 type not supported!")
            return
    except Exception as e:
        logger.error("Error while running experiment1: {0}".format(str(e)))


if __name__ == "__main__":
    main()
