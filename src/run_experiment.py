import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import socket
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import logging
import sys
import traceback
from optparse import OptionParser
from datetime import datetime
from utils import config_reference
from experiments import multistage_knowledge_distillation_train, \
    eskd_knowledge_distillation_train, eskd_student_noise_and_adversarial_evaluation, \
    eskd_baseline_train, eskd_baseline_noise_and_adversarial_evaluation, eskd_teacher_train_and_collect_logits


def config_option_parser():
    # reading command line input
    usage = """USAGE: %python run_experiment.py -m [model]
                 Model ResNet50:                  python run_experiment.py -m resnet50
                 Model VGG16:                     python run_experiment.py -m vgg16
                 Model LeNet-5:                   python run_experiment.py -m lenet5
                 Model AlexNet:                   python run_experiment.py -m alexnet
                 Model Xception:                  python run_experiment.py -m xception
                 Model InceptionV3:               python run_experiment.py -m inceptionv3
                 Model CustomTeacher:             python run_experiment.py -m custom_teacher
                 Model CustomStudent:             python run_experiment.py -m custom_student
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
    parser.add_option('-g', "--gpu",
                      action="store",
                      type="string",
                      default="0",
                      dest="gpu",
                      help="Which GPU to use for training")
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
    logger.info(config_reference.spacer + "NEW TRAINING SESSION STARTED" + config_reference.spacer)
    logger.info("Initialized logger")
    return logger


def main():
    # command line input
    (options, usage) = config_option_parser()

    # set GPU to train on
    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

    # get log directory
    log_dir = os.getcwd() + config_reference.log_dir
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%Y_%H:%M:%S")
    if options.experiment == "search-alpha-temp-configurations":
        log_dir = log_dir + config_reference.dataset + "_grid_search_" + now_datetime
        os.mkdir(log_dir)
        logits_dir = log_dir + "/" + "saved_logits"
        os.mkdir(logits_dir)
        models_dir = log_dir + "/" + "saved_models"
        os.mkdir(models_dir)

    # add high priority
    os.nice(1)

    try:
        if options.experiment == "multistage_knowledge_distillation_train":
            print(f"Selected: {options.experiment}")
            session_log_file = log_dir + "/training_session.log"
            multistage_knowledge_distillation_train.run(None, options, session_log_file, logits_dir, models_dir)
        elif options.experiment == "eskd_teacher_train_and_collect_logits":
            print(f"Selected: {options.experiment}")
            eskd_teacher_train_and_collect_logits.run()
        elif options.experiment == "eskd_knowledge_distillation_train":
            print(f"Selected: {options.experiment}")
            eskd_knowledge_distillation_train.run(options.gpu)
        elif options.experiment == "eskd_student_noise_and_adversarial_evaluation":
            print(f"Selected: {options.experiment}")
            eskd_student_noise_and_adversarial_evaluation.run()
        elif options.experiment == "eskd_baseline_train":
            print(f"Selected: {options.experiment}")
            eskd_baseline_train.run()
        elif options.experiment == "eskd_baseline_noise_and_adversarial_evaluation":
            print(f"Selected: {options.experiment}")
            eskd_baseline_noise_and_adversarial_evaluation.run()
        else:
            print(f"[ERROR]: Experiment type \"{options.experiment}\" type not supported!")
            return
    except Exception:
        traceback.print_exc()
        error = traceback.format_exc()
        # error.upper()
        logging.error('Error encountered: %s' % error, exc_info=True)


if __name__ == "__main__":
    main()
