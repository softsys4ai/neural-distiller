# convenient references and values to be used for all experiments across the project
import os
import numpy as np

# logging
spacer = "--------------------------------"

# remote configurations
log_dir = "/home/blakete/cookie-cutter-neural-distiller/neural-distiller/src/logs"
util_dir = "/home/blakete/cookie-cutter-neural-distiller/neural-distiller/src/utils"

# # local configurations
# log_dir="/Users/blakeedwards/Desktop/Repos/research/neural-distiller-softsys4ai/src/logs"
# util_dir="/Users/blakeedwards/Desktop/Repos/research/neural-distiller-softsys4ai/src/utils"

#  training
model_type = "resnet"
dataset = "cifar100"
network_input_shape = (28, 28, 1)
dataset_num_classes = 100
train_batch_size = 128
learning_rate = 0.01
use_fit_generator_teacher = False
use_fit_generator_student = False
teacher_optimizer = "sgd"
student_optimizer = "sgd"
teacher_model_size = 6
student_model_size = 6

# 0. eskd_teacher_logits_train
debug0 = False
use_datagen0 = True
epoch_min0 = 0
epoch_max0 = 200
interval_size0 = 1
epoch_intervals0 = np.arange(epoch_min0, epoch_max0+interval_size0, interval_size0)

# 1. eskd_knowledge_distillation_train
USE_EXPLICIT_START_MODEL = False
EXPLICIT_START_MODEL_PATH = ""
EXPLICIT_START_MODEL_WEIGHT_PATH = ""
# # remote
# logit_experiment_dir = "/home/blakete/cookie-cutter-neural-distiller/neural-distiller/src/logs/ESKD_Logit_Harvesting_cifar100_6_21-01-20_22:40:14"
# local
logit_experiment_dir = "/Users/blakeedwards/Documents/jamshidi-offline-research/ESKD/Training-Results/Experiment 3/ESKD_cifar100_10_16-12-19_11:19:41"
logits_dir = os.path.join(logit_experiment_dir, "logits")
alpha = 1.0
student_epochs = 150
teacher_logit_model_size = 10
teacher_logit_distillation_epoch_interval = 1
min_teacher_logit_epoch = 0
max_teacher_logit_epochs = 150
total_teacher_logit_epochs = 200
arr_of_distillation_epochs = np.arange(min_teacher_logit_epoch,
                                       max_teacher_logit_epochs + teacher_logit_distillation_epoch_interval - 1e-2,
                                       teacher_logit_distillation_epoch_interval)
min_temp = 1
max_temp = 10
temp_interval = 1
arr_of_distillation_temps = np.arange(min_temp, max_temp + temp_interval, temp_interval)

# 2. eskd_student_noise_and_adversarial_evaluation
MODEL_DIR = ""
attack_type = "fgm"
MEAN = 0
SIGMA = 0.1
MIN_SIGMA = 0.1
MAX_SIGMA = 0.3
STEP_SIGMA = 0.1
SIGMA_VALS = np.arange(MIN_SIGMA, MAX_SIGMA + STEP_SIGMA - 1e-2, STEP_SIGMA)
MIN_EPS = 0.1
MAX_EPS = 0.1
STEP_EPS = 0.1
EPS_VALS = np.arange(MIN_EPS, MAX_EPS + STEP_EPS - 1e-2, STEP_EPS)


# 3. eskd_baseline_train
debug3 = False
USE_EXPLICIT_START = False
EXPLICIT_START_WEIGHT_PATH = "/home/blakete/model_10_0|200_0.01_0.008.h5"
USE_SAME_STARTING_WEIGHTS = False
USE_BASELINE_DATA_AUGMENTATION = True
USE_BASELINE_LR_SCHEDULER = True
num_models_to_train = 2
baseline_models_train_epochs = 200
model_size = 2


# 4. eskd_baseline_noise_and_adversarial_evaluation
MODEL_DIR4 = "/Users/blakeedwards/Desktop/Repos/personal/Neural-Distillation/neural-distillation/src/logs/ESKD_baseline_cifar100_2_19-01-20_14:59:09/models"
attack_type4 = "fgm"
MEAN4 = 0
MIN_SIGMA4 = 0.1
MAX_SIGMA4 = 0.3
STEP_SIGMA4 = 0.1
SIGMA_VALS4 = np.arange(MIN_SIGMA, MAX_SIGMA + STEP_SIGMA - 1e-2, STEP_SIGMA)
MIN_EPS4 = 0.1
MAX_EPS4 = 0.1
STEP_EPS4 = 0.1
EPS_VALS4 = np.arange(MIN_EPS, MAX_EPS + STEP_EPS - 1e-2, STEP_EPS)


# plotting and figures
figures_path = "/Users/blakeedwards/Desktop/Repos/research/neural-distiller-softsys4ai/src/visualization/figures"
generated_csv_path = "/Users/blakeedwards/Desktop/Repos/personal/Neural-Distillation/neural-distillation/data/raw"










