# load dataset
# load student and teacher models

# import the names of all student models from a given experiment directory
# parse all student names for interval, temperature, size
# import the names of all teacher models from a given experiment directory
# parse all teacher names for interval, size
# check if the student and teacher model parsed values are equal in size and value

# create differences array, dimensions: 1, # classes, # samples, # models

# iterate student model names
# find / generate name of the corresponding teacher network
# load teacher model weights
# load student model weights
# collect logits from student and teacher models

# iterate student and teacher logits
# iterate each logit set element
# store each element difference in difference array

from data import load_dataset
from models import knowledge_distillation_models

from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import backend as K

import os
import re
import glob
import math
import pickle
import numpy as np
import pandas as pd
import config_reference as cfg
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

RESULTS_FILE = "logit_soft_analysis.csv"

# load dataset
X_train, Y_train, X_test, Y_test = load_dataset.load_cifar_100(None)
X_train, X_test = load_dataset.z_standardization(X_train, X_test)

# uncomment to debug the code faster
X_train = X_train[:10000]
X_test = X_test[:10000]

# load all model paths
KD_EXPERIMENT_PATH = "/Users/blakeedwards/Desktop/Repos/research/neural-distiller/post-experiment/ESKD-Analysis/ESKD Accuracy/results/experiment-3/ESKD_Knowledge_Distillation_cifar100_2_18-12-19_18:04:58/models"
LOGIT_EXPERIMENT_PATH = "/Users/blakeedwards/Desktop/Repos/research/neural-distiller/post-experiment/ESKD-Analysis/ESKD Accuracy/results/experiment-3/ESKD_cifar100_10_duplicate_of_experiment_2_logits/models"
STUDENT_DIR_QUERY = os.path.join(KD_EXPERIMENT_PATH, "*.h5")
TEACHER_DIR_QUERY = os.path.join(LOGIT_EXPERIMENT_PATH, "*.h5")
STUDENT_MODEL_PATHS = glob.glob(STUDENT_DIR_QUERY)
TEACHER_MODEL_PATHS = glob.glob(TEACHER_DIR_QUERY)

# parse information from student model names
student_rm_path = KD_EXPERIMENT_PATH + "/"
STUDENT_MODEL_NAMES = [x[len(student_rm_path):] for x in STUDENT_MODEL_PATHS]
teacher_rm_path = LOGIT_EXPERIMENT_PATH + "/"
TEACHER_MODEL_NAMES = [x[len(teacher_rm_path):] for x in TEACHER_MODEL_PATHS]


# easily parse student model information from a list of student model names
def parse_info_from_student_names(student_names):
    sizes = []
    intervals = []
    temperatures = []
    interval_max = re.findall(rf"model_\d+_\d+\|(\d+)_\d+_\d+.\d+_\d+.\d+.", student_names[0])[0]
    for name in student_names:
        size, interval, temp = re.findall(rf"model_(\d+)_(\d+)\|\d+_(\d+)_\d+.\d+_\d+.\d+.", name)[0]
        sizes.append(int(size))
        intervals.append(int(interval))
        temperatures.append(float(temp))
    return sizes, intervals, temperatures, interval_max


# easily parse student model information from a list of student model names
def parse_info_from_teacher_names(teacher_names):
    sizes = []
    intervals = []
    interval_max = re.findall(rf"model_\d+_\d+\|(\d+)_\d+.\d+_\d+.\d+.", teacher_names[0])[0]
    for name in teacher_names:
        size, interval = re.findall(rf"model_(\d+)_(\d+)\|\d+_\d+.\d+_\d+.\d+.", name)[0]
        sizes.append(int(size))
        intervals.append(int(interval))
    return sizes, intervals, interval_max

# convert loaded logits to soft targets at a specified temperature
def modified_kd_targets_from_logits(train_logits, test_logits, temp=1):
    # create soft targets from loaded logits
    if temp <= 0:
        temp = 1
    train_logits_t = train_logits / temp
    test_logits_t = test_logits / temp
    Y_train_soft = K.softmax(train_logits_t)
    Y_test_soft = K.softmax(test_logits_t)
    sess = K.get_session()
    Y_train_soft = sess.run(Y_train_soft)
    Y_test_soft = sess.run(Y_test_soft)
    return Y_train_soft, Y_test_soft


student_sizes, student_intervals, student_temperatures, student_interval_max = parse_info_from_student_names(STUDENT_MODEL_NAMES)
teacher_sizes, teacher_intervals, teacher_interval_max = parse_info_from_teacher_names(TEACHER_MODEL_NAMES)
# create dataframe with the parsed data
df = pd.DataFrame(list(zip(student_sizes, student_intervals, student_temperatures)),
                     columns=['size', 'interval', 'temp'])

# TODO check the parsed information to make sure that the student and teacher experiments are correct for each other

# loading and compiling student and teacher models for test
student_model = knowledge_distillation_models.get_vanilla_model_cifar100(100, X_train, student_sizes[0], )
teacher_model = knowledge_distillation_models.get_vanilla_model_cifar100(100, X_train, teacher_sizes[0], )
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
student_model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
teacher_model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

zeros = [0 for name in STUDENT_MODEL_NAMES]
df["test_logit_diff"] = zeros
df["train_logit_diff"] = zeros
df["test_soft_diff"] = zeros
df["train_soft_diff"] = zeros
df["avg_train_entropy"] = zeros
df["avg_test_entropy"] = zeros

# iterate all student and teacher models, calculate the differences in their output distributions
logit_differences = []
for i in range(len(STUDENT_MODEL_NAMES)):
    print(f"[INFO] {i+1}/{len(STUDENT_MODEL_NAMES)}")
    # load model weights for logit collection
    student_model_path = STUDENT_MODEL_PATHS[i]
    teacher_model_path = TEACHER_MODEL_PATHS[teacher_intervals.index(student_intervals[i])]
    print(student_model_path)
    print(teacher_model_path)
    student_model.load_weights(student_model_path)
    teacher_model.load_weights(teacher_model_path)
    # forward propogate and collect logits
    print(f"[INFO] Making model predictions...")
    s_train_logits = student_model.predict(X_train)
    s_test_logits = student_model.predict(X_test)
    t_train_logits = teacher_model.predict(X_train)
    t_test_logits = teacher_model.predict(X_test)
    print(f"[INFO] Performing Euclidean Distance, Entropy, and KL Divergence Analysis...")
    # calculating total Euclidean distance between student and teacher logits
    train_logit_diff = np.subtract(t_train_logits, s_train_logits)
    test_logit_diff = np.subtract(t_test_logits, s_test_logits)
    df.iloc[i, df.columns.get_loc("train_logit_diff")] = math.sqrt(np.sum(np.square(train_logit_diff)))
    df.iloc[i, df.columns.get_loc("test_logit_diff")] = math.sqrt(np.sum(np.square(test_logit_diff)))

    # create soft targets at the correct temperature
    temp = float(re.findall(rf"model_\d+_\d+\|\d+_(\d+)_\d+.\d+", STUDENT_MODEL_NAMES[i])[0])
    t_train_soft_targets, t_test_soft_targets = modified_kd_targets_from_logits(t_train_logits, t_test_logits, temp)

    # calculating total Euclidean distance between student and teacher soft targets
    s_train_soft_targets, s_test_soft_targets = modified_kd_targets_from_logits(s_train_logits, s_test_logits, 1)
    train_soft_diff = np.subtract(t_train_soft_targets, s_train_soft_targets)
    test_soft_diff = np.subtract(t_test_soft_targets, s_test_soft_targets)
    df.iloc[i, df.columns.get_loc("train_soft_diff")] = math.sqrt(np.sum(np.square(train_soft_diff)))
    df.iloc[i, df.columns.get_loc("test_soft_diff")] = math.sqrt(np.sum(np.square(test_soft_diff)))

    # calculating average entropy of teacher train logits
    train_test_min = min(np.min(t_train_logits), np.min(t_test_logits))+1
    t_train_logits += train_test_min
    t_test_logits += train_test_min
    total_entropy = 0
    for j in range(len(t_train_logits)):
        curr_entropy = 0
        for k in range(len(t_train_logits[0])):
            curr_val = t_train_logits[j][k]
            curr_entropy += t_train_logits[j][k] * math.log2(1 / t_train_logits[j][k])
        total_entropy += curr_entropy
    avg_train_entropy = total_entropy / len(t_train_logits)
    df.iloc[i, df.columns.get_loc("avg_train_entropy")] = avg_train_entropy
    # calculating average entropy of teacher train logits
    total_entropy = 0
    for j in range(len(t_test_logits)):
        curr_entropy = 0
        for k in range(len(t_test_logits[0])):
            curr_entropy += t_test_logits[j][k] * math.log2(1 / t_test_logits[j][k])
        total_entropy += curr_entropy
    avg_train_entropy = total_entropy / len(t_test_logits)
    df.iloc[i, df.columns.get_loc("avg_test_entropy")] = avg_train_entropy

    print(f"[INFO] Recording difference results to {RESULTS_FILE}...")
    df.to_csv(os.path.join(cfg.generated_csv_path, RESULTS_FILE), sep=',')

print("[INFO] Complete")

