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
from utils import config_reference as cfg
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

RESULTS_FILE = "logit_analysis.csv"
ENTROPY_RESULTS_FILE = "logit_entropy_analysis.csv"

FAST_MODE = True
INCLUDE_LOGIT_DIFF = False
INCLUDE_SOFT_DIFF = False

INCLUDE_ENTROPY = True
interval_of_interest = 5

# load dataset
X_train, Y_train, X_test, Y_test = load_dataset.load_cifar_100(None)
X_train, X_test = load_dataset.z_standardization(X_train, X_test)

# # uncomment to debug the code faster
if FAST_MODE:
    X_train = X_train[:10000]
    X_test = X_test[:10000]

# load all model paths
KD_EXPERIMENT_PATH = "/Users/blakeedwards/Documents/jamshidi-offline-research/ESKD/Training-Results/Experiment 2/ESKD_Knowledge_Distillation_cifar100_2_17-12-19_20:33:15/models"
LOGIT_EXPERIMENT_PATH = "/Users/blakeedwards/Documents/jamshidi-offline-research/ESKD/ResNet-Results/Teacher-Logit-Results/ESKD_Logit_Harvesting_cifar100_6_23-01-20_21:33:10/logits"
STUDENT_DIR_QUERY = os.path.join(KD_EXPERIMENT_PATH, "*.h5")
TEACHER_DIR_QUERY = os.path.join(LOGIT_EXPERIMENT_PATH, "*.pkl")
STUDENT_MODEL_PATHS = glob.glob(STUDENT_DIR_QUERY)
TEACHER_LOGIT_PATHS = glob.glob(TEACHER_DIR_QUERY)

# remove file path to parse information from model names
student_rm_path = KD_EXPERIMENT_PATH + "/"
STUDENT_MODEL_NAMES = [x[len(student_rm_path):] for x in STUDENT_MODEL_PATHS]
teacher_rm_path = LOGIT_EXPERIMENT_PATH + "/"
TEACHER_LOGIT_NAMES = [x[len(teacher_rm_path):] for x in TEACHER_LOGIT_PATHS]


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
    interval_max = re.findall(rf"logits_\d+_\d+\|(\d+).", teacher_names[0])[0]
    for name in teacher_names:
        size, epoch = re.findall(rf"logits_(\d+)_(\d+)\|\d+.", name)[0]
        sizes.append(int(size))
        intervals.append(int(epoch))
    return sizes, intervals, interval_max

# method to load logits from the specified experiment directory
def load_logits(logit_filename):
    with open(logit_filename, "rb") as file:
        teacher_train_logits = pickle.load(file)
        teacher_test_logits = pickle.load(file)
    return teacher_train_logits, teacher_test_logits

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
teacher_sizes, teacher_intervals, teacher_interval_max = parse_info_from_teacher_names(TEACHER_LOGIT_NAMES)
# create dataframe with the parsed data
df = pd.DataFrame(list(zip(student_sizes, student_intervals, student_temperatures)),
                     columns=['size', 'interval', 'temp'])

# TODO check the parsed information to make sure that the student and teacher experiments are correct for each other

# loading and compiling student and teacher models for test
student_model = knowledge_distillation_models.get_vanilla_model_cifar100(100, X_train, student_sizes[0], )
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
student_model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

zeros = [0 for name in STUDENT_MODEL_NAMES]
df["teacher_interval"] = zeros
if INCLUDE_LOGIT_DIFF:
    df["test_logit_diff"] = zeros
    df["train_logit_diff"] = zeros
if INCLUDE_SOFT_DIFF:
    df["test_soft_diff"] = zeros
    df["train_soft_diff"] = zeros
if INCLUDE_ENTROPY:
    teacher_entropy_intervals = teacher_intervals.copy()
    for interval in teacher_intervals:
        if interval % interval_of_interest != 0:
            teacher_entropy_intervals.remove(interval)
    zeros = [0 for name in teacher_entropy_intervals]
    df_entropy = pd.DataFrame(list(zip(zeros, zeros)), columns=["avg_train_entropy", "avg_test_entropy"])
    df_entropy["teacher_entropy_intervals"] = teacher_entropy_intervals

# calculating average entropy of teacher train logits
if INCLUDE_ENTROPY:
    for i in range(len(teacher_entropy_intervals)):
        print(f"[INFO] Entropy calculations {i + 1}/{len(teacher_entropy_intervals)}")
        teacher_logit_path = TEACHER_LOGIT_PATHS[teacher_intervals.index(teacher_entropy_intervals[i])]
        # load teacher logits
        t_train_logits, t_test_logits = load_logits(teacher_logit_path)
        print("[INFO] Calculating teacher logit entropy...")
        train_min = abs(np.min(t_train_logits))+1
        test_min = abs(np.min(t_test_logits))+1
        t_train_logits += train_min
        t_test_logits += test_min
        train_min = np.min(t_train_logits)
        test_min = np.min(t_test_logits)
        total_entropy = 0
        for j in range(len(t_train_logits)):
            curr_entropy = 0
            for k in range(len(t_train_logits[0])):
                curr_val = t_train_logits[j][k]
                curr_entropy += t_train_logits[j][k] * math.log2(1 / t_train_logits[j][k])
            total_entropy += curr_entropy
        avg_train_entropy = total_entropy / len(t_train_logits)
        df_entropy.iloc[i, df_entropy.columns.get_loc("avg_train_entropy")] = avg_train_entropy
        # calculating average entropy of teacher train logits
        total_entropy = 0
        for j in range(len(t_test_logits)):
            curr_entropy = 0
            for k in range(len(t_test_logits[0])):
                curr_entropy += t_test_logits[j][k] * math.log2(1 / t_test_logits[j][k])
            total_entropy += curr_entropy
        avg_test_entropy = total_entropy / len(t_test_logits)
        df_entropy.iloc[i, df_entropy.columns.get_loc("avg_test_entropy")] = avg_test_entropy
        print(f"[INFO] Recording results to {ENTROPY_RESULTS_FILE}...")
        df_entropy.to_csv(os.path.join(cfg.processed_csv_path, ENTROPY_RESULTS_FILE), sep=',')

# iterate all student and teacher models, calculate the differences in their output distributions
if INCLUDE_LOGIT_DIFF or INCLUDE_SOFT_DIFF:
    logit_differences = []
    for i in range(len(STUDENT_MODEL_NAMES)):
        print(f"[INFO] Difference calculations {i+1}/{len(STUDENT_MODEL_NAMES)}")
        # load teacher logits
        teacher_logit_path = TEACHER_LOGIT_PATHS[teacher_intervals.index(student_intervals[i])]
        t_train_logits, t_test_logits = load_logits(teacher_logit_path)
        # load model weights for logit collection

        student_model_path = STUDENT_MODEL_PATHS[i]
        student_model.load_weights(student_model_path)
        # forward propogate and collect logits
        print(f"[INFO] Making model predictions...")
        s_train_logits = student_model.predict(X_train)
        s_test_logits = student_model.predict(X_test)

        if FAST_MODE:
            t_train_logits = t_train_logits[:10000]
            t_test_logits = t_test_logits[:10000]
        print(f"[INFO] Performing Euclidean Distance, Entropy, and KL Divergence Analysis...")

        # calculating total Euclidean distance between student and teacher logits
        if INCLUDE_LOGIT_DIFF:
            print("[INFO] Calculating logit difference...")
            train_logit_diff = np.subtract(t_train_logits, s_train_logits)
            test_logit_diff = np.subtract(t_test_logits, s_test_logits)
            df.iloc[i, df.columns.get_loc("train_logit_diff")] = math.sqrt(np.sum(np.square(train_logit_diff)))
            df.iloc[i, df.columns.get_loc("test_logit_diff")] = math.sqrt(np.sum(np.square(test_logit_diff)))

        # create soft targets at the correct temperature
        if INCLUDE_SOFT_DIFF:
            print("[INFO] Calculating soft difference...")
            temp = float(re.findall(rf"model_\d+_\d+\|\d+_(\d+)_\d+.\d+", STUDENT_MODEL_NAMES[i])[0])
            t_train_soft_targets, t_test_soft_targets = modified_kd_targets_from_logits(t_train_logits, t_test_logits, temp)

            # calculating total Euclidean distance between student and teacher soft targets
            s_train_soft_targets, s_test_soft_targets = modified_kd_targets_from_logits(s_train_logits, s_test_logits, 1)
            train_soft_diff = np.subtract(t_train_soft_targets, s_train_soft_targets)
            test_soft_diff = np.subtract(t_test_soft_targets, s_test_soft_targets)
            df.iloc[i, df.columns.get_loc("train_soft_diff")] = math.sqrt(np.sum(np.square(train_soft_diff)))
            df.iloc[i, df.columns.get_loc("test_soft_diff")] = math.sqrt(np.sum(np.square(test_soft_diff)))
        print(f"[INFO] Recording results to {RESULTS_FILE}...")
        df.to_csv(os.path.join(cfg.processed_csv_path, RESULTS_FILE), sep=',')

print("[INFO] Complete")

