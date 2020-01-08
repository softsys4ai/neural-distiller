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

from Data import LoadDataset
from Models import KnowledgeDistillationModels

from keras.optimizers import SGD

import os
import re
import glob
import math
import pickle
import numpy as np
import pandas as pd

RESULTS_FILE = "experiment2_difference_results.csv"
# SAVE_FILENAME = "logit_diffs_"

# load dataset
X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean

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

student_sizes, student_intervals, student_temperatures, student_interval_max = parse_info_from_student_names(STUDENT_MODEL_NAMES)
teacher_sizes, teacher_intervals, teacher_interval_max = parse_info_from_teacher_names(TEACHER_MODEL_NAMES)
# create dataframe with the parsed data
df = pd.DataFrame(list(zip(student_sizes, student_intervals, student_temperatures)),
                     columns=['size', 'interval', 'temp'])

# TODO check the parsed information to make sure that the student and teacher experiments are correct for each other

# loading and compiling student and teacher models for test
student_model = KnowledgeDistillationModels.get_vanilla_model_cifar100(100, X_train, student_sizes[0], )
teacher_model = KnowledgeDistillationModels.get_vanilla_model_cifar100(100, X_train, teacher_sizes[0], )
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
student_model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
teacher_model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

zeros = [0 for name in STUDENT_MODEL_NAMES]
df["test_diff"] = zeros
df["train_diff"] = zeros

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
    train_diff = np.subtract(t_train_logits, s_train_logits)
    test_diff = np.subtract(t_test_logits, s_test_logits)
    df.iloc[i, df.columns.get_loc("train_diff")] = math.sqrt(np.sum(np.square(train_diff)))
    df.iloc[i, df.columns.get_loc("test_diff")] = math.sqrt(np.sum(np.square(test_diff)))
    print(f"[INFO] Recording difference results to {RESULTS_FILE}...")
    df.to_csv(RESULTS_FILE, sep=',')
    # curr_model_diffs = []
    # curr_model_diffs.append(train_diff)
    # curr_model_diffs.append(test_diff)
    # logit_differences.append(curr_model_diffs)

# print(f"[INFO] Dumping logit differences to: {SAVE_FILENAME}")
# with open(SAVE_FILENAME, 'wb') as save_file:
#     pickle.dump(logit_differences, save_file)
print("[INFO] Complete")

