# Blake Edwards, 12/21/19
# Step 3
# Evaluate student models for robustness (adversarial and noisy images)

# external imports
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, 0-4;
os.environ["CUDA_VISIBLE_DEVICES"]="2";

# external dependencies
import os
import re
import glob
import numpy as np
import pandas as pd

from tensorflow.python.keras.optimizers import SGD

from art.attacks import FastGradientMethod, BasicIterativeMethod
from art.classifiers import KerasClassifier

# project imports
from Data import LoadDataset
from Models import KnowledgeDistillationModels

# EXPERIMENT PARAMETERS
dataset = "cifar100"
net_type = "vanilla"
attack_type = "fgm"
# gaussian noise
MEAN = 0
SIGMA = 0.1
# adversarial attack
MIN_EPS = 0.1
MAX_EPS = 0.1
STEP_EPS = 0.1
EPS_VALS = np.arange(MIN_EPS, MAX_EPS+STEP_EPS-1e-2, STEP_EPS)

# experiment results CSV
RESULTS_FILE = "experiment3_AR_"+attack_type+".csv"
# generate a list of paths to the student models
# MODEL_DIR = "/home/blakete/ESKD_Knowledge_Distillation_cifar100_2_18-12-19_18:04:58/models"
MODEL_DIR = "/Users/blakeedwards/Documents/jamshidi-offline-research/ESKD/Training-Results/Experiment 1/ESKD_Knowledge_Distillation_cifar100_2_16-12-19_23:17:30/models"
DIR_QUERY = os.path.join(MODEL_DIR, "*.h5")
STUDENT_MODEL_WEIGHT_PATHS = glob.glob(DIR_QUERY)
# generate a list of parsed student model information
rm_path = MODEL_DIR + "/"
STUDENT_MODEL_NAMES = [x[len(rm_path):] for x in STUDENT_MODEL_WEIGHT_PATHS]
# parse values out of model names
sizes = []
intervals = []
temperatures = []
test_accs = []
train_accs = []
for name in STUDENT_MODEL_NAMES:
    size, interval, temp, test_acc, train_acc = re.findall(rf"model_(\d+)_(\d+)\|\d+_(\d+)_(\d+.\d+)_(\d+.\d+).", name)[0]
    sizes.append(int(size))
    intervals.append(int(interval))
    temperatures.append(float(temp))
    test_accs.append(float(test_acc))
    train_accs.append(float(train_acc))

# create dataframe with the parsed data
df = pd.DataFrame(list(zip(sizes, intervals, temperatures, test_accs, train_accs)),
                     columns=['size', 'interval', 'temp', 'test_acc', 'train_acc'])

# loading dataset
X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
# "centering" data samples
X_train, X_test = LoadDataset.z_standardization(X_train, X_test)
# creating gaussian noised set of images for evaluation
X_test_gaussian_noise = np.zeros_like(X_test)
X_train_gaussian_noise = np.zeros_like(X_train)
for i in range(len(X_test)):
    X_test_gaussian_noise[i] = X_test[i] + np.random.normal(MEAN, SIGMA, (X_test[i].shape[0], X_test[i].shape[1],
                                                                           X_test[i].shape[2]))
    X_train_gaussian_noise[i] = X_train[i] + np.random.normal(MEAN, SIGMA, (X_train[i].shape[0], X_train[i].shape[1],
                                                                          X_train[i].shape[2]))

# min and max values of the test set for adversarial example generation
dataset_min = np.min(X_test)
dataset_max = np.max(X_test)


# create column in dataframe for each adversarial accuracy value
zeros = [0 for name in STUDENT_MODEL_NAMES]
df["gauss_noise"] = zeros
for eps in EPS_VALS:
    df[("eps_"+str(format(eps, '.3f')))] = zeros

size = 2
print("[INFO] Loading student model...")
curr_student_model = KnowledgeDistillationModels.get_model(dataset, 100, X_train, int(size), net_type)
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
curr_student_model.compile(optimizer=optimizer,
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])
# curr_student_model.summary()

for i in range(len(STUDENT_MODEL_WEIGHT_PATHS)):
    print("\n--------------------------Starting new AR step--------------------------")
    # load weights for the student model
    print("[INFO] Loading student model weights...")
    curr_student_model.load_weights(STUDENT_MODEL_WEIGHT_PATHS[i])
    evald_gauss = False
    for curr_eps in EPS_VALS:
        print(f"[INFO] Evaluating {STUDENT_MODEL_NAMES[i]} with FGSM at epsilon {format(curr_eps, '.3f')}...")
        # generate adversarial examples for the given model
        student_art_model = KerasClassifier(model=curr_student_model, clip_values=(dataset_min, dataset_max), use_logits=False)
        # generate adv. examples for current loaded model
        print("[INFO] Generating adversarial examples for the current model...")
        if attack_type is "fgm":
            attack_student_model = FastGradientMethod(classifier=student_art_model, eps=curr_eps)
        elif attack_type is "bim":
            attack_student_model = BasicIterativeMethod(classifier=student_art_model, eps_step=0.025, eps=curr_eps,
                                                    max_iter=4, targeted=False, batch_size=1)
        else:
            print("[WARNING] attack type not supported!")
            break
        X_test_adv = attack_student_model.generate(x=X_test)
        print("[INFO] Evaluating student model's adversarial accuracy...")
        predictions = student_art_model.predict(X_test_adv)
        adv_acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
        df.iloc[i, df.columns.get_loc("eps_" + str(format(curr_eps, '.3f')))] = adv_acc
        if evald_gauss is False:
            print("[INFO] Evaluating student model's gaussian noise accuracy...")
            predictions2 = curr_student_model.predict(X_test_gaussian_noise)
            gauss_acc = np.sum(np.argmax(predictions2, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
            df.iloc[i, df.columns.get_loc("gauss_noise")] = gauss_acc
        print("[INFO] Completed adversarial evaluation...")
        # print(f"Adversarial accuracy: {adv_acc}")
        if evald_gauss is False:
            print(f"Gaussian noise accuracy: {gauss_acc}")
            evald_gauss = True
        print("[INFO] Cleaning up experiment variables...")
        # del X_test_adv
        # del predictions
        del predictions2
        print(f"[INFO] Recording adversarial robustness results to {RESULTS_FILE}...")
        df.to_csv(RESULTS_FILE, sep=',')
























