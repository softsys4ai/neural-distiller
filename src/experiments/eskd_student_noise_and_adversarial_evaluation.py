# Blake Edwards, 12/21/19
# Step 3
# Evaluate student models for robustness (adversarial and noisy images)

# external dependencies
import os
import re
import glob
import numpy as np
import pandas as pd
from datetime import datetime

from tensorflow.keras.optimizers import SGD

from art.attacks import FastGradientMethod, BasicIterativeMethod
from art.classifiers import KerasClassifier

# project imports
from data import load_dataset
from utils import config_reference as cfg
from models import knowledge_distillation_models

def run():
    # EXPERIMENT PARAMETERS
    log_dir = cfg.log_dir
    now = datetime.now()
    now_datetime = now.strftime("%d-%m-%y_%H:%M:%S")
    log_dir = os.path.join(log_dir, f"ESKD_student_noise_and_adv_evaluation_{cfg.dataset}_{cfg.student_model_size}_{now_datetime}")
    os.mkdir(log_dir)
    RESULTS_FILE = os.path.join(log_dir, "results.csv")

    DIR_QUERY = os.path.join(cfg.MODEL_DIR, "*.h5")
    STUDENT_MODEL_WEIGHT_PATHS = glob.glob(DIR_QUERY)
    # generate a list of parsed student model information
    rm_path = cfg.MODEL_DIR + "/"
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

    # loading dataset and "centering" data samples
    X_train, Y_train, X_test, Y_test = load_dataset.load_cifar_100(None)
    X_train, X_test = load_dataset.z_standardization(X_train, X_test)

    # creating gaussian noised set of images for evaluation
    X_test_gauss_noised_sets = []
    X_train_gauss_noised_sets = []
    for i in range(len(cfg.SIGMA_VALS)):
        X_test_gauss_noised = np.zeros_like(X_test)
        X_train_gauss_noised = np.zeros_like(X_train)
        for j in range(len(X_test)):
            X_test_gauss_noised[j] = X_test[j] + np.random.normal(cfg.MEAN, cfg.SIGMA_VALS[i], (X_test[j].shape[0],
                                                                    X_test[j].shape[1], X_test[j].shape[2]))
        for j in range(len(X_train)):
            X_train_gauss_noised[j] = X_train[j] + np.random.normal(cfg.MEAN, cfg.SIGMA_VALS[i], (X_train[j].shape[0],
                                                                    X_train[j].shape[1], X_train[j].shape[2]))
        X_test_gauss_noised_sets.append(X_test_gauss_noised)
        X_train_gauss_noised_sets.append(X_train_gauss_noised)


    # min and max values of the test set for adversarial example generation
    dataset_min = np.min(X_test)
    dataset_max = np.max(X_test)

    # create column in dataframe for each adversarial accuracy value
    zeros = [0 for name in STUDENT_MODEL_NAMES]
    for eps in cfg.EPS_VALS:
        df[("eps_"+str(format(eps, '.3f')))] = zeros
    for sig in cfg.SIGMA_VALS:
        df[("sig_"+str(format(sig, '.3f')))] = zeros

    print("[INFO] Loading student model...")
    curr_student_model = knowledge_distillation_models.get_model(cfg.dataset, 100, X_train, int(size), cfg.model_type)
    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    curr_student_model.compile(optimizer=optimizer,
                               loss="categorical_crossentropy",
                               metrics=["accuracy"])
    # curr_student_model.summary()

    for j in range(len(STUDENT_MODEL_WEIGHT_PATHS)):
        print("\n--------------------------Starting new AR step--------------------------")
        # load weights for the student model
        print("[INFO] Loading student model weights...")
        curr_student_model.load_weights(STUDENT_MODEL_WEIGHT_PATHS[j])
        for i in range(max(len(cfg.EPS_VALS), len(cfg.SIGMA_VALS))):
            if cfg.USE_ADV_ATTACK:
                if i < len(cfg.EPS_VALS):
                    # evaluating adversarial attack robustness
                    curr_eps = cfg.EPS_VALS[i]
                    print(f"[INFO] Evaluating {STUDENT_MODEL_NAMES[j]} with attack at epsilon {format(curr_eps, '.3f')}...")
                    student_art_model = KerasClassifier(model=curr_student_model, clip_values=(dataset_min, dataset_max), use_logits=False)
                    print("[INFO] Generating adversarial examples for the current model...")
                    if cfg.attack_type is "fgm":
                        attack_student_model = FastGradientMethod(classifier=student_art_model, eps=curr_eps)
                    elif cfg.attack_type is "bim":
                        attack_student_model = BasicIterativeMethod(classifier=student_art_model, eps_step=0.025, eps=curr_eps,
                                                                max_iter=4, targeted=False, batch_size=1)
                    else:
                        print("[WARNING] attack type not supported!")
                        break
                    X_test_adv = attack_student_model.generate(x=X_test)
                    print("[INFO] Evaluating student model's adversarial accuracy...")
                    predictions = student_art_model.predict(X_test_adv)
                    adv_acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
                    df.iloc[j, df.columns.get_loc("eps_" + str(format(curr_eps, '.3f')))] = adv_acc
                    print(f"Adversarial accuracy: {adv_acc}")
            if cfg.USE_GAUSS_NOISE:
                if i < len(cfg.SIGMA_VALS):
                    # evaluating gaussian noise robustness
                    curr_sig = cfg.SIGMA_VALS[i]
                    print(f"[INFO] Evaluating {STUDENT_MODEL_NAMES[j]} with Gaussian Noise at sigma {format(curr_sig, '.3f')}...")
                    predictions2 = curr_student_model.predict(X_test_gauss_noised_sets[i])
                    gauss_acc = np.sum(np.argmax(predictions2, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
                    df.iloc[j, df.columns.get_loc("sig_"+str(format(curr_sig, '.3f')))] = gauss_acc
                    print("[INFO] Completed adversarial evaluation...")
                    print(f"Gaussian noise accuracy: {gauss_acc}")
        del student_art_model
        print(f"[INFO] Recording adversarial robustness results to {RESULTS_FILE}...")
        df.to_csv(RESULTS_FILE, sep=',')
























