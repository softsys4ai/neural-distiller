import cleverhans
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import keras
from keras import backend
from keras import optimizers
from keras.models import load_model

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper

from matplotlib import pyplot as plt
import imageio

backend.set_learning_phase(False)
# keras_model = load_model('models/Jan-13-2018.hdf5')

import KnowledgeDistillationModels
import LoadDataset

X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean

vanilla_model = KnowledgeDistillationModels.get_model_cifar100(100, X_train, 8)
vanilla_model.load_weights("/Users/blakeedwards/Desktop/Repos/research/neural-distiller/run-experiment/Adversarial/trained_models/size_8.h5")
sgd = optimizers.sgd(lr=1e-2)
vanilla_model.compile(optimizer="sgd",
                      loss="categorical_crossentropy",
                      metrics=["acc"])

sess = backend.get_session()

# train_acc = vanilla_model.evaluate(X_train, Y_train, batch_size=128)
test_acc = vanilla_model.evaluate(X_test, Y_test, batch_size=128)
print("Validation accuracy is: {}".format(test_acc))

wrap_model = KerasModelWrapper(vanilla_model)
fgsm = FastGradientMethod(wrap_model, sess=sess)

fgsm_params = {
    'eps': 0.3,
    'clip_min':0.,
    'clip_max':1.
}
print("Generating adversarial examples...")
adv_x = fgsm.generate_np(X_test, **fgsm_params)
adversarial_accuracy = vanilla_model.evaluate(adv_x, Y_test, batch_size=128)
print("Adversarial accuracy: {}".format(adversarial_accuracy))

print("COMPLETE")































