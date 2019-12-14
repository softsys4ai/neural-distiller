import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_mnist

import KnowledgeDistillationModels
import LoadDataset

X_train, Y_train, X_test, Y_test = LoadDataset.load_cifar_100(None)
x_train_mean = np.mean(X_train, axis=0)
X_train -= x_train_mean
X_test -= x_train_mean
min = np.min(X_train)
max = np.max(X_train)

vanilla_model = KnowledgeDistillationModels.get_model_cifar100(100, X_train, 4)
vanilla_model.load_weights("/Users/blakeedwards/Desktop/Repos/research/neural-distiller/run-experiment/Adversarial/trained_models/size_4.h5")
vanilla_model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                      loss="categorical_crossentropy",
                      metrics=["acc"])

# vanilla_model.fit(X_train, Y_train,
#                   batch_size=128,
#                   epochs=100,
#                   verbose=1,
#                   callbacks=[],
#                   validation_data=(X_test, Y_test))
#
# vanilla_model.save_weights("/Users/blakeedwards/Desktop/Repos/research/neural-distiller/run-experiment/Adversarial/trained_models/size_4_NOKD.h5")s

art_model = KerasClassifier(model=vanilla_model, clip_values=(min, max), use_logits=False)

print("Evaluating model's test accuracy...")
predictions = art_model.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
print("Model accuracy: {}".format(accuracy))

attack = FastGradientMethod(classifier=art_model, eps=0.1)

print("Generating adversarial images...")
x_test_adversarial_images = attack.generate(x=X_test)
np.save("/Users/blakeedwards/Desktop/Repos/research/neural-distiller/run-experiment/Adversarial/adversarial_examples/fgsm.npy", x_test_adversarial_images)

print("Evaluating model's adversarial test accuracy...")
predictions = art_model.predict(x_test_adversarial_images)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
print("Model accuracy: {}".format(accuracy))




































