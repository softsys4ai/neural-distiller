import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from data import load_dataset
from models import knowledge_distillation_models
from tensorflow.python.keras.optimizers import SGD

net_size = 6
net_type = "resnet"
weights_file = "/Users/blakeedwards/Desktop/remote-files/ESKD_Logit_Harvesting_cifar100_6_23-01-20_21:33:10/models/model_15_0.52990_0.65412.h5"
# model_5_0.30820_0.43740.h5
# model_200_0.75040_0.99956.h5
cmap = "binary"

# load the dataset and normalize it
X_train, Y_train, X_test, Y_test = load_dataset.load_cifar_100(None)
X_train, X_test = load_dataset.z_standardization(X_train, X_test)

# load the model
model = knowledge_distillation_models.get_model("cifar100", 100, X_train, net_size, net_type=net_type)
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.load_weights(weights_file)

# use the loaded model to produce the model predictions
outputs = model.predict(X_test)

# create the confusion matrix based on the outputs and labels
labels = np.zeros(len(Y_test))
for i in range(len(Y_test)):
    labels[i] = np.argmax(Y_test[i])
preds = np.zeros(len(outputs))
for i in range(len(outputs)):
    preds[i] = np.argmax(outputs[i])
confusion = tf.math.confusion_matrix(labels=labels, predictions=preds, num_classes=100)

# plot the confusion matrix on a heat map
train_plot = sns.heatmap(confusion, cmap=cmap)
plt.show()
