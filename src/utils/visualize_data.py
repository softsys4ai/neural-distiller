import matplotlib.pyplot as plt
from numpy import load

x_train = load("../data/x_train.npy")
y_train = load("../data/y_train.npy")

offset = 45
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i+offset])
plt.show()

print(x_train[0].shape)
