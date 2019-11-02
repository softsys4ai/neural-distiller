import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

class VGG16:
    def __init__(self, dataset, checkpoint_dir, log_dir, epochs, batch_size, lr, weights_path=None):
        self.weights_path = weights_path
        self.dataset_name = dataset

        if self.dataset_name == 'cifar10':
            (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()
            self.num_classes = 10
            self.train_y = to_categorical(self.train_y, self.num_classes)
            self.test_y = to_categorical(self.test_y, self.num_classes)
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100':
            self.train_x, self.train_y, self.test_x, self.test_y = tf.keras.datasets.cifar100.load_data()
            self.num_classes = 100
            self.train_y = to_categorical(self.train_y, self.num_classes)
            self.test_y = to_categorical(self.test_y, self.num_classes)
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist':
            self.train_x, self.train_y, self.test_x, self.test_y = tf.keras.datasets.mnist.load_data()
            self.num_classes = 10
            self.train_y = to_categorical(self.train_y, self.num_classes)
            self.test_y = to_categorical(self.test_y, self.num_classes)
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        # if self.dataset_name == 'tiny-imagenet':
        #     self.train_x, self.train_y, self.test_x, self.test_y = load_tinyimagenet()
        #     self.num_classes = 200
        #     self.train_y = to_categorical(self.train_y, self.num_classes)
        #     self.test_y = to_categorical(self.test_y, self.num_classes)
        #     self.img_size = 64
        #     self.c_dim = 3
        #     self.label_dim = 200

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.epoch = epochs
        self.batch_size = batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = lr
        print("initialized")

    def build(self):
        start_time = time.time()
        print("build model started")
        model = Sequential([
            ZeroPadding2D((1, 1), input_shape=(32,32,3)),
            Convolution2D(64, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(64, 3, 3, activation='relu'),
            # MaxPooling2D((2, 2), strides=(2, 2)),

            ZeroPadding2D((1, 1)),
            Convolution2D(128, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(128, 3, 3, activation='relu'),
            # MaxPooling2D((2, 2), strides=(2, 2)),

            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            # MaxPooling2D((2, 2), strides=(2, 2)),

            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            # MaxPooling2D((2, 2), strides=(2, 2)),

            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            # MaxPooling2D((2, 2), strides=(2, 2)),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        if self.weights_path is not None:
            model.load_weights(self.weights_path)
        print(("build model finished: %ds" % (time.time() - start_time)))
        return model


    def build_as_teacher(self):
        print('TODO build VGG net w/ temp. and w/o softmax')

    def build_as_student(self):
        print('TODO build VGG w/ concatenated softmax output')