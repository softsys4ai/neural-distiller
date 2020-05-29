# from tensorflow.python.keras.layers import MaxPooling2D, Dense, Flatten, Activation, Conv2D, BatchNormalization, Dropout
# from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling2D, Dense, Flatten, Activation, Conv2D, BatchNormalization, Dropout
from tensorflow.python.keras.models import Sequential
from utils import config_reference as cfg
from models import generate_resnet


def get_model(dataset, numClasses, X_train, net_size, net_type="vanilla"):
    if dataset is "mnist" and net_type is "vanilla":
        return get_model_mnist(numClasses, X_train, net_size)
    elif dataset is "cifar100" and net_type is "vanilla":
        return get_vanilla_model_cifar100(numClasses, X_train, net_size)
    elif dataset is "cifar10" and net_type is "vanilla":
        return get_vanilla_model_cifar100(numClasses, X_train, net_size)
    elif dataset is "cifar100" and net_type is "resnet":
        return get_resnet_model_cifar100(numClasses, X_train, net_size)
    else:
        return None


def get_model_mnist(numClasses, X_train, net_size):
    # setting up model based on size
    if net_size == 10:
        model = Sequential([
            Conv2D(256, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
    elif net_size == 8:
        model = Sequential([
            Conv2D(128, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            # Conv2D(8, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(96, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
    elif net_size == 6:
        model = Sequential([
            Conv2D(64, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            # Conv2D(8, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_6_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 4:
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            # Conv2D(8, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_4_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 2:
        # model = Sequential([
        #     Conv2D(8, kernel_size=(3, 3),
        #            activation='relu',
        #            input_shape=X_train.shape[1:]),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(4, (3, 3), activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Flatten(),
        #     Dense(16, input_shape=X_train.shape[1:]),
        #     Activation('relu'),
        #     Dense(numClasses, name='logits'),
        #     Activation('softmax'),
        # ])
        model = Sequential([
            Dense(16, activation='relu', input_shape=X_train.shape[1:]),
            Dense(16, activation='relu', input_shape=X_train.shape[1:]),
            Activation('relu'),
            Flatten(),
            Dense(cfg.mnist_number_classes, name='logits'),
            Activation('softmax'),
        ])
    else:
        print('no model available for given size!')
    return model


def get_model_cifar100_raw_output(numClasses, X_train, net_size):
    # setting up model based on size
    if net_size == 10:
        model = Sequential([
            Conv2D(16, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
    elif net_size == 8:
        model = Sequential([
            Conv2D(16, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            # Conv2D(8, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(96, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
    elif net_size == 6:
        model = Sequential([
            Conv2D(16, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            # Conv2D(8, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_6_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 4:
        model = Sequential([
            Conv2D(16, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            # Conv2D(8, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax')  # Note that we add a normal softmax layer to begin with
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_4_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 2:
        model = Sequential([
            Conv2D(4, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=X_train.shape[1:]),
            # MaxPooling2D(pool_size=(2, 2)),
            Conv2D(8, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(16, input_shape=X_train.shape[1:]),
            Activation('relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
        # model = Sequential([
        #     Dense(16, activation='relu', input_shape=X_train.shape[1:]),
        #     Dense(16, activation='relu', input_shape=X_train.shape[1:]),
        #     Activation('relu'),
        #     Flatten(),
        #     Dense(numClasses, name='logits'),
        #     Activation('softmax'),
        # ])
    else:
        print('no model available for given size!')
    return model


#   '2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
# 	'4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
# 	'6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool','Conv128', 'Conv128' ,'FC100'],
# 	'8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
# 		  'Conv256', 'Conv256','MaxPool', 'FC64', 'FC100'],
# 	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
# 		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC512', 'FC100'],
def get_vanilla_model_cifar100(numClasses, X_train, net_size):
    # setting up model based on size
    if net_size == 10:
        model = Sequential([
            Conv2D(32, kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same',
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
    elif net_size == 8:
        model = Sequential([
            Conv2D(32, kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same',
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
    elif net_size == 6:
        model = Sequential([
            Conv2D(32, kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same',
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Flatten(),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_6_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 4:
        model = Sequential([
            Conv2D(32, kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same',
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Flatten(),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
        # model = load_model(cfg.teacher_model_dir + "/best_size_4_model.hdf5")
        # previousModel = model
        # continue
    elif net_size == 2:
        model = Sequential([
            Conv2D(64, kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same',
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            # Dropout(0.1),
            # MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            # Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Flatten(),
            Dense(256),
            Activation('sigmoid'),
            Dense(256),
            Activation('sigmoid'),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
        model = Sequential([
            Conv2D(32, kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same',
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Flatten(),
            Dense(100),
            Activation('sigmoid'),
            Dense(50),
            Activation('sigmoid'),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
    else:
        print('no model available for given size!')
    return model


def get_resnet_model_cifar100(numClasses, X_train, net_size):
    net_size_map = {2: 3, 4: 5, 6: 7, 8: 9, 10: 18, 12: 27}
    return generate_resnet.generate_resnet_model(net_size_map[net_size], X_train.shape[1:], 100)