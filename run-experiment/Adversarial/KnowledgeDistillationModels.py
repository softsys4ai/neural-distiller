from keras.layers import MaxPooling2D, Dense, Flatten, Activation, Conv2D, BatchNormalization, Dropout
from keras.models import Sequential
# from Configuration import Config as cfg

def get_model(dataset, numClasses, X_train, net_size):
    if dataset is "mnist":
        return get_model_mnist(numClasses, X_train, net_size)
    elif dataset is "cifar100":
        return get_model_cifar100(numClasses, X_train, net_size)

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
            Dense(10, name='logits'),
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

# '2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
# 	'4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
# 	'6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool','Conv128', 'Conv128' ,'FC100'],
# 	'8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
# 		  'Conv256', 'Conv256','MaxPool', 'FC64', 'FC100'],
# 	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
# 		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC512', 'FC100'],
def get_model_cifar100(numClasses, X_train, net_size):
    # setting up model based on size
    if net_size == 10:
        model = Sequential([
            Conv2D(32,  kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
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
            Conv2D(32,  kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(256,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
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
            Conv2D(32,  kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(128,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
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
            Conv2D(32,  kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Conv2D(32,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Conv2D(64,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
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
            Conv2D(32,  kernel_size=3, input_shape=X_train.shape[1:], strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(32,  kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            Flatten(),
            Dense(numClasses, name='logits'),
            Activation('softmax'),
        ])
    else:
        print('no model available for given size!')
    return model