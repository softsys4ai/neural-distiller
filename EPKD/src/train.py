import os
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils
import keras
import tempfile
from TeacherCNN import TeacherModel
from StudentDense import StudentModel
import argparse
from PrunePipeline import prune
import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "EPKD"})

def main():
    # reading command line input
    parser = argparse.ArgumentParser(description='add params to run.')
    parser.add_argument('-m', '--model_filename', default=None, type=str,
                    help='name of model configuration and weights files')
    parser.add_argument("-g", "--gpus", type=int, default=1,
                    help="# of GPUs to use for training")
    args = parser.parse_args()

    # grabbing the number of GPUs cores to use
    numGPUCores = args.gpus
    # preparing the MNIST dataset for training teacher and student models
    nb_classes = 10
    input_shape = (28, 28, 1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # convert y_train and y_test to categorical binary values 
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    # convert to float32 type
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # Normalize the values
    X_train /= 255
    X_test /= 255

    # setting up logs for tensorboard
    logdir = tempfile.mkdtemp()
    print('[INFO] writing training logs to ' + logdir)
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callbacks = [WandbCallback()]

    print('[INFO] creating teacher model')
    # compiling and training teacher network
    teacher = TeacherModel(callbacks)
    teacher.__init__()
    if args.model_filename is not None:
        teacher.load(args.model_filename + ".json", args.model_filename + ".h5")
    else:
        teacher.buildAndCompile()
        teacher.train(X_train, Y_train, X_test, Y_test)
        teacher.save(X_test, Y_test) # persisting trained teacher network

    # TODO perform pruning operations
    print("[INFO] Attempting to prune teacher network")
    pruningEpochs = 10
    batchSize = 256
    studentModel = prune(teacher.getModel(), X_train, Y_train, X_test, Y_test, len(X_train), batchSize, pruningEpochs, logdir)
    print(type(studentModel))
    print('[INFO] creating hybrid targets for student model training')
    # retreiving soft targets for student model training
    Y_train_new, Y_test_new = teacher.createStudentTrainingData(X_train, Y_train, X_test, Y_test)

    print('[INFO] creating and training student model')
    # compiling and training student network
    student = StudentModel(callbacks)
    student.__init__()
    student.buildAndCompile()
    student.train(X_train, Y_train_new, X_test, Y_test_new)
    student.save(X_test, Y_test_new)

    print('-- done')

if __name__ == "__main__":
    main()