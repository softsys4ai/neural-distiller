import numpy as np
from Configuration import Config as cfg
from tensorflow.python.keras.losses import categorical_crossentropy as logloss
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizers import adadelta
from tensorflow.python.keras.layers import Lambda, concatenate, Activation
from tensorflow.python.keras.models import Model, Sequential
nb_classes = 10

def apply_knowledge_distillation_modifications(logger, model):
    logger.info("Applying KD modifications to student network")
    # modifying student network for KD
    model.layers.pop()
    logits = model.layers[-1].output
    probs = Activation('softmax')(logits)
    logits_T = Lambda(lambda x: x / cfg.temp)(logits)
    probs_T = Activation('softmax')(logits_T)
    output = concatenate([probs, probs_T])
    model = Model(model.input, output)  # modified student model
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        # optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
        optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, cfg.alpha),
        metrics=[acc])
    return model

def revert_knowledge_distillation_modifications(logger, model):
    logger.info("Reverting KD modifications to student network")
    model.layers.pop()
    logits = model.layers[-1].output
    output = Activation('softmax')(logits)
    model = Model(model.input, output)  # reverted student model
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                         optimizer=adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                         metrics=['accuracy'])
    return model

def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

def knowledge_distillation_loss(y_true, y_pred, alpha):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    loss = alpha*logloss(y_true, y_pred) + logloss(y_true_softs, y_pred_softs)
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)

def calculate_weighted_score(logger, model, X_train, Y_train, X_test, Y_test):
    logger.info('Calculating weighted model score')
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    val_score = model.evaluate(X_test, Y_test, verbose=0)
    logger.info('Raw model scores: (train_acc, train_loss, val_acc, val_loss) --> (%s, %s, %s, %s)'
                % (train_score[1], train_score[0], val_score[1], val_score[0]))
    return ((train_score[0] + 3*val_score[0])/4), ((train_score[1] + 3*val_score[1])/4)

def calculate_score(logger, model, X_train, Y_train, X_test, Y_test):
    logger.info('Calculating model score')
    train_acc = model.evaluate(X_train, Y_train, verbose=0)
    val_acc = model.evaluate(X_test, Y_test, verbose=0)
    return ((train_acc[0] + val_acc[0])/2), ((train_acc[1] + val_acc[1])/2)

def find_layers_of_type(logger, model, layertype):
    logger.info("Finding convolutional layers")
    layerNames = [layer.name for layer in model.layers]
    convLayers = []
    for i in range(len(layerNames)):
        if layertype in layerNames[i]:
            convLayers.append(i)
    return convLayers

def find_trainable_layers(logger, model):
    logger.info("Finding all layers with weights")
    modelWeights = [x.get_weights() for x in model.layers]
    trainableLayers = []
    for i in range(len(modelWeights)):
        if len(modelWeights[i]) != 0:
            trainableLayers.append(i)
    return trainableLayers

