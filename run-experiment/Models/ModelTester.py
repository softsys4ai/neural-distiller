from VGG16 import VGG16
vgg16 = VGG16('cifar100', "/Users/blakeedwards/Desktop/Repos/research/neural-distiller-2/run-experiment/Models/ModelCheckpoints", "/Users/blakeedwards/Desktop/Repos/research/neural-distiller-2/run-experiment/Logs", 150, 100, 0.01)

model = vgg16.build()

from tensorflow.keras.optimizers import *
sgd = SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

train_score = model.evaluate(vgg16.train_x, vgg16.train_y, verbose=0)
val_score = model.evaluate(vgg16.test_x, vgg16.test_y, verbose=0)
print(train_score, val_score)

from tensorflow.keras.callbacks import *
model.fit(vgg16.train_x, vgg16.train_y,
                  batch_size=100,
                  epochs=150,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_acc', patience=8, min_delta=0.00007)],
                  validation_data=(vgg16.test_x, vgg16.test_y))