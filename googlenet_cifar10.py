import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import os
import pdb

NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data ...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

#pdb.set_trace()

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1,
    height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

callbacks = [LearningRateScheduler(poly_decay)]

print("[INFO] compiling model ...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.summary()

#pdb.set_trace()

print("[INFO] training_network ...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64,
    epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

print("[INFO] serializing model ...")
model.save(args["model"])