#-*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/04/02 14:13:35
@Description: 
'''

# here put the import lib

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckPoint
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-m", "--model", type=str)
ap.add_argument("-s", "--start_epoch", type=int, default=0)

args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data ...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float')
testX = testX.astype('float')

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, fill_mode='nearest')

if args['model'] is None:
    print("[INFO] compiling model ...")
    opt = SGD(lr=1e-1, momentum=0.9)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
        (16, 16, 32, 64), reg=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy'])
else:
    print('[INFO] loading {} ...'.format(args['model']))
    model = load_model(args['model'])

    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)
    ))       
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)
    ))

callbacks = [
    EpochCheckPoint(args['checkpoints'], every=5, startAt=args['start_epoch'])
]

print("[INFO] training network ...")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size=128),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 128, epochs=10,
    callbacks=callbacks, verbose=1
)
