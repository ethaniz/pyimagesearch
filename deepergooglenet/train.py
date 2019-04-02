# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/04/01 19:36:25
@Description: 
'''

# here put the import lib

import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import EpochCheckPoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", required=True)
ap.add_argument("-m", "--model", type=str)
ap.add_argument("-s", "--start-epoch", type=int, default=0)
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode='nearest')

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
    preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
    preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] compiling model ...")
    model = DeeperGoogLeNet.build(width=64, height=64, depth=3,
        classes=config.NUM_CLASSES, reg=0.0002)
    #opt = Adam(1e-3)
    opt = SGD(1e-2, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy'])
else:
    print("[INFO] loading {} ...".format(args['model']))
    model = load_model(args['model'])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [
    EpochCheckPoint(args['checkpoint'], every=5, startAt=args['start_epoch']),
]

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 64,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 64,
    epochs=25,
    max_queue_size = 64 * 2,
    callbacks=callbacks, verbose=1
)

trainGen.close()
valGen.close()
