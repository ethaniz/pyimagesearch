# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/04/01 16:08:43
@Description: 
'''

# here put the import lib

from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
import pdb

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

pdb.set_trace()

split = train_test_split(trainPaths, trainLabels,
    test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,
    random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

M = open(config.VAL_MAPPINGS).read().strip().split('\n')

M = [r.split('\t')[:2] for r in M]

valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])

