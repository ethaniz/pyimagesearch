# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/04/02 15:33:39
@Description: 
'''

# here put the import lib

from os import path

BASE_PATH = '/data/tangle/pyimagesearch/datasets/challenges-in-representation-learning-facial-expression-recognition-challenge'

INPUT_PATH = path.sep.join([BASE_PATH, 'fer2013/fer2013.csv'])

NUM_CLASSES = 6

TRAIN_HDF5 = path.sep.join([BASE_PATH, 'hdf5/train.hdf5'])
VAL_HDF5 = path.sep.join([BASE_PATH, 'hdf5/val.hdf5'])
TEST_HDF5 = path.sep.join([BASE_PATH, 'hdf5/test.hdf5'])

BATCH_SIZE = 128

OUTPUT_PATH = path.sep.join([BASE_PATH, 'output'])
