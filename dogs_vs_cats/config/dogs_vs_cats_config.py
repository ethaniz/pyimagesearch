# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/03/28 19:40:20
@Description: 
'''

# here put the import lib
IMAGES_PATH = '/data/tangle/pyimagesearch/dogs_vs_cats/train'

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

TRAIN_HDF5 = '/data/tangle/pyimagesearch/dogs_vs_cats/hdf5/train.hdf5'
VAL_HDF5 = '/data/tangle/pyimagesearch/dogs_vs_cats/hdf5/val.hdf5'
TEST_HDF5 = '/data/tangle/pyimagesearch/dogs_vs_cats/hdf5/test.hdf5'

MODEL_PATH = 'output/tmp.model'
DATASET_MEAN = 'output/mean.json'
OUTPUT_PATH = 'output'

