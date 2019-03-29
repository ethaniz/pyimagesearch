# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/03/28 19:45:52
@Description: 
'''

# here put the import lib

from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
import pdb

trainPaths = [x for x in paths.list_images(config.IMAGES_PATH) if '._' not in x]
trainLabels = [p.split(os.path.sep)[-1].split(".")[0]
    for p in trainPaths]

#pdb.set_trace()

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels,
    test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,
    random_state=42)

(trainPaths, testPaths, trainLabels, testLabels) = split

split = train_test_split(trainPaths, trainLabels,
    test_size=config.NUM_VAL_IMAGES, stratify=trainLabels,
    random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = split

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    print("[INFO] build {} ...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
        widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("[INFO] serializing means ...")
D = {"R": np.mean(R), "G":np.mean(G), "B":np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()