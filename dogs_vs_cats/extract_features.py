# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/03/30 14:23:57
@Description: 
'''

# here put the import lib

from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os
import pdb

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-b", "--batch-size", type=int, default=16)
ap.add_argument("-s", "--buffer-size", type=int, default=1000)
args = vars(ap.parse_args())

bs = args["batch_size"]

print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network ...")
model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

#pdb.set_trace()

dataset = HDF5DatasetWriter((len(imagePaths), 2048),
    args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
    widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i: i + bs]
    batchLabels = labels[i: i + bs]
    batchImages = []
    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    #pdb.set_trace()
    features = features.reshape((features.shape[0], 2048))
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
