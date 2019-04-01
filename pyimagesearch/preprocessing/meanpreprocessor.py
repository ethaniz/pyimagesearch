# -*- coding:utf8 -*-
'''
@Author: 异尘
@Date: 2019/03/29 10:17:44
@Description: 
'''

# here put the import lib
import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype('float32'))

        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])