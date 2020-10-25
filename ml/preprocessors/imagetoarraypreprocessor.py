# -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        # store image
        self.dataFormat = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)/255.0
