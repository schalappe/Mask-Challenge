# -*- coding: utf-8 -*-
from numpy import ndarray


def is_array(image):
    assert isinstance(image, ndarray), "Image must be an numpy array"
    return image
