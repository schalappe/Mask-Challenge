# -*- coding: utf-8 -*-
from numpy import ndarray


def is_array_class(func):
    def check_array(self, image):
        assert isinstance(image, ndarray), "Image must be an numpy array"
        return func(self, image)
    return check_array


