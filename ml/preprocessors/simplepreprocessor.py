# -*- coding: utf-8 -*-
import cv2
from numpy import ndarray

from ml.utils import is_array


class SimplePreprocessor:
    """
    Class used to resize image
    """

    def __init__(self, width: int, height: int, inter: int = cv2.INTER_AREA) -> None:
        """
        Initialization

        Args:
            width (int): new width of image
            height (int): new height of image
            inter (int): interpolation for resizing
        """
        # store the target image width, height, and interpolation
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: ndarray) -> None:
        """
        Resize an image
        Args:
            image (ndarray): image to resize

        Returns:
            (ndarray): new image
        """
        image = is_array(image)
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
