# -*- coding: utf-8 -*-
import cv2
import imutils
from numpy import ndarray

from ml.utils import is_array


class AspectAwarePreprocessor:
    """
    Class used to resize image while keeping the ratio
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

    def preprocess(self, image: ndarray) -> ndarray:
        """
        Resize an image
        Args:
            image (ndarray): image to resize

        Returns:
            (ndarray): new image
        """
        image = is_array(image)
        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW, dH = 0, 0

        # crop
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
