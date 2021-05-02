# -*- coding: utf-8 -*-
import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DatasetGenerator:
    """
    Generator for dataset
    """
    def __init__(self, db_path: str, batch_size: int, preprocessors: list = None, aug: ImageDataGenerator = None):
        """
        Initialization
        Args:
            db_path (str): Path of dataset stored
            batch_size (int): size of batch
            preprocessors (list): list of processors
            aug (ImageDataGenerator): Generator of data augmentation
        """
        # store
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug

        # open the HDF5 database
        self.db = h5py.File(db_path, "r")
        self.num_images = self.db["labels"].shape[0]

    def generator(self) -> ():
        """
        Continuously gives images and labels
        Returns:
            (): Array of images && Array of labels
        """
        for i in np.arange(0, self.num_images, self.batch_size):
            # extract the images and labels from the HDF dataset
            images = self.db["images"][i: i + self.batch_size]
            labels = self.db["labels"][i: i + self.batch_size]

            # if our preprocessors are not None
            if self.preprocessors is not None:
                # the list of processed images
                proc_images = []

                # loop over the images
                for image in images:
                    # loop over the preprocessors
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    proc_images.append(image)
                images = np.array(proc_images)

            # if the data augmentation exists, apply it
            if self.aug is not None:
                (images, labels) = next(
                    self.aug.flow(
                        images, labels, batch_size=self.batch_size
                    )
                )

            # yield a tuple of images and labels
            yield images, labels

    def close(self):
        """
        Close the generator
        """
        # close the database
        self.db.close()
