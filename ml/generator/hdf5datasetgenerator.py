# -*- coding: utf-8 -*-
import h5py
import numpy as np


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, preprocessors=None, aug=None, classes=7):
        # store
        self.batchSize = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.classes = classes

        # open the HDF5 database
        self.db = h5py.File(db_path, "r")
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

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
                            images, labels, batch_size=self.batchSize
                        )
                    )

                # yield a tuple of images and labels
                yield images, labels

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()
