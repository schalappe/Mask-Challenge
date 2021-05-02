# -*- coding: utf-8 -*-
import os

import h5py


class DatasetWriter:
    """
    Store data into h5py dataset
    """
    def __init__(self, dims: tuple, output_path: str, buf_size: int = 1000) -> None:
        """
        Initialization

        Args:
            dims (tuple): shape of dataset
            output_path (str): path where to store the dataset
            buf_size (int): length of the buffer
        """
        # check if the output path exists
        if os.path.exists(output_path):
            raise ValueError(
                "The output path already exists and cannot be "
                "overwritten. Manually delete it before continuing."
            )

        # store image/feature and class label
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset("images", dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # buffer size and initialization
        self.bufSize = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows: list, labels: list) -> None:
        """
        Add data to the buffer

        Args:
            rows (list): list of data
            labels (list): list of labels
        """
        # add the rows and the labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self) -> None:
        """
        Put data in dataset and empty th buffer
        """
        # write the buffer to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self) -> None:
        """
        Store the dataset into h5py file
        """
        # check if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
