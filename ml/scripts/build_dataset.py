# -*- coding: utf-8 -*-
"""
    Script used to create a h5py file that contains the data for training
"""
import cv2
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml.config import FEATURES_PATH
from ml.config import INPUT_CSV
from ml.config import TRAIN_PATH
from ml.generator import DatasetWriter
from ml.preprocessors import AspectAwarePreprocessor

# extracts from csv file
train_labels = read_csv(INPUT_CSV, header='infer')

# set
train_set = [TRAIN_PATH + image for image in train_labels.image]
label_set = [int(label) for label in train_labels.target]

# split
x_train, x_val, y_train, y_val = train_test_split(
    train_set, label_set,
    test_size=0.2, random_state=1337, stratify=label_set
)

# path output
datasets = [
    ("train", x_train, y_train, FEATURES_PATH + 'train_set.hdf5'),
    ("test", x_val, y_val, FEATURES_PATH + 'test_set.hdf5'),
]

# initialize preprocessor
aap = AspectAwarePreprocessor(256, 256)

# loop over dataset and write
for (dType, paths, labels, output) in datasets:
    print("[INFO] building {} ...".format(output))
    writer = DatasetWriter(
        (len(x_train), 256, 256, 3), output
    )

    # loop over images
    for (path, label) in tqdm(zip(paths, labels), ncols=100, desc="Loop over images: "):
        # load image
        image = cv2.imread(path)
        image = aap.preprocess(image)
        # add image and label to HDF5 dataset
        writer.add([image], [label])

    # close the HDF5 writer
    writer.close()
