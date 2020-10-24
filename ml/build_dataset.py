# -*- coding: utf-8 -*-
import pandas as pd
import progressbar
import cv2

from sklearn.model_selection import train_test_split
from preprocessors import AspectAwarePreprocessor
from generator import HDF5DatasetWriter


# extracts from csv file
train_labels = pd.read_csv('./train_labels.csv', header='infer')
# set
train_set = ['./train/' + image for image in train_labels.image]
label_set = [int(label) for label in train_labels.target]
# split
x_train, x_val, y_train, y_val = train_test_split(
    train_set, label_set,
    test_size=0.2, random_state=1337, stratify=label_set
)
# path output
datasets = [
    ("train", x_train, y_train, './output/train_set.hdf5'),
    ("val", x_val, y_val, './output/val_set.hdf5'),
]

# initialize preprocessor
aap = AspectAwarePreprocessor(256, 256)

# loop over dataset and write
for (dType, paths, labels, output) in datasets:
    print("[INFO] building {} ...".format(output))
    writer = HDF5DatasetWriter(
        (len(x_train), 256, 256, 3), output
    )
    widgets = [
        "Building Dataset: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()
    ]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()
    # loop over images
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load image
        image = cv2.imread(path)
        image = aap.preprocess(image)
        # add image and label to HDF5 dataset
        writer.add([image], [label])
        # show verbose
        pbar.update(i)
        # close the HDF5 writer
    pbar.finish()
    writer.close()
