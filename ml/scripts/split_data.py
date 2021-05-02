# -*- coding: utf-8 -*-
"""
    Script used to split data into train and test set
"""
import shutil

from pandas import read_csv
from tqdm import tqdm

from ml.config import IMAGES_PATH
from ml.config import INPUT_CSV
from ml.config import TRAIN_PATH

# extracts from csv file
train_labels = read_csv(INPUT_CSV, header='infer')

# loop over label
for path in tqdm(train_labels["image"], ncols=100, desc="Move image for training: "):
    shutil.move(IMAGES_PATH+path, TRAIN_PATH)

print("Successfully complete")
