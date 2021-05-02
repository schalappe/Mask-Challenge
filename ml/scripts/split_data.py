# -*- coding: utf-8 -*-
import shutil
import pandas as pd
import progressbar

# extracts from csv file
train_labels = pd.read_csv('../train_labels.csv', header='infer')

# progress bar
widgets = [
    "Split data: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()
]
pbar = progressbar.ProgressBar(maxval=len(train_labels), widgets=widgets).start()

# loop over label
for i, path in enumerate(train_labels["image"]):
    shutil.move("./images/"+path, './train/')
    pbar.update(i)
pbar.finish()
