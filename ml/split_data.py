import pandas as pd
import progressbar
import argparse
import shutil

# argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-l", "--label", required=True,
    help="path to input label")
args = vars(ap.parse_args())

# extracts from csv file
train_labels = pd.read_csv(args["label"], header='infer')

# progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(train_labels), widgets=widgets).start()

i = 0
for path in train_labels["image"]:
    shutil.move("./images/"+path, args["dataset"])
    i += 1
    pbar.update(i)

pbar.finish()