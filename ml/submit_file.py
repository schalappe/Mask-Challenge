# necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import progressbar
import argparse
import pickle
import csv
import os

# argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
    help="path to output file")
ap.add_argument("-m", "--model", required=True,
    help="path model")
args = vars(ap.parse_args())

# list of images and shuffling
print("[INFO]: loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
print("[INFO]: {} images".format(len(imagePaths)))

# load the Logistic Model
print("[INFO]: loading model ...")
model = pickle.load(open(args["model"], "rb"))

# laod the ResNEt50 network
print("[INFO]: loading network ...")
res = ResNet50(weights="imagenet", include_top=False)

# create submit file
with open(args["output"], "w") as submit:
    writer = csv.writer(submit)
    writer.writerow(["image", "target"])

# progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# process on test image
for (i, imagePath) in enumerate(imagePaths):
    # load input image
    # resized in 224x224 pixels if necessary
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # preprocessing
    # expending dimensions
    # substracting the mean RGB pixel
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    feature = res.predict(image)
    feature = feature.reshape((feature.shape[0], 100352))

    # prediction
    pred = model.predict(feature)

    # write on submit file
    with open(args["output"], "a+", newline='') as submit:
        writer = csv.writer(submit)
        writer.writerow([str(imagePath.split(os.path.sep)[-1]), str(pred[0])])
    #print("store {} with value {}".format(imagePath.split(os.path.sep)[-1], pred[0]))

    pbar.update(i)

# close dataset
pbar.finish()