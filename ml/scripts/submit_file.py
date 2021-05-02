# -*- coding: utf-8 -*-
import csv
import os
import cv2
import progressbar
import tensorflow as tf
from imutils import paths
from preprocessors import AspectAwarePreprocessor
from preprocessors import ImageToArrayPreprocessor

# list of images and shuffling
print("[INFO]: loading images ...")
imagePaths = list(paths.list_images('./images'))
print("[INFO]: {} images".format(len(imagePaths)))

# load the Model
print("[INFO]: loading model ...")
model = tf.keras.models.load_model('./output/best_nasnet.h5')

# create submit file
with open('./output/sub-nasnet-best.csv', "w") as submit:
    writer = csv.writer(submit)
    writer.writerow(["image", "target"])

# progress bar
widgets = [
    "Predict: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()
]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# initialize preprocessor
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
# process on test image
for (i, image_path) in enumerate(imagePaths):
    # load input image
    image = cv2.imread(image_path)

    # preprocessing
    image = aap.preprocess(image)
    image = iap.preprocess(image)
    image = image.reshape(1, 224, 224, 3)
    # prediction
    # pred = np.argmax(model.predict(image)[0][0])
    pred = model.predict(image)[0][0]

    # write on submit file
    with open('./output/sub-nasnet-best.csv', "a+", newline='') as submit:
        writer = csv.writer(submit)
        writer.writerow([str(image_path.split(os.path.sep)[-1]), str(pred)])

    pbar.update(i)

# close dataset
pbar.finish()
