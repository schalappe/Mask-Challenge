# -*- coding: utf-8 -*-
import csv
import os

import cv2
import tensorflow as tf
from imutils import paths
from tqdm import tqdm

from ml.config import IMAGES_PATH
from ml.config import MODEL_PATH
from ml.config import SUBMIT_PATH
from ml.preprocessors import AspectAwarePreprocessor
from ml.preprocessors import ImageToArrayPreprocessor

# list of images and shuffling
print("[INFO]: loading images ...")
imagePaths = list(paths.list_images(IMAGES_PATH))
print("[INFO]: {} images".format(len(imagePaths)))

# load the Model
print("[INFO]: loading model ...")
model = tf.keras.models.load_model(MODEL_PATH + 'best_nasnet.h5')

# create submit file
with open(SUBMIT_PATH + 'sub-nasnet-best.csv', "w") as submit:
    writer = csv.writer(submit)
    writer.writerow(["image", "target"])

# initialize preprocessor
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
# process on test image
for image_path in tqdm(imagePaths, ncols=100, desc="Prediction ..."):
    # load input image
    image = cv2.imread(image_path)

    # preprocessing
    image = aap.preprocess(image)
    image = iap.preprocess(image)
    image = image.reshape((1, 224, 224, 3))
    # prediction
    # pred = np.argmax(model.predict(image)[0][0])
    pred = model.predict(image)[0][0]

    # write on submit file
    with open(SUBMIT_PATH + 'sub-nasnet-best.csv', "a+", newline='') as submit:
        writer = csv.writer(submit)
        writer.writerow([str(image_path.split(os.path.sep)[-1]), str(pred)])

print("Prediction complete")
