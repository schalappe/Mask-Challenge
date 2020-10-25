# -*- coding: utf-8 -*-
import tensorflow as tf
from generator import HDF5DatasetGenerator
from preprocessors import ImageToArrayPreprocessor
from preprocessors import PatchPreprocessor
from preprocessors import SimplePreprocessor
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data augmentation
aug = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, fill_mode="nearest"
)
# initialization preprocessors
sp = SimplePreprocessor(224, 224)
pp = PatchPreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# ############ WARM UP #############
# initialize the number of epochs to train for and batch size
LR = 1e-2
BS = 32
# data generator
trainGen = HDF5DatasetGenerator(
    './output/train_set.hdf5', BS, aug=aug,
    preprocessors=[pp, iap]
)
valGen = HDF5DatasetGenerator(
    './output/val_set.hdf5', BS,
    preprocessors=[sp, iap]
)
# construct our model
print("[INFO]: Create model")
head = tf.keras.applications.NASNetMobile(
    input_shape=(224, 224, 3), include_top=False,
    weights='imagenet', classes=2
)
# Freeze the pretrained weights
head.trainable = False
# Rebuild top
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(head.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
# Compile
model = tf.keras.Model(head.input, outputs, name="NASNetMobile")
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)

# callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs/nasnet/',
        profile_batch=0
    )
]

# train the head of the network
print("[INFO] training: warm up ...")
H = model.fit(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // BS,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // BS,
    epochs=15,
    callbacks=callbacks, verbose=2
)

trainGen.close()
valGen.close()

# ############ FINE TURN #############
# unfreeze layers
for layer in model.layers[:-4]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)

# data generator
trainGen = HDF5DatasetGenerator(
    './output/train_set.hdf5', BS, aug=aug,
    preprocessors=[pp, iap]
)
valGen = HDF5DatasetGenerator(
    './output/val_set.hdf5', BS,
    preprocessors=[sp, iap]
)

# callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs/nasnet-fine/',
        profile_batch=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        './output/best_nasnet.h5', monitor='val_loss',
        mode='min', verbose=1,
        save_best_only=True, save_weights_only=False
    )
]

# train the head of the network
print("[INFO] training: fine tune...")
H = model.fit(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // BS,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // BS,
    epochs=25,
    callbacks=callbacks, verbose=2
)

trainGen.close()
valGen.close()
