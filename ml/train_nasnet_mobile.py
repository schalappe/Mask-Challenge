# -*- coding: utf-8 -*-
import tensorflow as tf

from ml.config import FEATURES_PATH
from ml.config import LOGS_PATH
from ml.config import MODEL_PATH
from ml.generator import DatasetGenerator
from ml.preprocessors import ImageToArrayPreprocessor
from ml.preprocessors import PatchPreprocessor
from ml.preprocessors import SimplePreprocessor

# data augmentation
aug = tf.keras.preprocessing.image.ImageDataGenerator(
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
OUTPUT_SHAPE = ((None, 224, 224, 3), (None,))

# data generator
train_generator = DatasetGenerator(
    FEATURES_PATH + 'train_set.hdf5', BS, aug=aug,
    preprocessors=[pp, iap]
)
train_set = tf.data.Dataset.from_generator(
    train_generator.generator,
    output_types=(tf.int64, tf.int64),
    output_shapes=OUTPUT_SHAPE
).prefetch(tf.data.AUTOTUNE)

test_generator = DatasetGenerator(
    FEATURES_PATH + 'test_set.hdf5', BS,
    preprocessors=[sp, iap]
)
test_set = tf.data.Dataset.from_generator(
    test_generator.generator,
    output_types=(tf.int64, tf.int64),
    output_shapes=OUTPUT_SHAPE
).prefetch(tf.data.AUTOTUNE)

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
        log_dir=LOGS_PATH+'nasnet/',
        profile_batch=0
    )
]

# train the head of the network
print("[INFO] training: warm up ...")
H = model.fit(
    train_set,
    validation_data=test_set,
    epochs=15,
    callbacks=callbacks
)

# ############ FINE TURN #############
# unfreeze layers
for layer in model.layers[:-4]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)

# callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_PATH+'nasnet-fine/',
        profile_batch=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH+'best_nasnet.h5', monitor='val_loss',
        mode='min', verbose=1,
        save_best_only=True, save_weights_only=False
    )
]

# train the head of the network
print("[INFO] training: fine tune...")
H = model.fit(
    train_set,
    validation_data=test_set,
    epochs=25,
    callbacks=callbacks
)

train_generator.close()
test_generator.close()
