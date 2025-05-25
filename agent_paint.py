## IMPORTS
import dataset_loader as dl
import datasets_from_loader_utils as dflu
from some_functions import *

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
layers = tf.keras.layers

## PARAMETERS
dl.IMAGE_SIZE = 128
dl.BATCH_SIZE = 32
EPOCHS = 2

MODEL_PATH = "models/max128_5_12.keras"

TFRECORD_PATH_TRAIN = "tfrecords/image_mask_train.tfrecord"  # 128x128 train precomputed images and masks
# TFRECORD_PATH_TRAIN = "tfrecords/train_32.tfrecord"  # 32x32 train precomputed images and masks
TFRECORD_PATH_VAL = "tfrecords/image_mask_val.tfrecord"  # 128x128 test precomputed images and masks
# TFRECORD_PATH_VAL = "tfrecords/val_32.tfrecord"  # 32x32 test precomputed images and masks

TRAIN_STEPS, VAL_STEPS = dl.coco_cardinality()
DATASET_RATIO = 1  # 1/?

## LOAD MODEL FROM AGENT 1
msh_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_combined_loss': weighted_combined_loss,
        'WeightedMeanIoU': WeightedMeanIoU(num_classes=dl.COCO_NUM_CLASSES)
    }
)

## LOAD TRAIN AND VAL DATASETS TODO: ADD RGB IMAGES IF THIS IS NECESSARY
coco_train = dl.coco_RGB_dataset_precomputed(
    split='train',
    channels=1,
    tfrecord_path=TFRECORD_PATH_TRAIN
).take(TRAIN_STEPS//DATASET_RATIO)

coco_val = dl.coco_RGB_dataset_precomputed(
    split='val',
    channels=1,
    tfrecord_path=TFRECORD_PATH_VAL
).take(VAL_STEPS//DATASET_RATIO)

## CREATE MODEL FOR AGENT 2
# TODO: CHECK IF THIS IS NECESSARY
resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(dl.IMAGE_SIZE, dl.IMAGE_SIZE),
  tf.keras.layers.Rescaling(1./255)
])

# TODO: CHECK IF THIS IS NECESSARY
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomZoom(0.2),
])

# TODO: RECEIVE GRAYSCALE IMAGE, PREDICT MASK USING AGENT 1 MODEL, MAKE RGB IMAGE USING AGENT 2 MODEL TAKING GRAYSCALE IMAGE AND MASK AS BASE, BUT GENERETE RGB IMAGE BY ITSELF
model = tf.keras.models.Sequential(
    [
        layers.Input(shape=(dl.IMAGE_SIZE, dl.IMAGE_SIZE, 1)),  # RECEIVE (?, 128, 128, 1), values from 0 to 1 or 0 to 255
        # layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

        layers.Conv2D(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(4, 4),

        # layers.Conv2D(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(2, 2),

        layers.Dropout(0.3),

        # layers.UpSampling2D(2),
        # layers.Conv2DTranspose(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same'),
        # layers.BatchNormalization(),

        layers.UpSampling2D(4),
        layers.Conv2DTranspose(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.UpSampling2D(2),
        layers.Conv2DTranspose(dl.IMAGE_SIZE // 2, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(dl.IMAGE_SIZE // 4, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(dl.IMAGE_SIZE // 8, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(3, 1, activation='softmax')  # RETURN (?, 128, 128, 3), values from 0 to 1 or 0 to 255
    ]
)

model.summary()

# TODO: VALIDATE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae']
)

## TRAIN MODEL
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH.split(".")[0] + "_{val_loss:.2f}.keras",
        monitor='val_loss', save_best_only=True
    )
]
history = model.fit(
    coco_train,
    epochs=EPOCHS,
    # train_steps=train_steps,
    validation_data=coco_val,
    # validation_steps=val_steps,
    callbacks=callbacks
)
