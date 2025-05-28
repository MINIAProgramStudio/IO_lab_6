from logging import WARNING
import os

from keras import Model
from keras.metrics import SparseCategoricalAccuracy
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tqdm
from keras.layers import Conv2DTranspose

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy
import keras

from pycocotools.coco import COCO
from tensorflow.python.ops.gen_experimental_dataset_ops import data_service_dataset

from some_functions import *

import numpy as np

import dataset_loader
train_steps, val_steps = dataset_loader.coco_cardinality()
import datasets_from_loader_utils as dflu

BAD_MODEL_COEFFICIENT = 1 # reduces model size
TRAIN_STEPS = train_steps
VAL_STEPS = val_steps
dataset_loader.BATCH_SIZE = 96
dataset_loader.IMAGE_SIZE = 128
#EPOCHS = 25
#START_EPOCH = 0
TOTAL_EPOCHS = 2000
AGENT1_MODELS = "models/Agent1/"
MODEL_SAVE_PATH = AGENT1_MODELS + "unet64filtered_e{epoch:02d}_l{val_loss:.4f}.keras"
#tf.debugging.set_log_device_placement(True)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
"""
print("precomputing train")
train_tfrecord_path = dataset_loader.precompute_image_and_mask_dataset(
    split='train',
    train_img_dir=dataset_loader.coco_train_img_dir,
    channels=1,
    output_tfrecord_path="FAILSAFE.tfrecord"
)

print("precomputing val")
val_tfrecord_path = dataset_loader.precompute_image_and_mask_dataset(
    split='val',
    val_img_dir=dataset_loader.coco_val_img_dir,
    channels=1,
    output_tfrecord_path="FAILSAFE.tfrecord"
)
exit()"""
print("creating datasets")
# Create datasets

coco_train = dataset_loader.coco_RGB_dataset_precomputed(
    split='train',
    channels=1,
    tfrecord_path="tfrecords/128filtered_train.tfrecord"
)

coco_val = dataset_loader.coco_RGB_dataset_precomputed(
    split='val',
    channels=1,
    tfrecord_path="tfrecords/128filtered_val.tfrecord"
)
print("MS COCO loaded.")
"""
coco_test, coco_train = dflu.split_test_and_train(coco_train_and_test)
del coco_train_and_test
print("COCO split completed.")

print("Some COCO labels:")
dflu.first_batch_labels(coco_test, dflu.coco_labels)"""

#dflu.first_batch_images(coco_train_and_test)

# dflu.first_batch_masks(coco_train_and_test)



for _, masks in coco_train.take(1):
    print("min/max mask IDs:", tf.reduce_min(masks), tf.reduce_max(masks))

for _, masks in coco_val.take(1):
    print("min/max mask IDs:", tf.reduce_min(masks), tf.reduce_max(masks))



"""
def create_segmentation_model(input_shape=(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    #model.add(MaxPooling2D((1)))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (4 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 1) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 2) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 2) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // BAD_MODEL_COEFFICIENT, (dataset_loader.IMAGE_SIZE // 4) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // BAD_MODEL_COEFFICIENT, (dataset_loader.IMAGE_SIZE // 4) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2)))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 2) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 2) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2)))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (4 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 1) * 2 + 1,
                     activation='relu', padding='same'))
    #model.add(UpSampling2D((1)))
    model.add(Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax'))
    return model


tf.keras.backend.clear_session()
#model = create_segmentation_model()
KERNEL_SIZE = 3

model = tf.keras.models.Sequential(
    [
        Input(shape=(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE, 1)),
        # layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        Dropout(0.3),
        # layers.UpSampling2D(4),
        # layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Dropout(0.3),
        BatchNormalization(),
        UpSampling2D(2),
        # layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'),
        Conv2DTranspose(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        UpSampling2D(2),
        Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        UpSampling2D(2),
        Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax')
    ]
)
"""

resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE),
    tf.keras.layers.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(0.2),
])
"""
model = tf.keras.models.Sequential(
    [
        Input(shape=(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE, 1)),
        #resize_and_rescale,
        #data_augmentation,

        Conv2D(dataset_loader.IMAGE_SIZE, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(dataset_loader.IMAGE_SIZE, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(4, 4),

        Conv2D(dataset_loader.IMAGE_SIZE, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Dropout(0.3),

        UpSampling2D(2),
        Conv2DTranspose(dataset_loader.IMAGE_SIZE, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),

        UpSampling2D(2),
        Conv2DTranspose(dataset_loader.IMAGE_SIZE, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),

        UpSampling2D(4),
        Conv2DTranspose(dataset_loader.IMAGE_SIZE // 2, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),

        Conv2DTranspose(dataset_loader.IMAGE_SIZE // 4, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),

        Conv2DTranspose(dataset_loader.IMAGE_SIZE // 8, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax')
        # Conv2D(3, 1, activation='softmax')
    ]
)"""
UNET_BASE = 64
DROPOUT = 0.2
def build_unet(input_shape=(None, None, 1), num_classes=9):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(UNET_BASE, 3, activation='relu', padding='same')(inputs)
    c1 = Dropout(DROPOUT)(c1)
    c1 = Conv2D(UNET_BASE, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    d1 = Dropout(DROPOUT)(p1)

    c2 = Conv2D(UNET_BASE*2, 3, activation='relu', padding='same')(d1)
    c2 = Dropout(DROPOUT)(c2)
    c2 = Conv2D(UNET_BASE*2, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    d2 = Dropout(DROPOUT)(p2)

    # Bottleneck
    b = Conv2D(UNET_BASE*4, 3, activation='relu', padding='same')(d2)
    b = Dropout(DROPOUT)(b)
    b = Conv2D(UNET_BASE*4, 3, activation='relu', padding='same')(b)

    # Decoder
    u1 = UpSampling2D()(b)
    u1 = tf.keras.layers.concatenate([u1, c2])
    ub = Dropout(DROPOUT)(u1)
    c3 = Conv2D(UNET_BASE*2, 3, activation='relu', padding='same')(ub)
    c3 = Dropout(DROPOUT)(c3)
    c3 = Conv2D(UNET_BASE*2, 3, activation='relu', padding='same')(c3)

    d3 = Dropout(DROPOUT)(c3)

    u2 = UpSampling2D()(d3)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c4 = Conv2D(UNET_BASE, 3, activation='relu', padding='same')(u2)
    c4 = Dropout(DROPOUT)(c4)
    c4 = Conv2D(UNET_BASE, 3, activation='relu', padding='same')(c4)
    c4 = Dropout(DROPOUT)(c4)
    outputs = Conv2D(num_classes, 1, activation='softmax')(c4)
    return Model(inputs, outputs)
model = build_unet()
print("model created")

model.summary()
plot_model(model, show_shapes=True)

# Compile the model with the masked loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=weighted_sparse_categorical_crossentropy,
    #metrics=[WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES), TopKCategoricalAccuracy(k=2), SparseCategoricalAccuracy()]
    metrics=[WeightedMeanIoU(), SparseCategoricalAccuracy()]
)
"""
counter = START_EPOCH
loss_list = list()
val_loss_list = list()
SMIoU_list = list()
val_SMIoU_list = list()
while counter < TOTAL_EPOCHS:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(f'models/unet32hsv_{counter}.keras', custom_objects={'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)})
    history = model.fit(

        coco_train,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=coco_val,
        validation_steps = val_steps//9
    )

    counter += EPOCHS
    model.save(f"models/unet32hsv_{counter}.keras")
    loss_list += history.history['loss']
    val_loss_list+=history.history['val_loss']
    val_SMIoU_list+=history.history['val_weighted_mean_iou']
    SMIoU_list+=history.history['weighted_mean_iou']
plt.plot(loss_list, label="loss")
plt.plot(val_loss_list, label="val_loss")
plt.legend()
plt.show()

plt.plot(SMIoU_list, label="weighted_mean_iou")
plt.plot(val_SMIoU_list, label="val_weighted_mean_iou")
plt.legend()
plt.show()
"""
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_freq='epoch'
    )
]
history = model.fit(
    coco_train,
    epochs=TOTAL_EPOCHS,
    steps_per_epoch=TRAIN_STEPS,
    validation_data=coco_val,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

"""
model_dir = 'models/'


# Evaluate all models
def evaluate_all_models(model_dir):

    for model_name in os.listdir(model_dir):
        tf.keras.backend.clear_session()
        model_path = os.path.join(model_dir, model_name)

        try:
            if model_name.endswith('.keras'):
                model = tf.keras.models.load_model(model_path, custom_objects={'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy, 'dice_loss': dice_loss, 'segmentationmeaniou': SegmentationMeanIoU(num_classes=9), 'weighted_combined_loss': weighted_combined_loss, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES, weights = [0.7, 0.7, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1])})
            else:
                print(f"Skipping unsupported file: {model_name}")
                continue

            print(f"Evaluating model: {model_name}")
            loss, acc = model.evaluate(coco_val.take(val_steps))

        except Exception as e:
            print(f"Error loading {model_name}: {e}")

# Run the evaluation
evaluate_all_models(model_dir)
exit()"""

