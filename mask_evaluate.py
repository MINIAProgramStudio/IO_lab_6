from logging import WARNING
import os

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
import datasets_from_loader_utils as dflu

STEPS_PER_EPOCH = 45
dataset_loader.BATCH_SIZE = 48
dataset_loader.IMAGE_SIZE = 128
#tf.debugging.set_log_device_placement(True)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("creating datasets")
# Create datasets
train_steps, val_steps = dataset_loader.coco_cardinality()
coco_train = dataset_loader.coco_RGB_dataset_precomputed(
    split='train',
    channels=1,
    tfrecord_path="tfrecords/image_mask_train.tfrecord"
)

coco_val = dataset_loader.coco_RGB_dataset_precomputed(
    split='val',
    channels=1,
    tfrecord_path="tfrecords/image_mask_val.tfrecord"
)


model = tf.keras.models.load_model('models/Agent1/unet48hsv_e79_l0.6199.keras', custom_objects={'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy, 'weighted_combined_loss': weighted_combined_loss, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES, weights = [0.7, 0.7, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1])})
print("model loaded")

model.evaluate(coco_val.take(val_steps))
print("model evaluated")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true_list = []
for _, masks in tqdm.tqdm(coco_val.take(val_steps), desc="a"):
    flat = tf.reshape(masks, [-1]).numpy()  # shape (batch*H*W,)
    y_true_list.append(flat)


y_pred_list = []
for batch_preds in tqdm.tqdm(model.predict(coco_val.take(val_steps)), desc="b"):
    preds_flat = np.argmax(batch_preds, axis=-1).reshape(-1)  # (batch*H*W,)
    y_pred_list.append(preds_flat)

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)

num_classes = len(dflu.coco_rgb_labels)  # 9
y_pred = np.clip(y_pred, 0, num_classes - 1)
y_true = np.clip(y_true, 0, num_classes - 1)

# Compute confusion matrix with all classes
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))  # Include all labels 0–8

# Use all label names
used_label_names = dflu.coco_rgb_labels  # All 9 labels

cm_log = np.log1p(cm)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=used_label_names)
disp.plot(include_values=False, xticks_rotation=90, cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()