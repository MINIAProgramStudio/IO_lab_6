from logging import WARNING

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tqdm

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



import numpy as np

import dataset_loader
import datasets_from_loader_utils as dflu

def dice_loss(y_true, y_pred, smooth=1e-6):
    # y_true: (batch, h, w)     — int32 labels in [0, num_classes)
    # y_pred: (batch, h, w, c)  — float32 softmax probabilities

    num_classes = tf.shape(y_pred)[-1]
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)  # (b,h,w,c)

    # Flatten
    y_true_f = tf.reshape(y_true_onehot, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)  # mean over all classes

def combined_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    d  = dice_loss(y_true, y_pred)
    return ce + d

class SegmentationMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, name="SegmentationMeanIoU", *, num_classes, **kwargs):
        # Ensure `name` is first and `num_classes` is keyword-only
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred comes in as (batch, H, W, C) softmax probabilities.
        # Convert to (batch, H, W) integer labels before computing IoU.
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred_labels, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        # Keras will pass in both 'name' and 'num_classes' here
        return cls(name=config.get("name"), num_classes=config["num_classes"])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Loading MS COCO dataset.")
coco_train_and_test = dataset_loader.coco_simple_segmentation_dataset('train',channels=1)
coco_val = dataset_loader.coco_simple_segmentation_dataset('val',channels=1)
print("MS COCO loaded. Splitting it into train and test.")
"""
coco_test, coco_train = dflu.split_test_and_train(coco_train_and_test)
del coco_train_and_test
print("COCO split completed.")

print("Some COCO labels:")
dflu.first_batch_labels(coco_test, dflu.coco_labels)"""

#dflu.first_batch_images(coco_train_and_test)

#dflu.first_batch_masks(coco_train_and_test)

train_steps, val_steps = dataset_loader.coco_cardinality()

for _, masks in coco_train_and_test.take(1):
    print("min/max mask IDs:", tf.reduce_min(masks), tf.reduce_max(masks))

for _, masks in coco_val.take(1):
    print("min/max mask IDs:", tf.reduce_min(masks), tf.reduce_max(masks))

def create_segmentation_model(input_shape=(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(MaxPooling2D((2)))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE//4, (dataset_loader.IMAGE_SIZE//2)*2+1, activation='relu', padding='same'))
    model.add(MaxPooling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE//2, (dataset_loader.IMAGE_SIZE//4)*2+1, activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // 2, (dataset_loader.IMAGE_SIZE // 4) * 2 + 1, activation='relu', padding='same'))
    model.add(MaxPooling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE, (dataset_loader.IMAGE_SIZE//8)*2+1, activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE, (dataset_loader.IMAGE_SIZE // 8) * 2 + 1, activation='relu', padding='same'))

    model.add(UpSampling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE//2, (dataset_loader.IMAGE_SIZE//4)*2+1, activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // 2, (dataset_loader.IMAGE_SIZE // 4) * 2 + 1, activation='relu', padding='same'))
    model.add(UpSampling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE//4, (dataset_loader.IMAGE_SIZE//2)*2+1, activation='relu', padding='same'))
    model.add(UpSampling2D((2)))
    model.add(Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax'))
    return model
#"""
model = create_segmentation_model()
print("model created")

model.summary()
plot_model(model, show_shapes=True)


def masked_loss(y_true, y_pred):
    mask = tf.cast(y_true != 4, tf.float32)

    # Compute the loss with masking
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    masked_loss = loss * mask

    # Average over non-masked pixels
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-6)  # Avoid division by zero


# Compile the model with the masked loss
model = create_segmentation_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=masked_loss,
    metrics=[SegmentationMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)]
)

history = model.fit(
    coco_train_and_test,
    epochs=10,
    steps_per_epoch=train_steps,
    validation_data=coco_val,
    validation_steps=val_steps
)
print(history.history.keys())
model.save("test.keras")

plt.plot(history.history['loss'], label = "loss")
plt.plot(history.history['val_loss'], label = "val_loss")
plt.legend()
plt.show()

plt.plot(history.history['SegmentationMeanIoU'], label = "SegmentationMeanIoU")
plt.plot(history.history['val_SegmentationMeanIoU'], label = "val_SegmentationMeanIoU")
plt.legend()
plt.show()
"""
plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.legend()
plt.show()
"""
#model = tf.keras.models.load_model('test.keras', custom_objects={'combined_loss': combined_loss, "SegmentationMeanIoU": SegmentationMeanIoU})
model.evaluate(coco_train_and_test, steps=train_steps)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



y_true_list = []
for _, masks in tqdm.tqdm(coco_val.take(val_steps), desc = "a"):
    # masks: (batch, H, W)
    flat = tf.reshape(masks, [-1]).numpy()         # shape (batch*H*W,)
    y_true_list.append(flat)

# 2) Run model.predict once and flatten predictions
y_pred_list = []
for batch_preds in tqdm.tqdm(model.predict(coco_val.take(val_steps)), desc = "b"):
    # batch_preds: (batch, H, W, num_classes)
    preds_flat = np.argmax(batch_preds, axis=-1).reshape(-1)  # (batch*H*W,)
    y_pred_list.append(preds_flat)

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)

num_classes = len(dflu.coco_rectangular_labels)
y_pred = np.clip(y_pred, 0, num_classes - 1)
y_true = np.clip(y_true, 0, num_classes - 1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Only use labels that appear in either y_true or y_pred
used_labels = np.unique(np.concatenate([y_true, y_pred]))
used_label_names = [dflu.coco_merged_mask_labels[i] for i in used_labels]

cm_log = np.log1p(cm)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=used_label_names)
disp.plot(include_values=False, xticks_rotation=90, cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()