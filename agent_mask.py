from logging import WARNING

import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout

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
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 7, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 7, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 7, activation='relu', padding='same')(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 7, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(32, 7, activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

model = create_segmentation_model()
print("model created")

model.summary()
#plot_model(model, show_shapes=True)
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

class SegmentationMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="segmentation_miou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits/probabilities to class predictions
        y_pred = tf.argmax(y_pred, axis=-1)

        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        self.mean_iou.update_state(y_true, y_pred)

    def result(self):
        return self.mean_iou.result()

    def reset_state(self):
        self.mean_iou.reset_state()

model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
              loss=combined_loss,
              metrics=[SegmentationMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES), "accuracy"])

history = model.fit(
    coco_train_and_test,
    epochs=30,
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

plt.plot(history.history['segmentation_miou'], label = "segmentation_miou")
plt.plot(history.history['val_segmentation_miou'], label = "val_segmentation_miou")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.legend()
plt.show()

model.evaluate(coco_train_and_test)

predictions = model.predict(coco_train_and_test)



#Predict
y_prediction = model.predict(coco_train_and_test)
y_prediction = np.argmax(y_prediction, axis=1)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Створення Confusion Matrix
cm = confusion_matrix(coco_train_and_test, y_prediction)

# 2. Відображення без чисел
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dflu.coco_mask_labels)
disp.plot(include_values=False,  # <- не показувати числа
          xticks_rotation=90,
          cmap='Blues',
          ax=ax)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()