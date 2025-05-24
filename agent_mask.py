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

BAD_MODEL_COEFFICIENT = 32 # reduces model size
BAD_DATASET_COEFFICIENT = 1 # reduces dataset size
dataset_loader.BATCH_SIZE = 1024
dataset_loader.IMAGE_SIZE = 128
EPOCHS = 5
#tf.debugging.set_log_device_placement(True)

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
    d = dice_loss(y_true, y_pred)
    return ce + d


import tensorflow as tf

class SegmentationMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, name="SegmentationMeanIoU", *, num_classes, image_size=dataset_loader.IMAGE_SIZE, **kwargs):
        # Filter out image_size from kwargs to avoid passing it to MeanIoU
        kwargs.pop('image_size', None)
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self.num_classes = num_classes
        self.image_size = image_size

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.ensure_shape(y_true, [None, self.image_size, self.image_size])
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_labels = tf.ensure_shape(y_pred_labels, [None, self.image_size, self.image_size])
        return super().update_state(y_true, y_pred_labels, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "image_size": self.image_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            name=config.get("name", "SegmentationMeanIoU"),
            num_classes=config["num_classes"],
            image_size=config.get("image_size", 128)
        )


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
"""
coco_train_and_test = dataset_loader.coco_RGB_dataset_precomputed('train', channels=1)
coco_val = dataset_loader.coco_RGB_dataset_precomputed('val', channels=1)

print("precomputing train")
train_tfrecord_path = dataset_loader.precompute_image_and_mask_dataset(
    split='train',
    train_img_dir=dataset_loader.coco_train_img_dir,
    channels=1  # Set to 1 for grayscale
)
print("precomputing val")
val_tfrecord_path = dataset_loader.precompute_image_and_mask_dataset(
    split='val',
    val_img_dir=dataset_loader.coco_val_img_dir,
    channels=1  # Set to 1 for grayscale
)"""
print("creating datasets")
# Create datasets
coco_train_and_test = dataset_loader.coco_RGB_dataset_precomputed(
    split='train',
    channels=1
)

coco_val = dataset_loader.coco_RGB_dataset_precomputed(
    split='val',
channels=1
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

train_steps, val_steps = dataset_loader.coco_cardinality()

for _, masks in coco_train_and_test.take(1):
    print("min/max mask IDs:", tf.reduce_min(masks), tf.reduce_max(masks))

for _, masks in coco_val.take(1):
    print("min/max mask IDs:", tf.reduce_min(masks), tf.reduce_max(masks))




def create_segmentation_model(input_shape=(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(MaxPooling2D((2)))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (4 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 2) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 4) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 4) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // BAD_MODEL_COEFFICIENT, (dataset_loader.IMAGE_SIZE // 8) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // BAD_MODEL_COEFFICIENT, (dataset_loader.IMAGE_SIZE // 8) * 2 + 1,
                     activation='relu', padding='same'))

    model.add(UpSampling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 4) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (2 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 4) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(UpSampling2D((2)))
    model.add(BatchNormalization())
    model.add(Conv2D(dataset_loader.IMAGE_SIZE // (4 * BAD_MODEL_COEFFICIENT), (dataset_loader.IMAGE_SIZE // 2) * 2 + 1,
                     activation='relu', padding='same'))
    model.add(UpSampling2D((2)))
    model.add(Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax'))
    return model



model = create_segmentation_model()
print("model created")

model.summary()
plot_model(model, show_shapes=True)

# Compile the model with the masked loss
model = create_segmentation_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=dice_loss,
    metrics=[SegmentationMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)]
)

model.save("models/precompv1_0.keras")
counter = 0
loss_list = []
val_loss_list = []
SMIoU_list = []
val_SMIoU_list = []
while counter < 500:
    model = tf.keras.models.load_model(f'models/precompv1_{counter}.keras', custom_objects={'dice_loss': dice_loss, 'combined_loss': combined_loss, "SegmentationMeanIoU": SegmentationMeanIoU})
    history = model.fit(
        coco_train_and_test,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=coco_val,
        validation_steps=val_steps
    )
    counter += EPOCHS
    model.save(f"models/precompv1_{counter}.keras")
    loss_list.append(np.mean(history.history['loss']))
    val_loss_list.append(np.mean(history.history['val_loss']))
    val_SMIoU_list.append(np.mean(history.history['val_SegmentationMeanIoU']))
    SMIoU_list.append(np.mean(history.history['SegmentationMeanIoU']))
model.save("night.keras")
plt.plot(np.linspace(0, counter, counter//EPOCHS),loss_list, label="loss")
plt.plot(np.linspace(0, counter, counter//EPOCHS), val_loss_list, label="val_loss")
plt.legend()
plt.show()

plt.plot(np.linspace(0, counter, counter//EPOCHS), SMIoU_list, label="SegmentationMeanIoU")
plt.plot(np.linspace(0, counter, counter//EPOCHS), val_SMIoU_list, label="val_SegmentationMeanIoU")
plt.legend()
plt.show()
"""
plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.legend()
plt.show()
"""

model.evaluate(coco_val, steps=val_steps)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
# Training set distribution
train_true_list = []
for _, masks in tqdm.tqdm(coco_train_and_test.take(train_steps), desc="Training Labels"):
    flat = tf.reshape(masks, [-1]).numpy()
    train_true_list.append(flat)
train_true = np.concatenate(train_true_list)
print("Training label distribution:", Counter(train_true))

# Validation set distribution
val_true_list = []
for _, masks in tqdm.tqdm(coco_val.take(val_steps), desc="Validation Labels"):
    flat = tf.reshape(masks, [-1]).numpy()
    val_true_list.append(flat)
val_true = np.concatenate(val_true_list)
print("Validation label distribution:", Counter(val_true))
"""

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
