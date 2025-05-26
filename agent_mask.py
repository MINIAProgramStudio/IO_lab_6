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

BAD_MODEL_COEFFICIENT = 1 # reduces model size
STEPS_PER_EPOCH = 45
dataset_loader.BATCH_SIZE = 2**6
dataset_loader.IMAGE_SIZE = 128
EPOCHS = 20
TOTAL_EPOCHS = 3000
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
)"""
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
"""

tf.keras.backend.clear_session()
#model = create_segmentation_model()
KERNEL_SIZE = 3

model = tf.keras.models.Sequential(
    [
        Input(shape=(dataset_loader.IMAGE_SIZE, dataset_loader.IMAGE_SIZE, 1)),
        # layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),

        Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(4, 4),

        # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(4, 4),

        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(4, 4),

        Dropout(0.3),

        # layers.UpSampling2D(4),
        # layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        Conv2D(1024, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),
        UpSampling2D(4),
        # layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'),
        Conv2DTranspose(256, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),

        UpSampling2D(4),
        Conv2DTranspose(256, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'),
        BatchNormalization(),

        Conv2D(dataset_loader.COCO_NUM_CLASSES, 1, activation='softmax')
    ]
)


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

print("model created")

model.summary()
plot_model(model, show_shapes=True)

# Compile the model with the masked loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=weighted_sparse_categorical_crossentropy,
    #metrics=[WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES), TopKCategoricalAccuracy(k=2), SparseCategoricalAccuracy()]
    metrics=[WeightedMeanIoU(), SparseCategoricalAccuracy()]
)

model.save("models/optimus_0.keras")

counter = 0
loss_list = []
val_loss_list = []
SMIoU_list = []
val_SMIoU_list = []
while counter < TOTAL_EPOCHS:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(f'models/optimus_{counter}.keras', custom_objects={'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)})
    history = model.fit(

        coco_train,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=coco_val,
        validation_steps = val_steps//4
    )

    counter += EPOCHS
    model.save(f"models/optimus_{counter}.keras")
    loss_list.append(np.mean(history.history['loss']))
    val_loss_list.append(history.history['val_loss'])
    val_SMIoU_list.append(history.history['val_weighted_mean_iou'])
    SMIoU_list.append(np.mean(history.history['weighted_mean_iou']))
plt.plot(loss_list, label="loss")
plt.plot(val_loss_list, label="val_loss")
plt.legend()
plt.show()

plt.plot(SMIoU_list, label="weighted_mean_iou")
plt.plot(val_SMIoU_list, label="val_weighted_mean_iou")
plt.legend()
plt.show()
"""
model_dir = 'models/'


# Evaluate all models
def evaluate_all_models(model_dir):

    for model_name in os.listdir(model_dir):
        tf.keras.backend.clear_session()
        model_path = os.path.join(model_dir, model_name)

        try:
            if model_name.endswith('.keras'):
                model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'segmentationmeaniou': SegmentationMeanIoU(num_classes=9), 'weighted_combined_loss': weighted_combined_loss, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES, weights = [0.7, 0.7, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1])})
            else:
                print(f"Skipping unsupported file: {model_name}")
                continue

            print(f"Evaluating model: {model_name}")
            loss, acc = model.evaluate(coco_val)
            print(f"  → Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")

        except Exception as e:
            print(f"Error loading {model_name}: {e}")

# Run the evaluation
evaluate_all_models(model_dir)
exit()
#model = tf.keras.models.load_model('models/st32_30.keras', custom_objects={'weighted_combined_loss': weighted_combined_loss, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES, weights = [0.7, 0.7, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1]})
print("model loaded")

model.evaluate(coco_val)
print("model evaluated")



""""""
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
