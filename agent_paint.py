## IMPORTS
import dataset_loader as dl
import datasets_from_loader_utils as dflu
from some_functions import *

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
layers = tf.keras.layers

import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


## PARAMETERS
dl.IMAGE_SIZE = 128
dl.BATCH_SIZE = 32
EPOCHS = 2

MODEL_PATH = "models/max128_5_12.keras"

TFRECORD_PATH_TRAIN = "tfrecords/Agent2_train.tfrecord"  # 128x128 train precomputed images and masks
# TFRECORD_PATH_TRAIN = "tfrecords/train_32.tfrecord"  # 32x32 train precomputed images and masks
TFRECORD_PATH_VAL = "tfrecords/Agent2_val.tfrecord"  # 128x128 test precomputed images and masks
# TFRECORD_PATH_VAL = "tfrecords/val_32.tfrecord"  # 32x32 test precomputed images and masks

TRAIN_STEPS, VAL_STEPS = dl.coco_steps()
DATASET_RATIO = 100  # 1/?
TRAIN_STEPS, VAL_STEPS = TRAIN_STEPS//DATASET_RATIO, VAL_STEPS//DATASET_RATIO

## LOAD MODEL FROM AGENT 1
msh_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_combined_loss': weighted_combined_loss,
        'WeightedMeanIoU': WeightedMeanIoU(num_classes=dl.COCO_NUM_CLASSES)
    }
)

## LOAD TRAIN AND VAL DATASETS TODO: ADD RGB IMAGES IF THIS IS NECESSARY
coco_train = dl.coco_RGB_dataset_precomputed_agent2(
    tfrecord_path=TFRECORD_PATH_TRAIN
)#.take(TRAIN_STEPS)

coco_val = dl.coco_RGB_dataset_precomputed_agent2(
    tfrecord_path=TFRECORD_PATH_VAL
)#.take(VAL_STEPS)

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
msh_model.trainable = False  # optional

# Input to the final model
input_gray = layers.Input(shape=(dl.IMAGE_SIZE, dl.IMAGE_SIZE, 1))  # (?, 128, 128, 1)

# Internal call to Agent 1
agent1_mask = msh_model(input_gray)  # (?, 128, 128, 9)
agent1_mask = tf.argmax(agent1_mask, axis=3)[..., tf.newaxis]  # (?, 128, 128, 1)
d3_gray_image = tf.concat([input_gray for _ in range(3)], axis=3)
coco_rgb_colors = tf.constant([
    [255, 255, 255],  # light
    [0, 0, 0],        # dark
    [255, 0, 0],      # red
    [0, 255, 0],      # green
    [0, 0, 255],      # blue
    [0, 255, 255],    # cyan
    [255, 255, 0],    # yellow
    [255, 0, 255],    # magenta
    [128, 128, 128],  # gray
])
agent1_mask = tf.squeeze(agent1_mask, axis=3)
agent1_mask = tf.gather(coco_rgb_colors, agent1_mask)
agent1_mask = d3_gray_image * (tf.cast(agent1_mask, tf.float32) / 255)

# Concatenate input and Agent 1 output
x = layers.Concatenate(axis=-1)([input_gray, agent1_mask])  # (?, 128, 128, 4)

x = layers.Conv2D(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2, 2)(x)

x = layers.Conv2D(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(4, 4)(x)

x = layers.Dropout(0.3)(x)

x = layers.UpSampling2D(4)(x)
x = layers.Conv2DTranspose(dl.IMAGE_SIZE, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.UpSampling2D(2)(x)
x = layers.Conv2DTranspose(dl.IMAGE_SIZE // 2, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(dl.IMAGE_SIZE // 4, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(dl.IMAGE_SIZE // 8, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

output = layers.Conv2D(3, 1, activation='softmax')(x)

model = tf.keras.Model(inputs=input_gray, outputs=output)
"""model = tf.keras.models.Sequential(
    [
        layers.Input(shape=(dl.IMAGE_SIZE, dl.IMAGE_SIZE, 1)),  # RECEIVE (?, 128, 128, 1), values from 0 to 1 or 0 to 255
        # TODO: ADD MASK FROM AGENT 1

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
"""
model.summary()

# TODO: VALIDATE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae', 'accuracy']
)

## TRAIN MODEL
# """
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH.split(".")[0] + "__{val_loss:.2f}.keras",
        monitor='val_loss', save_best_only=True
    )
]
history = model.fit(
    coco_train,
    epochs=EPOCHS,
    steps_per_epoch=TRAIN_STEPS,
    validation_data=coco_val,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)
# """
# model = tf.keras.models.load_model(MODEL_PATH.split(".")[0] + "_0.09.keras")

## DISPLAY HISTORY
# """
def display_history(history, names, title):
    plt.figure()
    plt.title(title)
    plt.plot(history.history[names[0]], label=names[0])
    plt.plot(history.history[names[1]], label=names[1])
    plt.xlabel("Epochs")
    plt.ylabel(names[0])
    plt.legend()
    plt.show()


display_history(history, ['loss', 'val_loss'], "Loss")
display_history(history, ['accuracy', 'val_accuracy'], "Accuracy")
display_history(history, ['mae', 'val_mae'], "MeanAbsoluteError")
# """


## Check model performance
# """
count = 0
images = []
# images = os.listdir("datasets/test2017")
images.insert(0, "charley.jpg")
images.insert(0, "untitl34ed.png")
images.insert(0, "NewYearSkelet.jpg")
images.insert(0, "kode-lgx-dr-ghostx.jpg")
images.insert(0, "lemon.jpg")
images.insert(0, "melon.jpg")
images.insert(0, "plane.jpg")
images.insert(0, "apple.jpg")
for file_name in images:
    try:
        # file_name = "datasets\\kode-lgx-dr-ghostx.jpg"

        # for images, mask in coco_train.take(1):
        #     print("train images batch shape:", images.shape)
        #     print("train masks  batch shape:", mask.shape)

        image_ = cv2.imread("datasets\\test2017\\" + file_name, cv2.IMREAD_COLOR_RGB)
        image_ = cv2.resize(image_, (dl.IMAGE_SIZE, dl.IMAGE_SIZE))

        image = cv2.imread("datasets\\test2017\\" + file_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (dl.IMAGE_SIZE, dl.IMAGE_SIZE))
    except Exception as e:
        # print(e)
        image_ = cv2.imread("datasets\\" + file_name, cv2.IMREAD_COLOR_RGB)
        image_ = cv2.resize(image_, (dl.IMAGE_SIZE, dl.IMAGE_SIZE))

        image = cv2.imread("datasets\\" + file_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (dl.IMAGE_SIZE, dl.IMAGE_SIZE))
    plt.subplot(1, 2, 1)
    plt.title("Our RGB")
    plt.imshow(image_)
    plt.axis("off")

    image = np.expand_dims(image, axis=0)/255

    rgb_pred = model.predict(image)

    plt.subplot(1, 2, 2)
    plt.title("Predicted RGB")
    plt.imshow(rgb_pred[0])
    plt.axis("off")
    plt.show()

    count += 1
    if count == 24:
        break
# """
"""
file_name = "datasets\\kode-lgx-dr-ghostx.jpg"

image_ = cv2.imread(file_name, cv2.IMREAD_COLOR_RGB)
image_ = cv2.resize(image_, (dl.IMAGE_SIZE, dl.IMAGE_SIZE))

image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (dl.IMAGE_SIZE, dl.IMAGE_SIZE))
image = np.expand_dims(image, axis=0)/255

print(image.shape)

rgb_pred = model.predict(image)
print(rgb_pred.shape)
"""
