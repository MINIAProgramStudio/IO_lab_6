## IMPORTS
import dataset_loader as dl
import datasets_from_loader_utils as dflu
import some_functions as sf
import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
layers = tf.keras.layers

import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


## PARAMETERS
dl.IMAGE_SIZE = 128
dl.BATCH_SIZE = 64
EPOCHS = 20

AGENT1_MODELS = "models/Agent1/"
# AGENT1_MODEL_NAME = "max128_5_12.keras"
AGENT1_MODEL_NAME = "unet32hsv_500.keras"
custom_objects = {
    'weighted_combined_loss': sf.weighted_combined_loss,
    # 'weighted_sparse_categorical_crossentropy': sf.weighted_sparse_categorical_crossentropy,
    'WeightedMeanIoU': sf.WeightedMeanIoU(num_classes=dl.COCO_NUM_CLASSES),
    "weighted_sparse_categorical_crossentropy": sf.weighted_sparse_categorical_crossentropy
}
"""
print("precomputing train")
dl.precompute_images(
    img_dir=dl.coco_train_img_dir,
    output_tfrecord_path='FAILSAFE.tfrecord'
)
print("precomputing val")
dl.precompute_images(
    img_dir=dl.coco_val_img_dir,
    output_tfrecord_path='FAILSAFE.tfrecord'
)"""
print("precomputed")
AGENT2_MODELS = "models/Agent2/"
count = len(os.listdir(AGENT2_MODELS))

AGENT1_PATH = AGENT1_MODELS + AGENT1_MODEL_NAME
try:
    Agent1 = model = tf.keras.models.load_model(
        AGENT1_PATH,
        custom_objects=custom_objects
    )
except:
    tf.keras.backend.clear_session()
    print("W I am agent_paint.py Failed to load Agent1")

MODEL_SAVE_PATH = AGENT2_MODELS + "test{count}".format(count=count) + "_e{epoch:02d}_l{val_loss:.4f}.keras"
MODEL_LOAD_PATH = AGENT2_MODELS + "test1_e13_l0.1055.keras"

TFRECORD_PATH_TRAIN = "tfrecords/Agent2_train_hsv.tfrecord"  # 128x128 train precomputed images and masks
# TFRECORD_PATH_TRAIN = "tfrecords/train_32.tfrecord"  # 32x32 train precomputed images and masks
TFRECORD_PATH_VAL = "tfrecords/Agent2_val_hsv.tfrecord"  # 128x128 test precomputed images and masks
# TFRECORD_PATH_VAL = "tfrecords/val_32.tfrecord"  # 32x32 test precomputed images and masks

# TRAIN_STEPS, VAL_STEPS = dl.coco_steps()
# DATASET_RATIO = 400  # 1/?
TRAIN_STEPS, VAL_STEPS = 50, 10

## LOAD MODEL FROM AGENT 1
"""
msh_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects
)
"""

## LOAD TRAIN AND VAL DATASETS TODO: ADD RGB IMAGES IF THIS IS NECESSARY
coco_train = dl.coco_RGB_dataset_precomputed_agent2(
    tfrecord_path=TFRECORD_PATH_TRAIN
)#.take(TRAIN_STEPS)

coco_val = dl.coco_RGB_dataset_precomputed_agent2(
    tfrecord_path=TFRECORD_PATH_VAL
)#.take(VAL_STEPS)

## CREATE MODEL FOR AGENT 2
"""
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
# """

# TODO: RECEIVE GRAYSCALE IMAGE, PREDICT MASK USING AGENT 1 MODEL, MAKE RGB IMAGE USING AGENT 2 MODEL TAKING GRAYSCALE IMAGE AND MASK AS BASE, BUT GENERETE RGB IMAGE BY ITSELF
"""
msh_model.trainable = False  # optional

# Input to the final model
input_gray = layers.Input(shape=(dl.IMAGE_SIZE, dl.IMAGE_SIZE, 1))  # (?, 128, 128, 1)

# Internal call to Agent 1
segmentation_output = msh_model(input_gray)  # (?, 128, 128, 9)
main_segmentation_output = tf.argmax(segmentation_output, axis=3)  # (?, 128, 128)
d3_gray_image = tf.concat([input_gray for _ in range(3)], axis=3)  # (?, 128, 128, 3)

# agent1_mask = tf.squeeze(agent1_mask, axis=3)  # (?, 128, 128)
agent1_rgb_mask = tf.gather(coco_rgb_colors, main_segmentation_output)  # (?, 128, 128, 3)
agent1_rgb_mask = (tf.cast(agent1_rgb_mask, tf.float32) / 255)  # (?, 128, 128, 3)
agent1_image_with_mask = d3_gray_image * agent1_rgb_mask  # (?, 128, 128, 3)
agent1_image_with_mask = tf.image.rgb_to_hsv(agent1_image_with_mask)

# Concatenate input and Agent 1 output
x = layers.Concatenate()([input_gray, agent1_image_with_mask])  # (None, 128, 128, 4)

x = layers.Conv2D(64, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(128, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(64, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

hs = layers.Conv2D(2, (1, 1), activation='sigmoid')(x)  # Hue and Saturation (None, 128, 128, 2)
output = layers.Concatenate(axis=3)([hs, input_gray])  # (None, 128, 128, 3)
# output = layers.Lambda(lambda x: tf.image.hsv_to_rgb(x))(hsv)  # (None, 128, 128, 3)
model = tf.keras.Model(inputs=input_gray, outputs=output)
# """

# """
kernel_size = (5, 5)
input_gray = layers.Input(shape=(dl.IMAGE_SIZE, dl.IMAGE_SIZE, 1))  # (None, 128, 128, 1)
input_mask = layers.Input(shape=(dl.IMAGE_SIZE, dl.IMAGE_SIZE), dtype=tf.int32)  # (None, 128, 128)

rgb_mask = tf.gather(dflu.coco_rgb_colors_tf, input_mask)  # (?, 128, 128, 3)
rgb_mask = (tf.cast(rgb_mask, tf.float32) / 255)  # (?, 128, 128, 3)
d3_gray = tf.concat([input_gray for _ in range(3)], axis=3)  # (?, 128, 128, 3)
image_with_mask = d3_gray * rgb_mask  # (?, 128, 128, 3)
image_with_mask = tf.image.rgb_to_hsv(image_with_mask)

x = layers.Concatenate(axis=-1)([input_gray, image_with_mask])  # (None, 128, 128, 4)

x = layers.Conv2D(64, kernel_size, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(64, kernel_size, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(128, kernel_size, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(64, kernel_size, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(64, kernel_size, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

hs = layers.Conv2D(2, (1, 1), activation='sigmoid')(x)  # Hue and Saturation (None, 128, 128, 2)
output = layers.Concatenate(axis=3)([hs, input_gray])  # (None, 128, 128, 3)
model = tf.keras.Model(inputs=[input_gray, input_mask], outputs=output)
# """


"""
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
# x = layers.Conv2D(dl.IMAGE_SIZE // 2, (5, 5), activation='relu', padding='same')(x)
# x = layers.BatchNormalization()(x)

# x = layers.Conv2D(dl.IMAGE_SIZE // 4, (5, 5), activation='relu', padding='same')(x)
# x = layers.BatchNormalization()(x)

# x = layers.Conv2D(dl.IMAGE_SIZE // 8, (5, 5), activation='relu', padding='same')(x)
# x = layers.BatchNormalization()(x)

output = layers.Conv2D(3, 1, activation='softmax')(x)

model = tf.keras.Model(inputs=input_gray, outputs=output)
"""

"""
model = tf.keras.models.Sequential(
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


# TODO: VALIDATE MODEL
# """
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    # loss='mse',
    loss=sf.combined_loss_agent2,
    # metrics=[
    #     'mae',
    #     ImageQuality,
    #     PerceptualSimilarity,
    # ]
    # loss=tf.keras.losses.MeanSquaredError(),
    # metrics=['mae', 'accuracy']
)

model.summary()
# """

## TRAIN MODEL
# """
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_freq='epoch'
    )
]
"""
history = model.fit(
    coco_train,
    epochs=EPOCHS,
    steps_per_epoch=TRAIN_STEPS,
    validation_data=coco_val,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)
 """

model = tf.keras.models.load_model(
    MODEL_LOAD_PATH,
    custom_objects={"combined_loss_agent2": sf.combined_loss_agent2}
)


## DISPLAY HISTORY

def display_history(history, names, title):
    plt.figure()
    plt.title(title)
    plt.plot(history.history[names[0]], label=names[0])
    plt.plot(history.history[names[1]], label=names[1])
    plt.xlabel("Epochs")
    plt.ylabel(names[0])
    plt.legend()
    plt.show()


#display_history(history, ['loss', 'val_loss'], "Loss")
# display_history(history, ['accuracy', 'val_accuracy'], "Accuracy")
#display_history(history, ['mae', 'val_mae'], "MeanAbsoluteError")
#display_history(history, ['ImageQuality', 'val_ImageQuality'], "ImageQuality")



## Check model performance
# """
count = 0
images = []
# images = os.listdir("datasets/test2017")
images.insert(0, "untitl34ed.png")
images.insert(0, "NewYearSkelet.jpg")
images.insert(0, "kode-lgx-dr-ghostx.jpg")
images.insert(0, "lemon.jpg")
images.insert(0, "melon.jpg")
for file_name in images:
    try:
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
    mask = None
    #try:
    mask = Agent1.predict(image)
    mask = np.argmax(mask, axis = 3)
    """except:
        print("W I am agent_paint.py Failed to use Agent1 model to predict labels for examples")
        mask = dl.rgb_to_label_map(image_)
        mask = np.expand_dims(mask, axis=0)"""
    print(mask.shape)
    rgb_pred = model.predict([image, mask])
    rgb_pred = tf.image.hsv_to_rgb(rgb_pred)

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
