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

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

from pycocotools.coco import COCO

import numpy as np

coco_base_dir = "datasets"
coco_train_img_dir = os.path.join(coco_base_dir, "train2017")
coco_val_img_dir = os.path.join(coco_base_dir, "val2017")
coco_train_ann_file = os.path.join(coco_base_dir, "annotations_trainval2017/annotations", "instances_train2017.json")
coco_val_ann_file = os.path.join(coco_base_dir, "annotations_trainval2017/annotations", "instances_val2017.json")
IMAGE_SIZE = 128

def coco_load_train(channels=3):
    coco = COCO(coco_train_ann_file)
    img_ids = coco.getImgIds()
    img_files = coco.loadImgs(img_ids)

    def load_example(img_data):
        img_path = os.path.join(coco_train_img_dir, img_data['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img_data['id'], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bbox = [y / img_data['height'], x / img_data['width'],
                    (y + h) / img_data['height'], (x + w) / img_data['width']]
            boxes.append(bbox)
            labels.append(ann['category_id'])

        return img_path, boxes, labels

    def generator():
        for img_data in img_files:
            yield load_example(img_data)

    # TF dataset
    output_types = (tf.string, tf.float32, tf.int64)
    output_shapes = ((), (None, 4), (None,))
    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes)

    def preprocess(img_path, boxes, labels):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, {'boxes': boxes, 'labels': labels}

    BATCH_SIZE = 8
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def coco_load_val(channels=3):
    coco = COCO(coco_val_ann_file)
    img_ids = coco.getImgIds()
    img_files = coco.loadImgs(img_ids)

    def load_example(img_data):
        img_path = os.path.join(coco_val_img_dir, img_data['file_name'])

        ann_ids = coco.getAnnIds(imgIds=img_data['id'], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bbox = [y / img_data['height'], x / img_data['width'],
                    (y + h) / img_data['height'], (x + w) / img_data['width']]
            boxes.append(bbox)
            labels.append(ann['category_id'])

        return img_path, boxes, labels

    def generator():
        for img_data in img_files:
            yield load_example(img_data)

    # TF dataset
    output_types = (tf.string, tf.float32, tf.int64)
    output_shapes = ((), (None, 4), (None,))
    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes)

    def preprocess(img_path, boxes, labels):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, {'boxes': boxes, 'labels': labels}

    BATCH_SIZE = 8
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds