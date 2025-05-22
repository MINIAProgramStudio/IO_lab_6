import tensorflow as tf
import os
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
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [IMAGE_SIZE, IMAGE_SIZE, 1],  # Image shape
            {'boxes': [None, 4], 'labels': [None]}  # Variable-length boxes and labels
        ),
        padding_values=(
            0.0,  # Image padding value
            {'boxes': tf.constant(-1.0, dtype=tf.float32),
             'labels': tf.constant(-1, dtype=tf.int64)}  # Padding values for boxes and labels
        )
    )
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
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [IMAGE_SIZE, IMAGE_SIZE, 1],  # Image shape
            {'boxes': [None, 4], 'labels': [None]}  # Variable-length boxes and labels
        ),
        padding_values=(
            0.0,  # Image padding value
            {'boxes': tf.constant(-1.0, dtype=tf.float32),
             'labels': tf.constant(-1, dtype=tf.int64)}  # Padding values for boxes and labels
        )
    )
    return ds