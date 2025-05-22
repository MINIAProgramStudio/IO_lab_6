import tensorflow as tf
import os
from pycocotools.coco import COCO

import numpy as np

coco_base_dir = "datasets"
coco_train_img_dir = os.path.join(coco_base_dir, "train2017")
coco_val_img_dir = os.path.join(coco_base_dir, "val2017")
coco_train_ann_file = os.path.join(coco_base_dir, "stuff_annotations_trainval2017/annotations", "stuff_train2017.json")
coco_val_ann_file = os.path.join(coco_base_dir, "stuff_annotations_trainval2017/annotations", "stuff_val2017.json")
IMAGE_SIZE = 128

def load_example(img_data, image_dir, coco):
    # get original image dimensions
    img_h = img_data['height']
    img_w = img_data['width']

    img_path = os.path.join(image_dir, img_data['file_name'])
    ann_ids  = coco.getAnnIds(imgIds=img_data['id'], iscrowd=False)
    anns     = coco.loadAnns(ann_ids)

    boxes, labels, masks = [], [], []
    for ann in anns:
        # normalize bbox using img_h and img_w
        x, y, w, h = ann['bbox']
        boxes.append([
            y / img_h,
            x / img_w,
            (y + h) / img_h,
            (x + w) / img_w
        ])
        labels.append(ann['category_id'])

        # build mask of shape (img_h, img_w)
        mask = coco.annToMask(ann)
        masks.append(mask)

    # convert to numpy, ensuring the right shapes
    boxes  = np.array(boxes,  dtype=np.float32).reshape(-1, 4)
    labels = np.array(labels, dtype=np.int64).reshape(-1)
    if masks:
        masks = np.stack(masks, axis=0)   # -> (N, img_h, img_w)
    else:
        masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

    return img_path, boxes, labels, masks

def coco_load_train(channels=3):
    coco = COCO(coco_train_ann_file)
    img_ids = coco.getImgIds()
    img_files = coco.loadImgs(img_ids)
    def generator():
        for img_data in img_files:
            yield load_example(img_data, coco_train_img_dir, coco)

    # TF dataset
    output_types = (tf.string, tf.float32, tf.int64, tf.uint8)
    output_shapes = ((), (None, 4), (None,), (None, None, None))
    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes)

    def preprocess(img_path, boxes, labels, masks):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float32) / 255.0

        # Resize masks from (N, H, W) to (N, IMAGE_SIZE, IMAGE_SIZE)
        masks = tf.cast(masks, tf.float32)
        # tf.image.resize expects rank 3 or 4, so add batch dimension if needed
        # masks shape: (N, H, W)
        masks = tf.image.resize(masks[..., tf.newaxis], (IMAGE_SIZE, IMAGE_SIZE), method='nearest')
        masks = tf.squeeze(masks, axis=-1)
        masks = tf.cast(masks, tf.uint8)

        return img, {'boxes': boxes, 'labels': labels, 'masks': masks}

    BATCH_SIZE = 8
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [IMAGE_SIZE, IMAGE_SIZE, channels],
            {
                'boxes': [None, 4],
                'labels': [None],
                'masks': [None, IMAGE_SIZE, IMAGE_SIZE]
            }
        ),
        padding_values=(
            0.0,
            {
                'boxes': tf.constant(-1.0, dtype=tf.float32),
                'labels': tf.constant(-1, dtype=tf.int64),
                'masks': tf.constant(0, dtype=tf.uint8)
            }
        )
    )
    return ds


def coco_load_val(channels=3):
    coco = COCO(coco_val_ann_file)
    img_ids = coco.getImgIds()
    img_files = coco.loadImgs(img_ids)

    def generator():
        for img_data in img_files:
            yield load_example(img_data, coco_val_img_dir, coco)

    # TF dataset
    output_types = (tf.string, tf.float32, tf.int64, tf.uint8)
    output_shapes = ((), (None, 4), (None,), (None, None, None))
    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes)

    def preprocess(img_path, boxes, labels, masks):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float32) / 255.0

        # Resize masks from (N, H, W) to (N, IMAGE_SIZE, IMAGE_SIZE)
        masks = tf.cast(masks, tf.float32)
        # tf.image.resize expects rank 3 or 4, so add batch dimension if needed
        # masks shape: (N, H, W)
        masks = tf.image.resize(masks[..., tf.newaxis], (IMAGE_SIZE, IMAGE_SIZE), method='nearest')
        masks = tf.squeeze(masks, axis=-1)
        masks = tf.cast(masks, tf.uint8)

        return img, {'boxes': boxes, 'labels': labels, 'masks': masks}

    BATCH_SIZE = 8
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [IMAGE_SIZE, IMAGE_SIZE, channels],
            {
                'boxes': [None, 4],
                'labels': [None],
                'masks': [None, IMAGE_SIZE, IMAGE_SIZE]
            }
        ),
        padding_values=(
            0.0,
            {
                'boxes': tf.constant(-1.0, dtype=tf.float32),
                'labels': tf.constant(-1, dtype=tf.int64),
                'masks': tf.constant(0, dtype=tf.uint8)
            }
        )
    )
    return ds


