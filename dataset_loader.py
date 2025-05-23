import tensorflow as tf
import os
from pycocotools.coco import COCO

import numpy as np

import datasets_from_loader_utils as dflu

coco_base_dir = "datasets"
coco_train_img_dir = os.path.join(coco_base_dir, "train2017")
coco_val_img_dir = os.path.join(coco_base_dir, "val2017")
coco_train_ann_file = os.path.join(coco_base_dir, "stuff_annotations_trainval2017/annotations", "stuff_train2017.json")
coco_val_ann_file = os.path.join(coco_base_dir, "stuff_annotations_trainval2017/annotations", "stuff_val2017.json")
IMAGE_SIZE = 128
BATCH_SIZE = 128
COCO_NUM_CLASSES = 5

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

    def check_labels(img, mask):
        # mask should be int32 with values in [0, COCO_NUM_CLASSES-1].
        tf.debugging.assert_greater_equal(mask, 0, message="mask < 0")
        tf.debugging.assert_less(mask, COCO_NUM_CLASSES,
                                 message=f"mask ≥ {COCO_NUM_CLASSES}")
        return img, mask
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).map(check_labels, num_parallel_calls=tf.data.AUTOTUNE)
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

def coco_simple_segmentation_dataset(split='train', channels=3):
    if split == 'train':
        coco = COCO(coco_train_ann_file)
        img_dir = coco_train_img_dir
    else:
        coco = COCO(coco_val_ann_file)
        img_dir = coco_val_img_dir

    img_ids = coco.getImgIds()
    img_files = coco.loadImgs(img_ids)

    def generator():
        for img_data in img_files:
            yield load_example(img_data, img_dir, coco)

    def preprocess(img_path, boxes, labels, masks):
        # 1) Load & prep image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=channels)  # Assume RGB
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float16) / 255.0
        labels = dflu.coco_labels_index_merge(labels)
        if tf.shape(masks)[0] == 0:
            mask = tf.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=tf.int8)
            return img, mask

        resized_masks = tf.image.resize(
            tf.cast(masks[..., tf.newaxis], tf.float32),
            (IMAGE_SIZE, IMAGE_SIZE),
            method='nearest'
        )
        resized_masks = tf.cast(resized_masks[..., 0] > 0.5, tf.int8)

        expanded_labels = tf.expand_dims(tf.expand_dims(labels, axis=1), axis=2)
        expanded_labels = tf.cast(expanded_labels, tf.int8)  # Shape: (num_instances, 1, 1)
        class_masks = resized_masks * expanded_labels  # Broadcasting to (num_instances, IMAGE_SIZE, IMAGE_SIZE)
        canvas = tf.reduce_max(class_masks, axis=0)  # Shape: (IMAGE_SIZE, IMAGE_SIZE)

        return img, canvas

    output_types = (tf.string, tf.float32, tf.int64, tf.uint8)
    output_shapes = ((), (None, 4), (None,), (None, None, None))
    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(8).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()

def coco_cardinality():
    coco_train = COCO(coco_train_ann_file)
    num_train = len(coco_train.getImgIds())
    train_steps = num_train // BATCH_SIZE

    coco_val = COCO(coco_val_ann_file)
    num_val = len(coco_val.getImgIds())
    val_steps = num_val // BATCH_SIZE
    return train_steps, val_steps
