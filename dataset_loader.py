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
IMAGE_SIZE = 32
BATCH_SIZE = 512
COCO_NUM_CLASSES = 9


def load_example(img_data, image_dir, coco):
    # get original image dimensions
    img_h = img_data['height']
    img_w = img_data['width']

    img_path = os.path.join(image_dir, img_data['file_name'])
    ann_ids = coco.getAnnIds(imgIds=img_data['id'], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

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
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.array(labels, dtype=np.int64).reshape(-1)
    if masks:
        masks = np.stack(masks, axis=0)  # -> (N, img_h, img_w)
    else:
        masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

    return img_path, boxes, labels, masks


def rgb_to_label_map(img):
    # Extract R, G, B channels
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # Compute mean of all channels
    mean_rgb = tf.reduce_mean(img, axis=-1)  # Shape: (IMAGE_SIZE, IMAGE_SIZE)

    # Define conditions
    light_condition = mean_rgb > (230 / 255.0)  # Mean > 230/255
    dark_condition = mean_rgb < (20 / 255.0)  # Mean < 20/255
    red_condition = (r*0.7 - mean_rgb) > 0
    green_condition = (g*0.7 - mean_rgb) > 0
    blue_condition = (b*0.75 - mean_rgb) > 0
    cyan_condition = (r*1.12 - mean_rgb) < 0
    yellow_condition = (b*1.265 - mean_rgb) < 0
    magenta_condition = (g*1.14 - mean_rgb) < 0

    # Initialize label map with "gray" (index 8)
    label_map = tf.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=tf.int32) * 8

    # Apply conditions in order of precedence
    #low priority
    label_map = tf.where(cyan_condition, 5, label_map)  # cyan
    label_map = tf.where(yellow_condition, 6, label_map)  # yellow
    label_map = tf.where(magenta_condition, 7, label_map)  # magenta
    #medium priority
    label_map = tf.where(red_condition, 2, label_map)  # red
    label_map = tf.where(green_condition, 3, label_map)  # green
    label_map = tf.where(blue_condition, 4, label_map)  # blue
    #high priority
    label_map = tf.where(light_condition, 0, label_map)  # light
    label_map = tf.where(dark_condition, 1, label_map)  # dark

    return label_map


def coco_RGB_dataset(split='train', channels=3):
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
        # Load and prep image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)  # RGB image
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]

        # Generate label map from RGB values
        label_map = rgb_to_label_map(img)

        if channels == 1:
            img = tf.image.rgb_to_grayscale(img)

        # If you still need boxes or other outputs, adjust accordingly
        # For now, return only img and label_map
        return img, label_map

    output_types = (tf.string, tf.float32, tf.int64, tf.uint8)
    output_shapes = ((), (None, 4), (None,), (None, None, None))
    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


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
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


def coco_cardinality():
    coco_train = COCO(coco_train_ann_file)
    num_train = len(coco_train.getImgIds())
    train_steps = num_train // BATCH_SIZE

    coco_val = COCO(coco_val_ann_file)
    num_val = len(coco_val.getImgIds())
    val_steps = num_val // BATCH_SIZE
    return train_steps, val_steps
