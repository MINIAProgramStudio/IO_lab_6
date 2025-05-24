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


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # Convert tensor to bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord_for_images_and_masks(image_dir, output_tfrecord_path, channels=3):
    img_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        for img_file in img_files:
            img_path = os.path.join(image_dir, img_file)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = tf.cast(img, tf.float32) / 255.0
            label_map = rgb_to_label_map(img)
            if channels == 1:
                img = tf.image.rgb_to_grayscale(img)

            # Ensure shapes
            img = tf.ensure_shape(img, [IMAGE_SIZE, IMAGE_SIZE, channels])
            label_map = tf.ensure_shape(label_map, [IMAGE_SIZE, IMAGE_SIZE])

            feature = {
                'img_path': _bytes_feature(img_path.encode('utf-8')),
                'image': _bytes_feature(tf.io.serialize_tensor(img).numpy()),
                'label_map': _bytes_feature(tf.io.serialize_tensor(label_map).numpy())
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def parse_tfrecord_image_and_mask(serialized_example, channels=3):
    feature_description = {
        'img_path': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label_map': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    img = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    label_map = tf.io.parse_tensor(example['label_map'], out_type=tf.int32)

    # Explicitly set shapes
    img.set_shape([IMAGE_SIZE, IMAGE_SIZE, channels])
    label_map.set_shape([IMAGE_SIZE, IMAGE_SIZE])

    return img, label_map


def parse_tfrecord_rgb_mask(serialized_example):
    """
    Parse a single TFRecord example into img_path and label_map.

    Returns:
        img_path, label_map
    """
    feature_description = {
        'img_path': tf.io.FixedLenFeature([], tf.string),
        'label_map': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    img_path = example['img_path']
    label_map = tf.io.parse_tensor(example['label_map'], out_type=tf.int32)

    return img_path, label_map

def precompute_image_and_mask_dataset(split='train', train_img_dir=None, val_img_dir=None,
                                      output_tfrecord_path=None, channels=3):
    """
    Precompute resized/grayscaled images and RGB-based label maps for the dataset, save to TFRecord.

    Args:
        split: 'train' or 'val' to select dataset split.
        train_img_dir: Directory with training images.
        val_img_dir: Directory with validation images.
        output_tfrecord_path: Path to save the TFRecord file.
        image_size: Target image size for resizing.
        channels: Number of channels (1 for grayscale, 3 for RGB).

    Returns:
        Path to the generated TFRecord file.
    """
    if split == 'train':
        img_dir = train_img_dir
        tfrecord_path = output_tfrecord_path or 'image_mask_train.tfrecord'
    else:
        img_dir = val_img_dir
        tfrecord_path = output_tfrecord_path or 'image_mask_val.tfrecord'

    write_tfrecord_for_images_and_masks(img_dir, tfrecord_path, channels)
    return tfrecord_path


def precompute_rgb_mask_dataset(split='train', channels=3, tfrecord_path=None):
    """
    Create a TensorFlow dataset from precomputed TFRecords containing RGB-based label maps.

    Args:
        split: 'train' or 'val' to select dataset split.
        channels: Number of image channels (1 for grayscale, 3 for RGB).
        tfrecord_path: Path to the precomputed TFRecord file.
        batch_size: Batch size for the dataset.
        image_size: Target image size for resizing.

    Returns:
        A tf.data.Dataset yielding (img, label_map) pairs.
    """
    if tfrecord_path is None:
        tfrecord_path = 'rgb_train.tfrecord' if split == 'train' else 'rgb_val.tfrecord'

    def preprocess(img_path, label_map):
        # Load and prep image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)  # RGB image
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]

        if channels == 1:
            img = tf.image.rgb_to_grayscale(img)

        return img, label_map

    # Load TFRecord dataset
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(parse_tfrecord_rgb_mask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


def rgb_to_label_map(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mean_rgb = tf.reduce_mean(img, axis=-1)

    # Define all conditions in one go
    conditions = [
        mean_rgb > (230 / 255.0),  # light
        mean_rgb < (20 / 255.0),  # dark
        (r * 0.7 - mean_rgb) > 0,  # red
        (g * 0.7 - mean_rgb) > 0,  # green
        (b * 0.75 - mean_rgb) > 0,  # blue
        (r * 1.12 - mean_rgb) < 0,  # cyan
        (b * 1.265 - mean_rgb) < 0,  # yellow
        (g * 1.14 - mean_rgb) < 0  # magenta
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]

    label_map = tf.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=tf.int32) * 8
    for cond, label in zip(conditions[::-1], labels[::-1]):  # Reverse for precedence
        label_map = tf.where(cond, label, label_map)
    return label_map

def coco_RGB_dataset_precomputed(split='train', channels=3, tfrecord_path=None, batch_size=32, image_size=128):
    if tfrecord_path is None:
        tfrecord_path = 'image_mask_train.tfrecord' if split == 'train' else 'image_mask_val.tfrecord'
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(lambda x: parse_tfrecord_image_and_mask(x, channels=channels),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


RGB_train_coco_instance = None
RGB_val_coco_instance = None

def coco_RGB_dataset(split='train', channels=3):
    global RGB_train_coco_instance
    global RGB_val_coco_instance
    if split == 'train':
        if RGB_train_coco_instance is None:
            coco = COCO(coco_train_ann_file)
            img_dir = coco_train_img_dir
            RGB_train_coco_instance = coco
        else:
            coco = RGB_train_coco_instance
    else:
        if RGB_val_coco_instance is None:
            coco = COCO(coco_val_ann_file)
            img_dir = coco_val_img_dir
            RGB_val_coco_instance = coco
        else:
            coco = RGB_val_coco_instance

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

    def write_tfrecord(img_data, img_dir, coco, writer):
        img_path, boxes, labels, masks = load_example(img_data, img_dir, coco)
        # Serialize to TFRecord (example serialization)
        feature = {
            'img_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path.encode()])),
            'boxes': tf.train.Feature(float_list=tf.train.FloatList(value=boxes.flatten())),
            'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
            'masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(masks).numpy()]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    # Write TFRecords
    with tf.io.TFRecordWriter('dataset.tfrecord') as writer:
        for img_data in img_files:
            write_tfrecord(img_data, img_dir, coco, writer)

    # Load dataset
    ds = tf.data.TFRecordDataset('dataset.tfrecord')
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)  # Define parse_tfrecord
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
