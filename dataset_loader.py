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


def read_reshape_normalise(img_path: str) -> tf.Tensor:
    """_summary_

    Args:
        img_path (str): Path to image

    Returns:
        tf.Tensor: Image tensor
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def write_tfrecord_for_images_and_masks(image_dir, output_tfrecord_path, channels=3):
    img_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        for img_file in img_files:
            img_path = os.path.join(image_dir, img_file)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = tf.cast(img, tf.float32) / 255.0
            label_map = rgb_to_hsv_to_label_map(img)
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


def write_tfrecord_for_images(image_dir, output_tfrecord_path):
    img_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        for img_file in img_files:
            img_path = os.path.join(image_dir, img_file)
            img_rgb = read_reshape_normalise(img_path)
            image_mask = rgb_to_hsv_to_label_map(img_rgb)
            img_gray = tf.image.rgb_to_grayscale(img_rgb)
            img_rgb = tf.image.rgb_to_hsv(img_rgb)

            img_gray = tf.ensure_shape(img_gray, [IMAGE_SIZE, IMAGE_SIZE, 1])
            image_mask = tf.ensure_shape(image_mask, [IMAGE_SIZE, IMAGE_SIZE])
            img_rgb = tf.ensure_shape(img_rgb, [IMAGE_SIZE, IMAGE_SIZE, 3])

            feature = {
                'img_path': _bytes_feature(img_path.encode('utf-8')),
                'image_gray': _bytes_feature(tf.io.serialize_tensor(img_gray).numpy()),
                'image_mask': _bytes_feature(tf.io.serialize_tensor(image_mask).numpy()),
                'image_rgb': _bytes_feature(tf.io.serialize_tensor(img_rgb).numpy())
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


def parse_tfrecord_images(serialized_example):
    feature_description = {
        'img_path': tf.io.FixedLenFeature([], tf.string),
        'image_gray': tf.io.FixedLenFeature([], tf.string),
        'image_mask': tf.io.FixedLenFeature([], tf.string),
        'image_rgb': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image_gray = tf.io.parse_tensor(example['image_gray'], out_type=tf.float32)
    image_mask = tf.io.parse_tensor(example['image_mask'], out_type=tf.int32)
    image_rgb = tf.io.parse_tensor(example['image_rgb'], out_type=tf.float32)

    image_gray.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    image_mask.set_shape([IMAGE_SIZE, IMAGE_SIZE])
    image_rgb.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    return (image_gray, image_mask), image_rgb


# def parse_tfrecord_rgb_mask(serialized_example):
#     """
#     Parse a single TFRecord example into img_path and label_map.

#     Returns:
#         img_path, label_map
#     """
#     feature_description = {
#         'img_path': tf.io.FixedLenFeature([], tf.string),
#         'label_map': tf.io.FixedLenFeature([], tf.string)
#     }

#     example = tf.io.parse_single_example(serialized_example, feature_description)
#     img_path = example['img_path']
#     label_map = tf.io.parse_tensor(example['label_map'], out_type=tf.int32)

#     return img_path, label_map

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


def precompute_images(
    img_dir: str,
    output_tfrecord_path: str,
) -> None:
    """Precompute resized/grayscaled images and resized/RGB images for the dataset, save to TFRecord.

    Args:
        img_dir (str): Directory with images
        output_tfrecord_path (str): Path to save the TFRecord file
    """
    write_tfrecord_for_images(img_dir, output_tfrecord_path)


# def precompute_rgb_mask_dataset(split='train', channels=3, tfrecord_path=None):
#     """
#     Create a TensorFlow dataset from precomputed TFRecords containing RGB-based label maps.

#     Args:
#         split: 'train' or 'val' to select dataset split.
#         channels: Number of image channels (1 for grayscale, 3 for RGB).
#         tfrecord_path: Path to the precomputed TFRecord file.
#         batch_size: Batch size for the dataset.
#         image_size: Target image size for resizing.

#     Returns:
#         A tf.data.Dataset yielding (img, label_map) pairs.
#     """
#     if tfrecord_path is None:
#         tfrecord_path = 'rgb_train.tfrecord' if split == 'train' else 'rgb_val.tfrecord'

#     def preprocess(img_path, label_map):
#         # Load and prep image
#         img = tf.io.read_file(img_path)
#         img = tf.image.decode_jpeg(img, channels=3)  # RGB image
#         img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#         img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]

#         if channels == 1:
#             img = tf.image.rgb_to_grayscale(img)

#         return img, label_map

#     # Load TFRecord dataset
#     ds = tf.data.TFRecordDataset(tfrecord_path)
#     ds = ds.map(parse_tfrecord_rgb_mask, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#     return ds.repeat()


# """
def rgb_to_label_map(img):
    r, g, b = tf.cast(img[..., 0], dtype=tf.float32), tf.cast(img[..., 1], dtype=tf.float32), tf.cast(img[..., 2], dtype=tf.float32)
    #mean_rgb = tf.reduce_mean(tf.cast(img, dtype=tf.float32), axis=-1)
    max_rgb = np.max(r.numpy()+g.numpy()+b.numpy())
    if max_rgb <= 4:
        max_rgb = 2.8
        min_rgb = 0.004*40
    else:
        min_rgb = 40
        max_rgb = 255*2.8
    # Define all conditions in one go
    conditions = [
        r+b+g > max_rgb,  # light
        r+b+g < min_rgb,  # dark
        r*1.2 > b+g,  # red
        g*1.7 > r+b,  # green
        b*1.6 > r+g,  # blue
        r*1.8 < b+g,  # cyan
        b*1.75 < r+g,  # yellow
        g*1.8 < r+b  # magenta
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]

    label_map = tf.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=tf.int32) * 8
    for cond, label in zip(conditions[::-1], labels[::-1]):  # Reverse for precedence
        label_map = tf.where(cond, label, label_map)
    return label_map
# """


def rgb_to_hsv_to_label_map(img):
    hsv = tf.image.rgb_to_hsv(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    conditions = [
        tf.logical_and(s < 0.1, v > 0.9),  # light
        v < 0.3,  # dark
        tf.logical_or(h < 1/12, h > 11/12),  # red
        tf.math.abs(h-1/3) < 1/12,  # green
        tf.math.abs(h-2/3) < 1/12,  # blue
        tf.math.abs(h-1/2) < 1/12,  # cyan
        tf.math.abs(h-1/8) < 1/12,  # yellow
        tf.math.abs(h-5/6) < 1/12  # magenta
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    label_map = tf.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=tf.int32) * 8
    for cond, label in zip(conditions[::-1], labels[::-1]):  # Reverse for precedence
        label_map = tf.where(cond, label, label_map)
    return label_map

def coco_RGB_dataset_precomputed(split='train', channels=3, tfrecord_path=None):
    if tfrecord_path is None:
        tfrecord_path = 'image_mask_train.tfrecord' if split == 'train' else 'image_mask_val.tfrecord'
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(lambda x: parse_tfrecord_image_and_mask(x, channels=channels),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


def coco_RGB_dataset_precomputed_agent2(tfrecord_path: str) -> tf.data.TFRecordDataset:
    """From precomputed TFRecords creates a TF dataset

    Args:
        tfrecord_path (str): Path where tfrecord is located

    Returns:
        tf.data.TFRecordDataset: TFRecordDataset
    """
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(
        lambda x: parse_tfrecord_images(x),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


def coco_RGB_datasets_agent2() -> tf.data.TFRecordDataset:
    """Creates a dataset from images

    Args:
        img_dir (str): Path where images is located

    Returns:
        tf.data.TFRecordDataset: TFRecordDataset
    """

    train = tf.keras.preprocessing.image_dataset_from_directory(
        coco_train_img_dir,
        labels=None,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE).repeat()
    val = tf.keras.preprocessing.image_dataset_from_directory(
        coco_val_img_dir,
        labels=None,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE).repeat()
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     img_dir,
    #     image_size=(IMAGE_SIZE, IMAGE_SIZE),
    #     batch_size=BATCH_SIZE
    # )
    # ds = ds.prefetch(tf.data.AUTOTUNE)
    # return ds.repeat()
    return train, val


def coco_cardinality():
    coco_train = COCO(coco_train_ann_file)
    num_train = len(coco_train.getImgIds())
    print("Number of train images (FROM COCO):", num_train)
    print("Number of train images: (EXACT)", len(os.listdir(coco_train_img_dir)))
    print("They Are Equal:", num_train == len(os.listdir(coco_train_img_dir)))
    train_steps = num_train // BATCH_SIZE

    coco_val = COCO(coco_val_ann_file)
    num_val = len(coco_val.getImgIds())
    print("Number of val images (FROM COCO):", num_val)
    print("Number of val images: (EXACT)", len(os.listdir(coco_val_img_dir)))
    print("They Are Equal:", num_val == len(os.listdir(coco_val_img_dir)))
    val_steps = num_val // BATCH_SIZE
    return train_steps, val_steps


def coco_steps() -> list[int, int]:
    """Returns the number of steps for train and val

    Returns:
        list[int, int]: steps for train and val
    """
    num_train = len(os.listdir(coco_train_img_dir))
    train_steps = num_train // BATCH_SIZE
    num_val = len(os.listdir(coco_val_img_dir))
    val_steps = num_val // BATCH_SIZE
    return train_steps, val_steps
