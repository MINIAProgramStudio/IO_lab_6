import tensorflow as tf
import os
from pycocotools.coco import COCO

import numpy as np

import datasets_from_loader_utils as dflu

from dataset_loader import *


def calculate_label_distribution(tfrecord_path, channels=1):
    # Load dataset without repeat for one full pass
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(lambda x: parse_tfrecord_image_and_mask(x, channels=channels))
    ds = ds.batch(BATCH_SIZE)

    # Initialize counts array for each class label (0 to 8)
    label_counts = np.zeros(COCO_NUM_CLASSES, dtype=np.int64)

    for _, label_batch in ds:
        # label_batch shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE]
        # Flatten batch labels to 1D and count occurrences of each label in batch
        labels_np = label_batch.numpy().reshape(-1)
        counts_batch = np.bincount(labels_np, minlength=COCO_NUM_CLASSES)
        label_counts += counts_batch

    # Optionally normalize counts to get proportions
    total_pixels = np.sum(label_counts)
    label_distribution = label_counts / total_pixels

    return label_counts, label_distribution

tfrecord_path = 'tfrecords/128hsv1_train.tfrecord'  # your precomputed TFRecord path
counts, distribution = calculate_label_distribution(tfrecord_path)

print("Pixel counts per label:")
for c in range(len(counts)):
    print(c, counts[c])
print("Normalized label distribution:", distribution)