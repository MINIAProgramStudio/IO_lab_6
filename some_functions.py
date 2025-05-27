from logging import WARNING

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tqdm
from numpy.core.memmap import dtypedescr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy
import keras

from pycocotools.coco import COCO
from tensorflow.python.ops.gen_experimental_dataset_ops import data_service_dataset

import dataset_loader

coco_weights = [1, 1e-4, 1, 1, 1, 1, 0.2, 1, 1]
class WeightedMeanIoU(tf.keras.metrics.Metric):

    def __init__(self, num_classes=9, class_weights=coco_weights, name="weighted_mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_weights = tf.constant(class_weights if class_weights is not None else [1.0] * num_classes, dtype=tf.float32)
        self.confusion_matrix = self.add_weight(
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32,
            name="confusion_matrix"
        )

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to predicted class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        self.confusion_matrix.assign_add(cm)

    @tf.function
    def result(self):
        sum_over_row = tf.reduce_sum(self.confusion_matrix, axis=0)
        sum_over_col = tf.reduce_sum(self.confusion_matrix, axis=1)
        true_positives = tf.linalg.tensor_diag_part(self.confusion_matrix)

        denominator = sum_over_row + sum_over_col - true_positives
        iou = tf.math.divide_no_nan(true_positives, denominator)
        weighted_iou = iou * self.class_weights
        return tf.reduce_sum(weighted_iou) / tf.reduce_sum(self.class_weights)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


@tf.function
def dice_loss(y_true, y_pred, smooth=1e-6):
    # y_true: (batch, h, w)     — int32 labels in [0, num_classes)
    # y_pred: (batch, h, w, c)  — float32 softmax probabilities

    num_classes = tf.shape(y_pred)[-1]
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)  # (b,h,w,c)

    # Flatten
    y_true_f = tf.reshape(y_true_onehot, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)  # mean over all classes

@tf.function
def combined_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    d = dice_loss(y_true, y_pred)
    return ce + d




@tf.function
def weighted_dice_loss(y_true, y_pred, class_weights = coco_weights, smooth=1e-6):
    """
    Weighted Dice loss for multi-class segmentation.
    Args:
        y_true: Tensor of shape (batch, h, w), integer labels.
        y_pred: Tensor of shape (batch, h, w, c), softmax probabilities.
        class_weights: Tensor or list of shape (num_classes,) with class weights.
    Returns:
        Weighted dice loss.
    """
    num_classes = tf.shape(y_pred)[-1]
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)  # (b,h,w,c)

    # Flatten
    y_true_f = tf.reshape(y_true_onehot, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f + y_pred_f, axis=0)


    dice = (2. * intersection + smooth) / (union + smooth)

    # Apply class weights
    class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    weighted_dice = dice * class_weights

    return 1 - tf.reduce_sum(weighted_dice) / tf.reduce_sum(class_weights)

@tf.function
def weighted_combined_loss(y_true, y_pred, class_weights = coco_weights):
    """
    Combined weighted cross-entropy and weighted dice loss.
    Args:
        y_true: Ground truth labels (batch, h, w).
        y_pred: Predicted probabilities (batch, h, w, c).
        class_weights: List or tensor of class weights.
    """
    # Weighted categorical cross-entropy
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce_weighted = tf.reduce_mean(tf.gather(class_weights, tf.cast(y_true, tf.int32)) * ce)

    # Weighted dice
    d = weighted_dice_loss(y_true, y_pred, class_weights)

    return ce_weighted + d


tf_weights = tf.constant(coco_weights, dtype=tf.float32)  # Example weights for 9 classes
@tf.function
def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    loss_fn = SparseCategoricalCrossentropy(from_logits=False)
    pixel_weights = tf.gather(tf_weights, y_true)
    unweighted_loss = loss_fn(y_true, y_pred)
    return tf.reduce_mean(unweighted_loss * pixel_weights)


class SegmentationMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, name="SegmentationMeanIoU", *, num_classes, image_size=dataset_loader.IMAGE_SIZE, **kwargs):
        # Filter out image_size from kwargs to avoid passing it to MeanIoU
        kwargs.pop('image_size', None)
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self.num_classes = num_classes
        self.image_size = image_size

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.ensure_shape(y_true, [None, self.image_size, self.image_size])
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_labels = tf.ensure_shape(y_pred_labels, [None, self.image_size, self.image_size])
        return super().update_state(y_true, y_pred_labels, sample_weight)

    @tf.function
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "image_size": self.image_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            name=config.get("name", "SegmentationMeanIoU"),
            num_classes=config["num_classes"],
            image_size=config.get("image_size", 128)
        )


def ImageQuality(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def PerceptualSimilarity(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def combined_loss_agent2(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = 1 - tf.reduce_mean(PerceptualSimilarity(y_true, y_pred))
    psnr = tf.reduce_mean(1/(ImageQuality(y_true, y_pred) + 1e-10))
    return 1 * mse + 1 * ssim + 1 * psnr

def combined_loss_agent2_v2(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true[0] - y_pred[0]))*0.6
    mse += (tf.reduce_mean(y_true[1]-y_pred[1]) * tf.reduce_mean(y_true[1] + 0.1 - y_pred[1]))*0.4
    ssim = 1 - tf.reduce_mean(PerceptualSimilarity(y_true, y_pred))
    psnr = tf.reduce_mean(1 / (ImageQuality(y_true, y_pred) + 1e-10))
    return 0.5 * mse + 0.2 * ssim + 0.3 * psnr