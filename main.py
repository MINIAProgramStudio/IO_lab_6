from logging import WARNING

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

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
import keras

from pycocotools.coco import COCO

import numpy as np

import dataset_loader
import warnings

from some_functions import *

model = tf.keras.models.load_model('models/Agent1/unet32_100.keras', custom_objects={'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy, "WeightedMeanIoU": WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)})
plot_model(model, show_shapes=True)