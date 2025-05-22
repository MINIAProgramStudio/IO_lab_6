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
import datasets_from_loader_utils as dflu

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Loading MS COCO dataset.")
coco_train_and_test = dataset_loader.coco_load_train(channels=1)
coco_val = dataset_loader.coco_load_val(channels=1)
print("MS COCO loaded. Splitting it into train and test.")

coco_test, coco_train = dflu.splt_test_and_train(coco_train_and_test)
del coco_train_and_test
print("COCO split completed.")

print("Some COCO labels:")
dflu.first_batch_labels(coco_test, dflu.coco_labels)