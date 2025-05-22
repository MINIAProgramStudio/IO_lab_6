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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Loading MS COCO dataset.")
coco_train = dataset_loader.coco_load_train()
coco_val = dataset_loader.coco_load_val()
print("MS COCO loaded. Splitting it into train and test.")

def is_test(x, y):
    return x % 4 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y

coco_test = coco_train.enumerate() \
                    .filter(is_test) \
                    .map(recover)

coco_train = coco_train.enumerate() \
                    .filter(is_train) \
                    .map(recover)
print("MS COCO splitted and ready for work.")