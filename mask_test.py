import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
import dataset_loader
import datasets_from_loader_utils as dflu
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from PythonTableConsole import PythonTableConsole as PTC
from some_functions import *

# --- Configuration ---
MODEL_PATH = 'models/Agent1/unet48hsv_e164_l0.6135.keras'
IMAGE_PATH = 'datasets/lemon.jpg'
IMAGE_SIZE = dataset_loader.IMAGE_SIZE

# --- Load model ---
model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy,
        'WeightedMeanIoU': WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)
    }
)

# --- Load & preprocess grayscale image ---
gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
gray_norm = gray.astype('float32') / 255.0
inp_gray = gray_norm[..., np.newaxis][np.newaxis, ...]

# --- Predict ---
logits = model.predict(inp_gray)
print(PTC(logits[0][0].tolist()))
predicted_classes = tf.argmax(logits, axis=3)[0].numpy()
print()
print(PTC(predicted_classes.tolist()))

# --- Prepare RGB image for visualization ---
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

# --- Ground truth label matrix ---
gt_labels = dataset_loader.rgb_to_hsv_to_label_map(img_resized)

# --- Convert label map to RGB mask using coco_rgb_colors ---
def label_to_rgb(mask, color_map):
    """
    Convert a 2D label map (H x W) into a 3D RGB image (H x W x 3) using a color map.
    """
    return color_map[mask]

# --- Create RGB masks ---
gt_rgb_mask = label_to_rgb(gt_labels, dflu.coco_rgb_colors)
pred_rgb_mask = label_to_rgb(predicted_classes, dflu.coco_rgb_colors)

# --- Plot all three side by side ---
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Original RGB image
axes[0].imshow(img_resized)
axes[0].set_title("Original Image (RGB)")
axes[0].axis('off')

# Ground Truth Mask
axes[1].imshow(gt_rgb_mask.astype(np.uint8))
axes[1].set_title("Ground Truth Mask (RGB)")
axes[1].axis('off')

# Predicted Mask
axes[2].imshow(pred_rgb_mask.astype(np.uint8))
axes[2].set_title("Predicted Mask (RGB)")
axes[2].axis('off')

plt.tight_layout()
plt.show()