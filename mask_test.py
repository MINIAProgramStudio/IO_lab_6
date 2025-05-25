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
MODEL_PATH = 'models/max128_12.keras'
IMAGE_PATH = 'datasets/lemon.jpg'
IMAGE_SIZE = dataset_loader.IMAGE_SIZE

# --- Load model (grayscale input) ---
model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_combined_loss': weighted_combined_loss,
        'WeightedMeanIoU': WeightedMeanIoU(num_classes=dataset_loader.COCO_NUM_CLASSES)
    }
)

# --- Load & preprocess as GRAYSCALE ---
gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
# Resize
gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))

# Normalize
gray_norm = gray.astype('float32') / 255.0

# Add channel and batch dims -> (1, H, W, 1)
inp_gray = gray_norm[..., np.newaxis][np.newaxis, ...]
print(f"Model input shape: {inp_gray.shape}")  # (1, 32, 32, 1)


image = inp_gray  # shape: (1, H, W, channels)
logits = model.predict(image)
print(PTC(logits[0][0].tolist()))
predicted_classes = tf.argmax(logits, axis=3)  # shape: (1, H, W)
predicted_matrix = predicted_classes[0].numpy()
print()
table = PTC(predicted_matrix.tolist())
print(table)


def plot_class_matrix(predicted_matrix, num_classes, class_names=dflu.coco_rgb_labels):
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap('tab20', num_classes)  # discrete colormap with num_classes colors
    im = plt.imshow(predicted_matrix, cmap=cmap, vmin=0, vmax=num_classes - 1)
    plt.colorbar(im, ticks=np.arange(num_classes))

    if class_names:
        # Optional: show colorbar ticks with class names (if not too many)
        im.colorbar.set_ticks(np.arange(num_classes))
        im.colorbar.set_ticklabels(class_names)

    plt.title("Predicted Class Matrix")
    plt.axis('off')
    plt.show()
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB

img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
plt.imshow(img_resized)
plt.show()
plot_class_matrix(predicted_matrix, 9)
plot_class_matrix(dataset_loader.rgb_to_label_map(img_resized), 9)

