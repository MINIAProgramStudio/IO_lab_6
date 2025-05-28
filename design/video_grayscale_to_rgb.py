from itertools import repeat
from random import random

import cv2
import numpy as np
from tqdm import tqdm
import frame_paralel

import dataset_loader as dl
import some_functions as sf
import datasets_from_loader_utils as dflu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from multiprocessing import Pool, cpu_count
from functools import partial
import cv2

from .functions import create_image

import tensorflow as tf
layers = tf.keras.layers

AGENT1_NAME = "unet48hsv_e164_l0.6135.keras"
AGENT2_NAME = "test45_e65_l0.0939.keras"
input_path = 'demonstrator_video/The Arrival of a Train at La Ciotat Station - Lumière Brothers - 1896 3 4.mp4'
output_path = 'demonstrator_video/output_video_3.mp4'
custom_objects = {
    'weighted_combined_loss': sf.weighted_combined_loss,
    # 'weighted_sparse_categorical_crossentropy': sf.weighted_sparse_categorical_crossentropy,
    'WeightedMeanIoU': sf.WeightedMeanIoU(num_classes=dl.COCO_NUM_CLASSES),
    "weighted_sparse_categorical_crossentropy": sf.weighted_sparse_categorical_crossentropy,
    'combined_loss_agent2_v2': sf.combined_loss_agent2_v2
}

# def agent1_process(image: Image.Image, model_name: str) -> Image.Image:

#     try:
#         model = tf.keras.models.load_model(
#             f"./models/Agent1/{model_name}",
#             custom_objects=custom_objects
#         )
#     except Exception:
#         try:
#             model = tf.keras.models.load_model(
#                 f"./models/Agent1/{model_name}",
#                 custom_objects=custom_objects
#             )
#         except Exception:
#             return

#     img = image.copy()  # (128, 128)
#     masks = model.predict(img, verbose = 0)  # (128, 128, 9)
#     mask = np.argmax(masks, axis=3)  # (128, 128)

#     image_with_mask = dflu.apply_mask_to_image(img, mask)  # (?, 128, 128, 3)

#     return image_with_mask, mask


# def agent2_process(image: Image.Image, mask: Image.Image, model_name: str) -> Image.Image:
#     model = tf.keras.models.load_model(
#         f"./models/Agent2/{model_name}",
#         custom_objects={
#             'combined_loss_agent2': sf.combined_loss_agent2,
#             'combined_loss_agent2_v2': sf.combined_loss_agent2_v2
#         }
#     )

#     img = image.copy()

#     img_true_hsv = model.predict([img, mask], verbose = 0)  # (128, 128, 1) -> (128, 128, 3)

#     img_true_rgb = tf.image.hsv_to_rgb(img_true_hsv).numpy()

#     return img_true_rgb


# def mother_agent(image, model1, model2, train=None, val=None):
#     image_with_mask, mask = agent1_process(image, model1)
#     image_rgb = agent2_process(image, mask, model2)
#     return image_with_mask, image_rgb


def preprocess_frame(frame):
    # resized = cv2.resize(frame, (128, 128))
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


# Batch inference and postprocessing
def infer_and_postprocess(frames, AGENT1_NAME, AGENT2_NAME, batch_size=32):
    results = []
    for i in tqdm(range(0, len(frames), batch_size), "Predicting color in batches"):
        batch = frames[i:i+batch_size]

        # Step-by-step expansion to ensure correct shape
        batch_array = np.array(batch, dtype=np.float32) / 255.0  # (B, 128, 128)
        batch_array = np.expand_dims(batch_array, axis=-1)       # (B, 128, 128, 1)

        # rgb_pred = np.zeros((batch_size, 128, 128, 3), dtype=np.float32)
        rgb_pred = []
        for j in range(min(batch_size, len(frames)-i)):
            # rgb_pred[i] = create_image(batch_array[i], 128, AGENT1_NAME, AGENT2_NAME, custom_objects)[1]
            rgb_pred.append(create_image(batch_array[j], 128, AGENT1_NAME, AGENT2_NAME, custom_objects, is_video=True)[1])
        # rgb_pred = create_image(batch_array, 128, AGENT1_NAME, AGENT2_NAME, custom_objects)[1]         # Output: (B, 128, 128, 3)
        rgb_pred = np.array(rgb_pred)
        rgb_uint8 = (rgb_pred * 255).astype(np.uint8)
        results.extend(rgb_uint8)
    return results

def create_video_from_gray_to_rgb(input_path: str, output_path: str, AGENT1_NAME: str, AGENT2_NAME: str) -> None:
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_w, original_h))
    # Read all frames
    frames = []
    for _ in tqdm(range(frame_count), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Preprocess frames in parallel: center crop + grayscale
    with Pool(cpu_count() // 2) as pool:
        gray_frames = list(tqdm(pool.map(preprocess_frame, frames), total=len(frames), desc="Preprocessing"))

    # Inference
    """
    iterable = [[frame, AGENT1_NAME, AGENT2_NAME] for frame in gray_frames]
    print("Total frames", len(iterable))
    with Pool(cpu_count() // 2) as pool:
        rgb_frames = list(tqdm(pool.map(frame_paralel.infer_and_postprocess_paralel, iterable, chunksize=1),
                               total=len(iterable), desc="Color prediction"))
                               """
    rgb_frames = infer_and_postprocess(gray_frames, AGENT1_NAME, AGENT2_NAME, 8)

    # Resize output frames back to original size and write video

    for rgb_frame in tqdm(rgb_frames, desc="Writing output"):
        rgb_upscaled = cv2.resize(rgb_frame, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        rgb_upscaled = cv2.cvtColor(rgb_upscaled, cv2.COLOR_RGB2BGR)
        out.write(rgb_upscaled)
    out.release()

    print(f"✅ Done! Video written to: {output_path}")


if __name__ == "__main__":
    create_video_from_gray_to_rgb(input_path, output_path, AGENT1_NAME, AGENT2_NAME)
