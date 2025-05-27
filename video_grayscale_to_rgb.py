import cv2
import numpy as np
from tqdm import tqdm

import dataset_loader as dl
import some_functions as sf
import datasets_from_loader_utils as dflu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import tensorflow as tf
layers = tf.keras.layers
from multiprocessing import Pool, cpu_count
from functools import partial
import cv2

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

def agent1_process(image: Image.Image, model_name: str) -> Image.Image:

    try:
        model = tf.keras.models.load_model(
            f"./models/Agent1/{model_name}",
            custom_objects=custom_objects
        )
    except Exception:
        try:
            model = tf.keras.models.load_model(
                f"./models/Agent1/{model_name}",
                custom_objects=custom_objects
            )
        except Exception:
            return

    img = image.copy()  # (128, 128)
    masks = model.predict(img, verbose = 0)  # (128, 128, 9)
    mask = np.argmax(masks, axis=3)  # (128, 128)
    # d3_gray_image = tf.concat([img for _ in range(3)], axis=2)  # (128, 128, 3)

    # rgb_mask = tf.gather(dflu.coco_rgb_colors_tf, mask)  # (?, 128, 128, 3)
    # rgb_mask = (tf.cast(rgb_mask, tf.float32) / 255)  # (?, 128, 128, 3)
    # image_with_mask = d3_gray_image * rgb_mask  # (?, 128, 128, 3)
    image_with_mask = dflu.apply_mask_to_image(img, mask)  # (?, 128, 128, 3)

    # st.image(image_with_mask)
    # draw = ImageDraw.Draw(image_with_mask)
    # draw.text((10, 10), f"Agent 1: {model_name}", fill="red")
    return image_with_mask, mask


def agent2_process(image: Image.Image, mask: Image.Image, model_name: str) -> Image.Image:
    model = tf.keras.models.load_model(
        f"./models/Agent2/{model_name}",
        custom_objects={
            'combined_loss_agent2': sf.combined_loss_agent2,
            'combined_loss_agent2_v2': sf.combined_loss_agent2_v2
        }
    )

    img = image.copy()

    img_true_hsv = model.predict([img, mask], verbose = 0)  # (128, 128, 1) -> (128, 128, 3)

    img_true_rgb = tf.image.hsv_to_rgb(img_true_hsv).numpy()

    # st.image(img_true_rgb)
    # draw = ImageDraw.Draw(img_true_rgb)
    # w, h = img.size
    # draw.text((10, h - 30), f"Agent 2: {model_name}", fill="blue")
    return img_true_rgb


def mother_agent(image, model1, model2, train=None, val=None):
    image_with_mask, mask = agent1_process(image, model1)
    image_rgb = agent2_process(image, mask, model2)
    return image_with_mask, image_rgb

def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray

# Batch inference and postprocessing
def infer_and_postprocess(gray_frames, AGENT1_NAME, AGENT2_NAME, batch_size=32):
    results = []
    for i in tqdm(range(0, len(gray_frames), batch_size), desc="Running inference"):
        batch = gray_frames[i:i+batch_size]

        # Step-by-step expansion to ensure correct shape
        batch_array = np.array(batch, dtype=np.float32) / 255.0  # (B, 128, 128)
        batch_array = np.expand_dims(batch_array, axis=-1)       # (B, 128, 128, 1)

        rgb_pred = mother_agent(batch_array, AGENT1_NAME, AGENT2_NAME)[1]         # Output: (B, 128, 128, 3)
        rgb_uint8 = (rgb_pred * 255).astype(np.uint8)
        results.extend(rgb_uint8)
    return results


def create_video_from_gray_to_rgb(input_path: str, output_path: str, AGENT1_NAME: str, AGENT2_NAME: str) -> None:
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    with Pool(cpu_count()) as pool:
        gray_frames = list(tqdm(pool.imap(preprocess_frame, frames), total=len(frames), desc="Preprocessing"))

    # Inference (assume infer_and_postprocess returns list of RGB frames 128x128 uint8)
    rgb_frames = infer_and_postprocess(gray_frames, AGENT1_NAME, AGENT2_NAME, batch_size=8)

    # Resize output frames back to original size and write video
    for rgb_frame in tqdm(rgb_frames, desc="Writing output"):
        rgb_upscaled = cv2.resize(rgb_frame, (original_w, original_h), interpolation=cv2.INTER_CUBIC)  #, interpolation=cv2.INTER_CUBIC
        rgb_upscaled = cv2.cvtColor(rgb_upscaled, cv2.COLOR_RGB2BGR)
        out.write(rgb_upscaled)
    out.release()

    print(f"✅ Done! Video written to: {output_path}")


if __name__ == "__main__":
    create_video_from_gray_to_rgb(input_path, output_path, AGENT1_NAME, AGENT2_NAME)
