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

AGENT1_NAME = "unet32hsv_500.keras"
AGENT2_NAME = "test45_e65_l0.0939.keras"
input_path = 'demonstrator_video/A Trip to the Moon (1902) Georges Méliès.mp4'
output_path = 'demonstrator_video/output_video.mp4'
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
def infer_and_postprocess(gray_frames, batch_size=32):
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

def main():
    global input_path
    global output_path
    target_size = (128, 128)

    # Read frames with progress bar
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    frames = []
    for _ in tqdm(range(frame_count), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Parallel preprocessing
    with Pool(cpu_count()) as pool:
        gray_frames = list(tqdm(pool.imap(preprocess_frame, frames), total=len(frames), desc="Preprocessing"))

    # Inference
    rgb_frames = infer_and_postprocess(gray_frames, batch_size=32)

    # Writing output
    for frame in tqdm(rgb_frames, desc="Writing output"):
        out.write(frame)
    out.release()
    print(f"✅ Done! Video written to: {output_path}")

if __name__ == "__main__":
    main()
