from .agents import mother_agent

import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import subprocess
# import tempfile
 #import numba
import base64
import tqdm
import json
import sys
import cv2
import os


def _download():
    if st.button("✅ Yes, download"):
        st.session_state.step = "download"
        st.experimental_rerun()


def _upload():
    if st.button("🔁 No, upload new"):
        st.session_state.step = "upload"
        st.experimental_rerun()


def image_to_base64(image):
    buffered = BytesIO()
    if isinstance(image, Image.Image):
        image.save(buffered, format="PNG")
    else:
        img = Image.open(image)
        img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def resize_image(image, h=128, w=128):
    image = np.array(image)
    image = cv2.resize(image, (h, w))

    if np.max(image) > 1:
        image = image / 255

    return image


def reproccess_images():
    st.session_state.processed_images = [
        create_image(
            img,
            st.session_state.IMAGE_SIZE,
            st.session_state.selected_agent1,
            st.session_state.selected_agent2,
            st.session_state.custom_objects
        ) for img in st.session_state.input_images
    ]


def reproccess_videos():
    processed_video_paths = []
    for file in st.session_state.input_videos:
        save_path = os.path.join("temp_videos", file.name)
        os.makedirs("temp_videos", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(file.read())

        with st.spinner("Processing video..."):
            subprocess.run([
                sys.executable, "connector_design_video.py",
                save_path,
                save_path + "_result.mp4",
                st.session_state.selected_agent1,
                st.session_state.selected_agent2
            ])
            # print("Done")
        processed_video_paths.append({
            "name": file.name,
            "path": save_path + "_result.mp4"
        })
    st.session_state.processed_videos = processed_video_paths


def list_files(folder, endswith=".keras"):
    if endswith:
        return [d for d in os.listdir(folder) if d.endswith(endswith)]
    else:
        return [d for d in os.listdir(folder)]


def find_smallest_loss(folder: list, agent_index: int):
    if (
        os.path.exists("./tfrecords/Agent1_val.tfrecord")
        and os.path.exists("./tfrecords/Agent2_val_hsv.tfrecord")
        and os.path.exists(f"./models/Agent{agent_index}")
    ):
        json_path = f"./models/Agent{agent_index}/models.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                model_losses = json.load(f)
        else:
            model_losses = {}

        smallest_loss = None
        smallest_loss_model = None
        bar = st.progress(0, f"Loading model {folder[0]}...")
        for model_name in folder:
            if model_name in model_losses:
                loss = model_losses[model_name]
            else:
                model = tf.keras.models.load_model(
                    f"./models/Agent{agent_index}/{model_name}",
                    custom_objects=st.session_state.custom_objects
                )
                try:
                    match agent_index:
                        case 1:
                            loss = model.evaluate(st.session_state.dataset_agent1, verbose=1)[0]
                        case 2:
                            loss = model.evaluate(st.session_state.dataset_agent2, verbose=1)
                except Exception as e:
                    print(f"Failed to evaluate {model_name}, error: {e}")
                    loss = float("inf")
                model_losses[model_name] = loss

            if smallest_loss is None or loss < smallest_loss:
                smallest_loss = loss
                smallest_loss_model = model_name

            bar.progress(
                (folder.index(model_name) + 1) / len(folder),
                f"Loading model {model_name}..."
            )

        bar.empty()
        with open(json_path, "w") as f:
            json.dump(model_losses, f, indent=4)

        return folder.index(smallest_loss_model)
    else:
        return 0



def nearest_multiple(x, base=128):
    nearest = int(round(x / base) * base)
    return nearest if nearest >= base else base


def receive_shape(image: np.ndarray[np.uint8] | np.ndarray[np.float32], patch_size: int = 128) -> np.ndarray[int]:
    return image[::patch_size, ::patch_size].shape



def receive_amount(shape):
    result = 1
    for index in shape:
        result *= index
    return result


def split_image(img: np.ndarray[np.uint8] | np.ndarray[np.float32], patch_size: int = 128, is_video: bool = False) -> tuple[np.ndarray[np.float32], int]:
    """Splits image into H * V smaller images. H (int): nearest to patch size, height // patch size. W (int): nearest to patch size, width // patch size

    Args:
        img (np.ndarray[np.uint8] | np.ndarray[np.float32]): Image to split. Can be 3-channel or just gray.
        patch_size (int, optional): Size of smaller images (Depends on model). Defaults to 128.

    Returns:
        \b tuple[np.ndarray[np.float32], int]: (H * W, patch_size, patch_size) Gray images. Amount of images
    """

    original_height = img.shape[0]
    original_width = img.shape[1]
    # print(img.shape)
    if not is_video:
        new_height = min(nearest_multiple(original_height, patch_size), st.session_state.height * patch_size)
        new_width = min(nearest_multiple(original_width, patch_size), st.session_state.width * patch_size)
    else:
        new_height = min(nearest_multiple(original_height, patch_size), 2 * patch_size)
        new_width = min(nearest_multiple(original_width, patch_size), 4 * patch_size)
    # print((new_height, new_width))
    resized_img = cv2.resize(img, (new_width, new_height))
    # resized_img = cv2.cvtColor(resized_img_, cv2.COLOR_RGB2GRAY)

    # print(resized_img.shape)
    small_matrices_shape = receive_shape(resized_img, patch_size)
    # print(small_matrices_shape)

    result = receive_amount(small_matrices_shape)

    matrices = np.zeros((result, patch_size, patch_size))

    if np.max(resized_img) > 1:
        resized_img = resized_img / 255

    row = np.array(np.split(resized_img, patch_size, axis=0))
    col = np.array(np.split(row, patch_size, axis=2))

    for row_index in range(0, patch_size):
        for col_index in range(0, patch_size):
            matrices[:, row_index, col_index] = col[col_index, row_index].reshape(-1)

    # print(small_matrices_shape)
    return matrices, result, small_matrices_shape


def merge_matrices(base_img: np.ndarray[np.float32] | np.ndarray[np.uint8], matrices_3d: np.ndarray[np.float32], patch_size: int = 128, small_matrices_shape=(1, 1)) -> np.ndarray[np.float32]:

    # small_matrices_shape = receive_shape(base_img, patch_size)
    row_step, col_step = small_matrices_shape[0], small_matrices_shape[1]
    output = np.zeros((row_step * patch_size, col_step * patch_size, 3), dtype=np.float32)
    rows, cols = row_step * patch_size, col_step * patch_size

    # print(small_matrices_shape, matrices_3d.shape)
    # print(output.shape, rows, cols)
    for row_index in range(0, rows, row_step):
        for col_index in range(0, cols, col_step):
            output[
                row_index:row_index + row_step,
                col_index:col_index + col_step
            ] = matrices_3d[
                    :,
                    row_index//row_step,
                    col_index//col_step
                ].reshape(
                        row_step,
                        col_step,
                        3
                    )

    return output


def create_image(img: np.ndarray[np.float32] | np.ndarray[np.uint8], patch_size: int = 128, agent1_name: str = "", agent2_name: str = "", custom_objects=None, is_video: bool = False) -> np.ndarray[np.float32]:
    # img = cv2.imread('./datasets/photo_2025-05-28_01-27-02.jpg', cv2.IMREAD_COLOR_RGB)
    img = np.array(img)

    matrices, result, small_matrices_shape = split_image(img, patch_size, is_video)
    matrices_3d = np.zeros((result, patch_size, patch_size, 3))
    matrices_imask_3d = np.zeros((result, patch_size, patch_size, 3))
    matrices_mask_3d = np.zeros((result, patch_size, patch_size, 3))
    # print(matrices.shape)
    iterator = range(matrices.shape[0])
    if not is_video:
        iterator = tqdm.tqdm(iterator)
    for matrix_index in iterator:
        predicat = mother_agent(matrices[matrix_index], agent1_name, agent2_name, custom_objects)
        matrices_3d[matrix_index] = predicat[1]
        matrices_imask_3d[matrix_index] = predicat[0]
        matrices_mask_3d[matrix_index] = predicat[2]

    image_with_mask = merge_matrices(img, matrices_imask_3d, patch_size, small_matrices_shape)
    image_rgb = merge_matrices(img, matrices_3d, patch_size, small_matrices_shape)
    mask = merge_matrices(img, matrices_mask_3d, patch_size, small_matrices_shape)
    return image_with_mask, image_rgb, mask
