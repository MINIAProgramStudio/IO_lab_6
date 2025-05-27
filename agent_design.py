import streamlit as st

from PIL import Image, ImageDraw
import cv2

import os
import io

import tensorflow as tf
import numpy as np

import dataset_loader as dl
import datasets_from_loader_utils as dflu
import some_functions as sf

st.set_page_config(layout="centered")
st.title("Drawer")


# Simulated agents — in reality, you'd load models with tf.keras.models.load_model()
def agent1_process(image: Image.Image, model_name: str) -> Image.Image:

    try:
        model = tf.keras.models.load_model(
            f"./models/Agent1/{model_name}",
            custom_objects={
                'weighted_sparse_categorical_crossentropy': sf.weighted_sparse_categorical_crossentropy,
                'WeightedMeanIoU': sf.WeightedMeanIoU(num_classes=dl.COCO_NUM_CLASSES)
            }
        )
    except Exception:
        try:
            model = tf.keras.models.load_model(
                f"./models/Agent1/{model_name}",
                custom_objects={
                    'weighted_combined_loss': sf.weighted_combined_loss,
                    'WeightedMeanIoU': sf.WeightedMeanIoU(num_classes=dl.COCO_NUM_CLASSES)
                }
            )
        except Exception:
            st.error(f"Error loading model: {model_name}")
            return

    img = image.copy()  # (128, 128)
    masks = model.predict(tf.expand_dims(img, 0))[0]  # (128, 128, 9)
    mask = np.argmax(masks, axis=2)  # (128, 128)
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

    img_true_hsv = model.predict([tf.expand_dims(img, 0), tf.expand_dims(mask, 0)])[0]  # (128, 128, 1) -> (128, 128, 3)
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

# 🗂️ Helper to list files and directories
def list_files(folder, endswith=None):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if endswith:
        files = [f for f in files if f.endswith(endswith)]
    return files

def list_dirs(folder):
    return [d for d in os.listdir(folder) if d.endswith(".keras")]

# 📂 File/Model selection
with st.sidebar:
    st.header("⚙️ Configuration")

    auto_update = st.checkbox("Auto update on model change", value=False)
    st.session_state.auto_update = auto_update

    # tfrecord_train_files = list_files("./tfrecords", ".tfrecord")
    # selected_tfrecord_train = st.selectbox("Select TFRecord train file", tfrecord_train_files) if tfrecord_train_files else "No TFRecords"
    selected_tfrecord_train = None

    # tfrecord_val_files = list_files("./tfrecords", ".tfrecord")
    # selected_tfrecord_val = st.selectbox("Select TFRecord val file", tfrecord_val_files) if tfrecord_val_files else "No TFRecords"
    selected_tfrecord_val = None

    agent1_models = list_files("./models/Agent1", ".keras")
    selected_agent1 = st.selectbox("Select Agent 1 Model", agent1_models) if agent1_models else "No models"

    agent2_models = list_files("./models/Agent2")
    selected_agent2 = st.selectbox("Select Agent 2 Model", agent2_models) if agent2_models else "No models"

    if "prev_agent1" not in st.session_state:
        st.session_state.prev_agent1 = selected_agent1
    if "prev_agent2" not in st.session_state:
        st.session_state.prev_agent2 = selected_agent2

    if (
        st.session_state.get("step") == "review"
        and auto_update
        and (selected_agent1 != st.session_state.prev_agent1 or selected_agent2 != st.session_state.prev_agent2)
    ):
        st.session_state.processed_images = [
            mother_agent(img, selected_agent1, selected_agent2, selected_tfrecord_train, selected_tfrecord_val)
            for img in st.session_state.input_images
        ]
        st.session_state.prev_agent1 = selected_agent1
        st.session_state.prev_agent2 = selected_agent2
        st.experimental_rerun()

# 🌄 Upload image and run agents
if "step" not in st.session_state:
    st.session_state.step = "upload"

if st.session_state.step == "upload":
    uploaded_file = []
    uploaded_file += st.file_uploader("Upload an image to process", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Process"):
        input_images = []
        processed_images = []
        for file in uploaded_file:
            input_image = Image.open(file).convert("L")
            input_image = cv2.resize(np.array(input_image), (128, 128))/255
            input_images.append(input_image)

            processed_image = mother_agent(
                input_image,
                selected_agent1, selected_agent2,
                selected_tfrecord_train, selected_tfrecord_val
            )
            processed_images.append(processed_image)

        st.session_state.input_images = input_images
        st.session_state.processed_images = processed_images
        st.session_state.step = "review"
        st.experimental_rerun()

if st.session_state.step == "review":
    st.markdown(f"## Input image with: \n##### Agent 1: {selected_agent1} \n##### Agent 2: {selected_agent2}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Original Images")
        for img in st.session_state.input_images:
            st.image(img, width = 128)

    with col2:
        st.markdown("### Images Agent 1")
        for img in st.session_state.processed_images:
            st.image(img[0], width = 128)
    with col3:
        st.markdown("### Images Agent 2")
        for img in st.session_state.processed_images:
            st.image(img[1], width = 128)

    st.markdown("Does the output suit you?")

    if auto_update:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, download"):
                buf = io.BytesIO()
                st.session_state.processed_image.save(buf, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

        with col2:
            if st.button("🔁 No, upload new"):
                st.session_state.step = "upload"
                st.experimental_rerun()

    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Yes, download"):
                buf = io.BytesIO()
                st.session_state.processed_image.save(buf, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

        with col2:
            if st.button("🔄 Restart"):
                st.session_state.processed_images = [
                    mother_agent(
                        img,
                        selected_agent1, selected_agent2,
                        selected_tfrecord_train, selected_tfrecord_val
                    )
                    for img in st.session_state.input_images
                ]
                st.experimental_rerun()

        with col3:
            if st.button("🔁 No, upload new"):
                st.session_state.step = "upload"
                st.experimental_rerun()
