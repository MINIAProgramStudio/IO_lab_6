import streamlit as st
import tensorflow as tf
import numpy as np


def agent1_process(image: np.ndarray, model_name: str) -> np.ndarray:
    try:
        model = tf.keras.models.load_model(
            f"./models/Agent1/{model_name}",
            custom_objects=st.session_state.custom_objects
        )
    except Exception:
        st.error(f"Error loading model: {model_name}")
        return

    img = image.copy()  # (128, 128)

    masks = model.predict(tf.expand_dims(img, 0))[0]  # (128, 128, 9)
    mask = np.argmax(masks, axis=2)  # (128, 128)

    d3_gray_image = np.concatenate([image[..., None] for _ in range(3)], axis=2)  # (128, 128, 3)
    d3_mask = st.session_state.rgb_colors[mask]  # (128, 128, 3)
    image_with_mask = d3_gray_image * (d3_mask / 255)  # (128, 128, 3)

    return image_with_mask, mask


def agent2_process(image: np.ndarray, mask: np.ndarray, model_name: str) -> np.ndarray:
    try:
        model = tf.keras.models.load_model(
            f"./models/Agent2/{model_name}",
            custom_objects=st.session_state.custom_objects
        )
    except Exception:
        st.error(f"Error loading model: {model_name}")
        return

    img = image.copy()

    img_true_hsv = model.predict([tf.expand_dims(img, 0), tf.expand_dims(mask, 0)])[0]  # (128, 128, 1) -> (128, 128, 3)
    img_true_rgb = tf.image.hsv_to_rgb(img_true_hsv).numpy()
    return img_true_rgb


# 4. TODO: CHECK IF TENSORFLOW CAN RUN
def mother_agent(image: np.ndarray, model1: str, model2: str):
    image_with_mask, mask = agent1_process(image, model1)
    image_rgb = agent2_process(image, mask, model2)
    return image_with_mask, image_rgb
