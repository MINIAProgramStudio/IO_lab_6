from .agents import mother_agent

import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO
import subprocess
import tempfile
import base64
import json
import sys
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


def reproccess_images():
    st.session_state.processed_images = [
        mother_agent(
            img,
            st.session_state.selected_agent1,
            st.session_state.selected_agent2
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
            print("Done")
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
                            loss = model.evaluate(st.session_state.dataset_agent1)[0]
                        case 2:
                            loss = model.evaluate(st.session_state.dataset_agent2)[0]
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
