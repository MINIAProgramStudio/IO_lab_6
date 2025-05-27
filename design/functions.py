from .agents import mother_agent

import tensorflow as tf
import streamlit as st
import json
import os
import io


# 1. TODO: FIX DOWNLOAD BUTTON
def _download():
    if st.button("✅ Yes, download"):
        buf = io.BytesIO()
        st.session_state.processed_image.save(buf, format="PNG")
        st.download_button(
            label="Download Processed Image",
            data=buf.getvalue(),
            file_name="processed_image.png",
            mime="image/png",
        )


def _upload():
    if st.button("🔁 No, upload new"):
        st.session_state.step = "upload"
        st.experimental_rerun()


def reproccess_images():
    st.session_state.processed_images = [
        mother_agent(
            img,
            st.session_state.selected_agent1,
            st.session_state.selected_agent2
        ) for img in st.session_state.input_images
    ]


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
                            loss = model.evaluate(st.session_state.dataset_agent1)
                        case 2:
                            loss = model.evaluate(st.session_state.dataset_agent2)
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
