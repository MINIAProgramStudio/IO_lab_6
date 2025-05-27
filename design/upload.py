from .agents import mother_agent

import streamlit as st

from PIL import Image
import os
import cv2
import subprocess
import sys
import numpy as np


def upload():
    uploaded_file = st.file_uploader("Upload an image or video to process", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=True)
    if st.button("Process"):
        input_images = []
        input_videos = []
        processed_images = []
        processed_video_paths = []
        for file in uploaded_file:
            # print(file)
            if not file.name.endswith(".mp4"):
                input_image = Image.open(file).convert("L")
                input_image = cv2.resize(np.array(input_image), (st.session_state.IMAGE_SIZE, st.session_state.IMAGE_SIZE))/255
                input_images.append(input_image)

                with st.spinner(f"Processing {file.name}..."):
                    processed_image = mother_agent(
                        input_image,
                        st.session_state.selected_agent1,
                        st.session_state.selected_agent2,
                    )

                processed_images.append(processed_image)
            else:
                save_path = os.path.join("temp_videos", file.name)
                os.makedirs("temp_videos", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(file.read())

                input_videos.append(file)
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

        st.session_state.input_images = input_images
        st.session_state.input_videos = input_videos
        st.session_state.processed_images = processed_images
        st.session_state.processed_videos = processed_video_paths
        st.session_state.step = "review"
        st.experimental_rerun()
