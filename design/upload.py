from .agents import mother_agent

import streamlit as st

from PIL import Image
import cv2

import numpy as np


def upload():
    uploaded_file = st.file_uploader("Upload an image to process", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Process"):
        input_images = []
        processed_images = []
        for file in uploaded_file:
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

        st.session_state.input_images = input_images
        st.session_state.processed_images = processed_images
        st.session_state.step = "review"
        st.experimental_rerun()
