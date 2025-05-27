from .agents import mother_agent

import os
import streamlit as st


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
