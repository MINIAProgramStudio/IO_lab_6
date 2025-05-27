import design as des
import some_functions as sf
import datasets_from_loader_utils as dflu
import dataset_loader as dl

import streamlit as st


st.session_state.custom_objects = {
    name: getattr(sf, name) for name in sf.__all__
}
st.session_state.rgb_colors = dflu.coco_rgb_colors
st.session_state.IMAGE_SIZE = dl.IMAGE_SIZE

# st.set_page_config(layout="centered")
st.title("Drawer")
st.write()

des.settings()

if "step" not in st.session_state:
    st.session_state.dict = {
        "upload": des.upload,
        "review": des.review
    }
    st.session_state.step = "upload"

st.session_state.dict.get(st.session_state.step)()
