import design as des
import some_functions as sf
import datasets_from_loader_utils as dflu
import dataset_loader as dl

import streamlit as st
import os

st.session_state.IMAGE_SIZE = dl.IMAGE_SIZE
if (
    os.path.exists("./tfrecords/Agent1_val.tfrecord")
    and os.path.exists("./tfrecords/Agent2_val_hsv.tfrecord")
    and ("dataset_agent1" not in st.session_state
         or "dataset_agent2" not in st.session_state)
):    
    dl.BATCH_SIZE = 32
    VAL_STEPS = 2
    st.session_state.dataset_agent1 = dl.coco_RGB_dataset_precomputed(
        split="val",
        channels=1,
        tfrecord_path="tfrecords/Agent1_val.tfrecord"
    ).take(VAL_STEPS)
    st.session_state.dataset_agent2 = dl.coco_RGB_dataset_precomputed_agent2(
        tfrecord_path="tfrecords/Agent2_val_hsv.tfrecord"
    ).take(VAL_STEPS)

st.session_state.rgb_colors = dflu.coco_rgb_colors

st.session_state.custom_objects = {
    name: getattr(sf, name) for name in sf.__all__
}


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
