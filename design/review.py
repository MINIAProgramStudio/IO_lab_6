from .functions import reproccess_images

import streamlit as st
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


def review():
    st.markdown(
        f"## Input image with: \n##### Agent 1: {st.session_state.selected_agent1} \n##### Agent 2: {st.session_state.selected_agent2}"
    )

    # 2. TODO: PERCENTAGE OF DISPLAY
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Original Images")
        for img in st.session_state.input_images:
            st.image(img, width=128)

    with col2:
        st.markdown("### Images Agent 1")
        for img in st.session_state.processed_images:
            st.image(img[0], width=128)
    with col3:
        st.markdown("### Images Agent 2")
        for img in st.session_state.processed_images:
            st.image(img[1], width=128)

    st.markdown("Does the output suit you?")

    if st.session_state.auto_update:
        col1, col2 = st.columns(2)
        with col1:
            _download()

        with col2:
            _upload()

    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            _download()

        with col2:
            if st.button("🔄 Restart"):
                reproccess_images()
                st.experimental_rerun()

        with col3:
            _upload()
