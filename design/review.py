from .functions import reproccess_images
from .functions import reproccess_videos
from .functions import _download
from .functions import _upload
# from .functions import image_to_base64
from .functions import resize_image

import streamlit as st




def review():
    st.markdown(
        f"## Input image with: \n##### Agent 1: {st.session_state.selected_agent1} \n##### Agent 2: {st.session_state.selected_agent2}"
    )

    # 2. TODO: PERCENTAGE OF USER DISPLAY, NOT WIDTH
    col1, col2, col3, col4, col5 = st.columns(5)

    for i in range(len(st.session_state.input_images)):
        with col1:
            st.image(resize_image(st.session_state.input_images[i]), width=128, caption="Original")
        with col2:
            st.image(resize_image(st.session_state.processed_images[i][0]), width=128, caption="Agent1 Out")
        with col3:
            st.image(resize_image(st.session_state.processed_images[i][2]), width=128, caption="Only Mask")
        with col4:
            st.image(resize_image(st.session_state.input_images3[i]), width=128, caption="Ruleset")
        with col5:
            st.image(resize_image(st.session_state.processed_images[i][1]), width=128, caption="Agent2 Out")
    # with col1:
    #     st.markdown("### Original Images")
    #     for img in st.session_state.input_images:
    #         st.image(img, width=128)

    # with col2:
    #     st.markdown("### Images Agent 1")
    #     for img in st.session_state.processed_images:
    #         st.image(img[0], width=128)
    # with col3:
    #     st.markdown("### Only Mask Agent 1")
    #     for img in st.session_state.processed_images:
    #         st.image(img[2], width=128)
    # with col4:
    #     st.markdown("### Ruleset")
    #     for img in st.session_state.input_images3:
    #         st.image(img, width=128)
    # with col5:
    #     st.markdown("### Images Agent 2")
    #     for img in st.session_state.processed_images:
    #         st.image(img[1], width=128)
    # col1, col2, col3 = st.columns(3)

    # def show_images(title, images, index=None):
    #     with col1 if index == 0 else col2 if index == 1 else col3:
    #         st.markdown(f"### {title}")
    #         for img in images:
    #             image = img if index is None else img[index]
    #             st.markdown(
    #                 f"<img src='data:image/png;base64,{image_to_base64(image)}' style='width: 90%; height: auto;'>",
    #                 unsafe_allow_html=True
    #             )

    # show_images("Original Images", st.session_state.input_images, index=None)
    # show_images("Images Agent 1", st.session_state.processed_images, index=0)
    # show_images("Images Agent 2", st.session_state.processed_images, index=1)

    st.markdown("## 🎬 Processed Videos")

    for i, video in enumerate(st.session_state.processed_videos):
        st.markdown(f"### Video {i + 1}: `{video['name']}`")
        print(video["path"])
        video_file = open(video["path"], "rb")
        video_bytes = video_file.read()
        st.video(st.session_state.input_videos[i])
        st.video(video_bytes)

        with open(video["path"], "rb") as f:
            st.video(f.read())
            st.download_button(
                label="⬇ Download",
                data=f,
                file_name=f"processed_{video['name']}",
                mime="video/mp4"
            )

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
                reproccess_videos()
                st.experimental_rerun()

        with col3:
            _upload()
