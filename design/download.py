import streamlit as st
import numpy as np
from PIL import Image
import io


def download():
    st.write("### Download Processed Images")
    if "processed_images" in st.session_state:
        for idx, img_array_pair in enumerate(st.session_state.processed_images):
            col1, col2 = st.columns(2)
            for j, col in enumerate([col1, col2]):
                img_array = img_array_pair[j]
                pil_img = Image.fromarray((img_array * 255).astype(np.uint8))

                with col:
                    st.image(pil_img, caption=f"Image {idx+1}-{j+1}", use_column_width=True)

                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)

                    st.download_button(
                        label=f"Download Image {idx+1}-{j+1}",
                        data=buf,
                        file_name=f"processed_image_{idx+1}_{j+1}.png",
                        mime="image/png",
                        key=f"download_{idx}_{j}"
                    )
    st.markdown("---")
    if st.button("Back", use_container_width=True):
        st.session_state.step = "review"
        st.experimental_rerun()
