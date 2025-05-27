import streamlit as st
from .functions import list_files
from .functions import reproccess_images


def settings():
    with st.sidebar:
        st.header("⚙️ Configuration")

        auto_update = st.checkbox("Auto update on model change", value=False)
        st.session_state.auto_update = auto_update

        # 3. TODO: CHOOSE SMALLEST LOSS AS FIRST
        # with st.spinner("Loading Agent1 models..."):
        agent1_models = list_files("./models/Agent1")
        selected_agent1 = st.selectbox("Select Agent 1 Model", agent1_models) if agent1_models else "No models"
        st.session_state.selected_agent1 = selected_agent1

        # 3. TODO: CHOOSE SMALLEST LOSS AS FIRST
        # with st.spinner("Loading Agent2 models..."):
        agent2_models = list_files("./models/Agent2")
        selected_agent2 = st.selectbox("Select Agent 2 Model", agent2_models) if agent2_models else "No models"
        st.session_state.selected_agent2 = selected_agent2

        if "prev_agent1" not in st.session_state:
            st.session_state.prev_agent1 = selected_agent1
        if "prev_agent2" not in st.session_state:
            st.session_state.prev_agent2 = selected_agent2

        if (
            st.session_state.get("step") == "review"
            and auto_update
            and (selected_agent1 != st.session_state.prev_agent1 or selected_agent2 != st.session_state.prev_agent2)
        ):
            reproccess_images()
            st.session_state.prev_agent1 = selected_agent1
            st.session_state.prev_agent2 = selected_agent2
            st.experimental_rerun()
