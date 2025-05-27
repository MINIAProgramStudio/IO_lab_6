import streamlit as st
from .functions import list_files
from .functions import reproccess_images
from .functions import find_smallest_loss


def settings():
    with st.sidebar:
        st.header("⚙️ Configuration")

        auto_update = st.checkbox("Auto update on model change", value=False)
        st.session_state.auto_update = auto_update

        agent1_models = list_files("./models/Agent1")
        if agent1_models:
            if "agent1_models" not in st.session_state or st.session_state.agent1_models != agent1_models:
                st.session_state.smallest_loss_index1 = find_smallest_loss(agent1_models, 1)
                st.session_state.agent1_models = agent1_models

            selected_agent1 = st.selectbox(
                "Select Agent 1 Model",
                agent1_models,
                index=st.session_state.smallest_loss_index1,
            )
        else:
            selected_agent1 = "No models"
        # print(smallest_loss_index1)
        st.session_state.selected_agent1 = selected_agent1

        agent2_models = list_files("./models/Agent2")
        if agent2_models:
            if "agent2_models" not in st.session_state or st.session_state.agent2_models != agent2_models:
                st.session_state.smallest_loss_index2 = find_smallest_loss(agent2_models, 2)
                st.session_state.agent2_models = agent2_models

            selected_agent2 = st.selectbox(
                "Select Agent 2 Model",
                agent2_models,
                index=st.session_state.smallest_loss_index2,
            )
        else:
            selected_agent2 = "No models"
        # print(smallest_loss_index2)
        st.session_state.selected_agent2 = selected_agent2

        if "prev_agent1" not in st.session_state:
            st.session_state.prev_agent1 = selected_agent1
        if "prev_agent2" not in st.session_state:
            st.session_state.prev_agent2 = selected_agent2

        if (
            st.session_state.get("step") != "upload"
            and auto_update
            and (selected_agent1 != st.session_state.prev_agent1 or selected_agent2 != st.session_state.prev_agent2)
        ):
            reproccess_images()
            st.session_state.prev_agent1 = selected_agent1
            st.session_state.prev_agent2 = selected_agent2
            st.experimental_rerun()
