"""
Group Selection Page - Statistical Group Balancing Tool
"""
import streamlit as st

from group_selection.ui_components import (
    render_data_upload,
    render_configuration,
    render_group_balancing
)
from group_selection.guide import render_guide
from utils.streamlit_artifacts import render_download_artifact_button as _render_download_artifact_button


def render_download_artifact_button():
    """Render the download artifact button"""
    _render_download_artifact_button(
        page_name="group_selection",
        state_key="group_selection_artifact",
        file_name="artifact_group_selection.zip",
        help_disabled="Upload data and perform operations to create an artifact",
    )


def show_group_selection_page():
    """Main function to render the Group Selection page"""
    st.title("🎯 Group Selection Tool")
    st.markdown("Upload data, apply filters, and balance groups for A/B testing experiments")
    
    # Create tabs
    tab_upload, tab_config, tab_balancing, tab_instructions = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "⚖️ Group Balancing",
        "📚 How-To Guide"
    ])
    
    with tab_upload:
        render_data_upload()
    
    with tab_config:
        render_configuration()
    
    with tab_balancing:
        render_group_balancing()
    
    with tab_instructions:
        render_guide()
