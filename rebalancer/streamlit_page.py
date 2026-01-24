"""
Rebalancer Page - Post-Experiment Group Balancing Tool
"""
import streamlit as st

from rebalancer.ui_components import (
    render_data_upload,
    render_configuration,
    render_rebalancing
)
from rebalancer.guide import render_guide
from utils.streamlit_artifacts import render_download_artifact_button as _render_download_artifact_button


def render_download_artifact_button():
    """Render the download artifact button"""
    _render_download_artifact_button(
        page_name="rebalancer",
        state_key="rebalancer_artifact",
        file_name="artifact_rebalancer.zip",
        help_disabled="Upload data and perform operations to create an artifact",
    )


def show_rebalancer_page():
    """Main function to render the Rebalancer page"""
    st.title("⚖️ Group Rebalancer Tool")
    st.markdown("Upload data with existing groups and rebalance by trimming rows")
    
    # Create tabs
    tab_upload, tab_config, tab_rebalancing, tab_instructions = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "⚖️ Rebalancing",
        "📚 How-To Guide"
    ])
    
    with tab_upload:
        render_data_upload()
    
    with tab_config:
        render_configuration()
    
    with tab_rebalancing:
        render_rebalancing()
    
    with tab_instructions:
        render_guide()
