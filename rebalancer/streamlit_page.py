"""
Rebalancer Page - Post-Experiment Group Balancing Tool
"""
import streamlit as st

from rebalancer.ui_components import (
    render_data_upload,
    render_configuration,
    render_rebalancing
)
from utils.artifact_builder import ArtifactBuilder


def render_download_artifact_button():
    """Render the download artifact button"""
    # Initialize artifact builder
    if 'rebalancer_artifact' not in st.session_state:
        st.session_state.rebalancer_artifact = ArtifactBuilder(page_name='rebalancer')
    
    artifact = st.session_state.rebalancer_artifact
    
    # Check if there's anything to download (dataframes, plots, config, or logs)
    has_data = len(artifact.dataframes) > 0 or len(artifact.plots) > 0 or len(artifact.config) > 0 or len(artifact.logs) > 0
    
    if has_data:
        try:
            zip_bytes = artifact.create_zip()
            st.download_button(
                label="💾 Download Artifact",
                data=zip_bytes,
                file_name=f"artifact_rebalancer.zip",
                mime="application/zip",
                use_container_width=True,
                help="Download complete artifact with data, plots, and transformation log"
            )
        except Exception as e:
            st.error(f"Error creating artifact: {str(e)}")
    else:
        st.download_button(
            label="💾 Download Artifact",
            data=b"",
            file_name="",
            disabled=True,
            use_container_width=True,
            help="Upload data and perform operations to create an artifact"
        )


def show_rebalancer_page():
    """Main function to render the Rebalancer page"""
    st.title("⚖️ Group Rebalancer Tool")
    st.markdown("Upload data with existing groups and rebalance by trimming rows")
    
    # Create tabs
    tab_upload, tab_config, tab_rebalancing = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "⚖️ Rebalancing"
    ])
    
    with tab_upload:
        render_data_upload()
    
    with tab_config:
        render_configuration()
    
    with tab_rebalancing:
        render_rebalancing()
