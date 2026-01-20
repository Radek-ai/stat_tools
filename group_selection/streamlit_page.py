"""
Group Selection Page - Statistical Group Balancing Tool
"""
import streamlit as st

from group_selection.ui_components import (
    render_data_upload,
    render_configuration,
    render_group_balancing
)
from utils.artifact_builder import ArtifactBuilder


def render_download_artifact_button():
    """Render the download artifact button"""
    # Initialize artifact builder
    if 'group_selection_artifact' not in st.session_state:
        st.session_state.group_selection_artifact = ArtifactBuilder(page_name='group_selection')
    
    artifact = st.session_state.group_selection_artifact
    
    # Check if there's anything to download
    has_data = len(artifact.dataframes) > 0 or len(artifact.plots) > 0
    
    if has_data:
        try:
            zip_bytes = artifact.create_zip()
            st.download_button(
                label="💾 Download Artifact",
                data=zip_bytes,
                file_name=f"artifact_group_selection.zip",
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


def show_group_selection_page():
    """Main function to render the Group Selection page"""
    st.title("🎯 Group Selection Tool")
    st.markdown("Upload data, apply filters, and balance groups for A/B testing experiments")
    
    # Create tabs
    tab_upload, tab_config, tab_balancing = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "⚖️ Group Balancing"
    ])
    
    with tab_upload:
        render_data_upload()
    
    with tab_config:
        render_configuration()
    
    with tab_balancing:
        render_group_balancing()
