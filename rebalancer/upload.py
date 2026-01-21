"""
Data upload UI for rebalancer page.
"""

import os

import streamlit as st

from utils.artifact_builder import ArtifactBuilder
from utils.streamlit_upload import render_csv_upload_with_dummy


def render_data_upload():
    """Render the data upload section"""
    # Initialize artifact builder
    if "rebalancer_artifact" not in st.session_state:
        st.session_state.rebalancer_artifact = ArtifactBuilder(page_name="rebalancer")

    render_csv_upload_with_dummy(
        header="📤 Upload Data with Existing Groups",
        description="Upload a CSV file that already contains group assignments. The rebalancer will trim rows to improve balance.",
        data_state_key="rebalancer_uploaded_data",
        filename_state_key="rebalancer_filename",
        uploader_label="Or choose a CSV file",
        uploader_key="rebalancer_file_upload",
        uploader_help="Upload a CSV file with existing group assignments",
        dummy_file_path=os.path.join("dummy_data", "rebalancer_dummy.csv"),
        dummy_button_key="rebalancer_load_dummy",
        dummy_loaded_filename="dummy_rebalancer_data.csv",
        artifact=st.session_state.rebalancer_artifact,
        artifact_df_name="uploaded_data",
        artifact_df_description="Original uploaded data",
        artifact_log_category="data_upload",
        show_overview=True,
    )
