"""
Data upload UI for group selection page.
"""

import os

import streamlit as st

from utils.artifact_builder import ArtifactBuilder
from utils.streamlit_upload import render_csv_upload_with_dummy


def render_data_upload():
    """Render the data upload section"""
    # Initialize artifact builder
    if "group_selection_artifact" not in st.session_state:
        st.session_state.group_selection_artifact = ArtifactBuilder(page_name="group_selection")

    render_csv_upload_with_dummy(
        header="📤 Upload Data",
        description="Upload CSV data for group balancing",
        data_state_key="uploaded_data_raw",
        filename_state_key="uploaded_filename",
        uploader_label="Or choose a CSV file",
        uploader_key=None,  # keep Streamlit's default key behavior
        uploader_help="Upload a CSV file with your data for group balancing",
        dummy_file_path=os.path.join("dummy_data", "group_selection_dummy.csv"),
        dummy_button_key="group_load_dummy",
        dummy_loaded_filename="dummy_data.csv",
        artifact=st.session_state.group_selection_artifact,
        artifact_df_name="uploaded_data",
        artifact_df_description="Original uploaded data",
        artifact_log_category="data_upload",
        show_overview=True,
        clear_data_on_upload_error=True,
    )
