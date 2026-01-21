"""
Shared Streamlit upload UI helpers.

These helpers centralize the repeated pattern used across pages:
- optional dummy CSV loader
- CSV file uploader
- store dataframe in st.session_state
- (optional) write to ArtifactBuilder
- show basic stats + preview + column information
"""

from __future__ import annotations

from typing import Optional

import os

import numpy as np
import pandas as pd
import streamlit as st

from utils.artifact_builder import ArtifactBuilder


def render_df_overview(
    df: pd.DataFrame,
    *,
    preview_rows: int = 20,
    preview_expanded: bool = True,
) -> None:
    """Render consistent summary metrics + preview + column information for a dataframe."""
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Rows", f"{len(df):,}")
    with col_stat2:
        st.metric("Total Columns", len(df.columns))
    with col_stat3:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_count)

    with st.expander("📋 Data Preview", expanded=preview_expanded):
        st.dataframe(df.head(preview_rows), use_container_width=True)
        st.caption(f"Showing first {min(preview_rows, len(df))} rows of {len(df):,} total rows")

    with st.expander("📊 Column Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Numeric Columns:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(", ".join(numeric_cols) if numeric_cols else "None found")
        with col2:
            st.write("**Categorical Columns:**")
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            st.write(", ".join(categorical_cols) if categorical_cols else "None found")
        with col3:
            st.write("**All Columns:**")
            st.write(", ".join(df.columns.tolist()))


def render_csv_upload_with_dummy(
    *,
    header: str,
    description: str,
    data_state_key: str,
    filename_state_key: Optional[str],
    uploader_label: str,
    uploader_key: Optional[str],
    uploader_help: Optional[str],
    dummy_file_path: Optional[str],
    dummy_button_key: Optional[str],
    dummy_loaded_filename: Optional[str],
    artifact: Optional[ArtifactBuilder],
    artifact_df_name: str = "uploaded_data",
    artifact_df_description: str = "Original uploaded data",
    artifact_log_category: str = "data_upload",
    show_overview: bool = True,
    clear_data_on_upload_error: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Render a standardized upload section with optional dummy data loader.

    Returns the current dataframe if available, else None.
    """
    # Initialize state
    if data_state_key not in st.session_state:
        st.session_state[data_state_key] = None

    st.header(header)
    st.markdown(description)

    # Dummy data loader
    if dummy_file_path and dummy_button_key:
        with st.expander("🎲 Load Dummy Data", expanded=False):
            st.markdown("Load pre-generated sample data for testing")
            if st.button("🎲 Load Dummy Data", key=dummy_button_key, type="primary"):
                if os.path.exists(dummy_file_path):
                    df_dummy = pd.read_csv(dummy_file_path)
                    st.session_state[data_state_key] = df_dummy
                    if filename_state_key and dummy_loaded_filename:
                        st.session_state[filename_state_key] = dummy_loaded_filename

                    if artifact is not None:
                        artifact.add_df(artifact_df_name, df_dummy, f"{artifact_df_description} (dummy)")
                        artifact.add_log(
                            category=artifact_log_category,
                            message="Dummy data loaded",
                            details={
                                "filename": dummy_loaded_filename or os.path.basename(dummy_file_path),
                                "rows": len(df_dummy),
                                "columns": len(df_dummy.columns),
                                "column_names": list(df_dummy.columns),
                            },
                        )

                    st.success(f"✅ Dummy data loaded! ({len(df_dummy)} rows, {len(df_dummy.columns)} columns)")
                    st.rerun()
                else:
                    st.error(f"❌ Dummy data file not found: {dummy_file_path}")
                    st.info("💡 Run 'python dummy_data_builders/generate_all_dummy_data.py' to generate the files")

    uploader_kwargs = {"type": ["csv"]}
    if uploader_key is not None:
        uploader_kwargs["key"] = uploader_key
    if uploader_help is not None:
        uploader_kwargs["help"] = uploader_help

    uploaded_file = st.file_uploader(uploader_label, **uploader_kwargs)

    df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state[data_state_key] = df
            if filename_state_key:
                st.session_state[filename_state_key] = uploaded_file.name

            if artifact is not None:
                artifact.add_df(artifact_df_name, df, artifact_df_description)
                artifact.add_log(
                    category=artifact_log_category,
                    message=f"Data uploaded: {uploaded_file.name}",
                    details={
                        "filename": uploaded_file.name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns),
                    },
                )

            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            if clear_data_on_upload_error:
                st.session_state[data_state_key] = None
                if filename_state_key and filename_state_key in st.session_state:
                    del st.session_state[filename_state_key]
            df = None

    # If not uploaded this run, use stored state (dummy or prior upload)
    if df is None and st.session_state.get(data_state_key) is not None:
        df = st.session_state.get(data_state_key)

    if df is not None and show_overview:
        render_df_overview(df)

    if df is None:
        st.info("ℹ️ Please upload a CSV file or load dummy data to continue")

    return df

