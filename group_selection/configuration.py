"""
Configuration UI for group selection page (filtering).
"""

import streamlit as st

from utils.artifact_builder import ArtifactBuilder
from utils.streamlit_filters import render_filtering_tabs, render_apply_reset_filters


def render_configuration():
    """Render the configuration section (filtering and group setup)"""
    # Initialize session state
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    
    # Get artifact builder
    artifact = st.session_state.get('group_selection_artifact')
    if artifact is None:
        artifact = ArtifactBuilder(page_name='group_selection')
        st.session_state.group_selection_artifact = artifact
    
    if st.session_state.get('uploaded_data_raw') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    st.header("⚙️ Configuration")
    st.markdown("Apply filters and configure group settings")
    
    # Data Filtering Section
    st.subheader("🔍 Data Filtering")
    st.markdown("Apply filters to remove outliers or filter by specific values")
    
    df = st.session_state.uploaded_data_raw.copy()

    filtered_df, filter_config = render_filtering_tabs(
        df,
        key_prefix="group_selection_",
        exclude_columns=None,
        exclude_id_columns=False,
    )

    render_apply_reset_filters(
        original_df=df,
        filtered_df=filtered_df,
        filtered_state_key="filtered_data",
        key_prefix="group_selection_",
        artifact=artifact,
        artifact_filtered_df_name="filtered_data",
        artifact_filtered_df_description="Data after applying filters",
        artifact_log_category="filtering",
        artifact_log_id="current_filters",
        filter_config=filter_config,
    )
    
    # Show filtered data summary
    if st.session_state.filtered_data is not None:
        st.subheader("📊 Filtered Data Summary")
        filtered_df = st.session_state.filtered_data
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Original Rows", f"{len(df):,}")
        with col_sum2:
            st.metric("Filtered Rows", f"{len(filtered_df):,}")
        with col_sum3:
            removed = len(df) - len(filtered_df)
            st.metric("Removed", f"{removed:,}", f"-{removed/len(df)*100:.1f}%")
        
        with st.expander("📋 Filtered Data Preview"):
            st.dataframe(filtered_df.head(20), use_container_width=True)
    
    st.divider()
    
    if st.session_state.filtered_data is not None:
        df_for_balancing = st.session_state.filtered_data
    elif st.session_state.uploaded_data_raw is not None:
        df_for_balancing = st.session_state.uploaded_data_raw
    else:
        st.warning("⚠️ Please upload data and apply filters (if needed) first")
        return
