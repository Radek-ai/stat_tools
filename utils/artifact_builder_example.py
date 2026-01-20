"""
Example usage of ArtifactBuilder

This file demonstrates how to use the ArtifactBuilder class
throughout the application.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.artifact_builder import ArtifactBuilder


def example_usage():
    """
    Example of how to use ArtifactBuilder in a Streamlit page.
    """
    
    # Initialize artifact builder in session state (one per page)
    if 'artifact_builder' not in st.session_state:
        st.session_state.artifact_builder = ArtifactBuilder(page_name='example_page')
    
    artifact = st.session_state.artifact_builder
    
    # Example 1: Add uploaded data
    if st.session_state.get('uploaded_df') is not None:
        df = st.session_state.uploaded_df
        artifact.add_df(
            name='uploaded_data',
            df=df,
            description='Original uploaded data'
        )
        artifact.add_log(
            category='data_upload',
            message=f'Data uploaded: {len(df)} rows, {len(df.columns)} columns',
            details={
                'filename': st.session_state.get('uploaded_filename', 'unknown'),
                'columns': list(df.columns)
            }
        )
    
    # Example 2: Add filtered data
    if st.session_state.get('filtered_data') is not None:
        filtered_df = st.session_state.filtered_data
        artifact.add_df(
            name='filtered_data',
            df=filtered_df,
            description='Data after applying filters'
        )
        artifact.add_log(
            category='filtering',
            message='Outlier filtering applied',
            details={
                'method': 'Percentile',
                'column': 'revenue',
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'rows_removed': len(st.session_state.uploaded_df) - len(filtered_df)
            },
            log_id='filter_outlier_revenue'  # Can be removed later if filters reset
        )
    
    # Example 3: Remove log when filters are reset
    if st.button("Reset Filters"):
        removed_count = artifact.remove_log(category='filtering')
        if 'filtered_data' in artifact.dataframes:
            del artifact.dataframes['filtered_data']
        st.info(f"Removed {removed_count} filter log entries")
    
    # Example 4: Add plot
    if st.session_state.get('balance_fig') is not None:
        fig = st.session_state.balance_fig
        artifact.add_plot(
            name='balance_report',
            fig=fig,
            description='Group balance visualization report'
        )
    
    # Example 5: Add configuration
    artifact.set_config({
        'balancing_mode': 'Advanced',
        'n_groups': 2,
        'group_names': ['Control', 'Treatment'],
        'objectives': {
            'revenue': {'target_p_value': 0.95},
            'region': {'max_imbalance': 5.0}
        }
    })
    
    # Example 6: Download artifact
    if st.button("💾 Save Artifact"):
        zip_bytes = artifact.create_zip()
        st.download_button(
            label="Download Artifact",
            data=zip_bytes,
            file_name=f"artifact_{artifact.page_name}.zip",
            mime="application/zip"
        )
    
    # Show summary
    with st.expander("Artifact Summary"):
        summary = artifact.get_summary()
        st.json(summary)


def example_integration_pattern():
    """
    Pattern for integrating ArtifactBuilder into existing pages.
    """
    
    # Pattern 1: Initialize at page start
    page_name = 'group_selection'
    if f'{page_name}_artifact' not in st.session_state:
        st.session_state[f'{page_name}_artifact'] = ArtifactBuilder(page_name=page_name)
    
    artifact = st.session_state[f'{page_name}_artifact']
    
    # Pattern 2: Add data when uploaded
    def on_data_upload(df, filename):
        artifact.add_df('uploaded_data', df, 'Original uploaded data')
        artifact.add_log(
            category='data_upload',
            message=f'Data uploaded: {filename}',
            details={'rows': len(df), 'columns': len(df.columns)}
        )
    
    # Pattern 3: Add log when filters applied
    def on_filters_applied(filter_config):
        artifact.add_log(
            category='filtering',
            message='Filters applied',
            details=filter_config,
            log_id='current_filters'  # Can be removed if reset
        )
    
    # Pattern 4: Remove logs when filters reset
    def on_filters_reset():
        artifact.remove_log(category='filtering')
        if 'filtered_data' in artifact.dataframes:
            del artifact.dataframes['filtered_data']
    
    # Pattern 5: Add plot when generated
    def on_plot_generated(plot_name, fig, description):
        artifact.add_plot(plot_name, fig, description)
    
    # Pattern 6: Add config when balancing/rebalancing
    def on_balancing_complete(balancing_config):
        artifact.set_config(balancing_config)
    
    # Pattern 7: Download button at the end
    def render_download_section():
        if len(artifact.dataframes) > 0 or len(artifact.plots) > 0:
            zip_bytes = artifact.create_zip()
            st.download_button(
                label="💾 Save Complete Artifact",
                data=zip_bytes,
                file_name=f"artifact_{artifact.page_name}.zip",
                mime="application/zip",
                use_container_width=True
            )
