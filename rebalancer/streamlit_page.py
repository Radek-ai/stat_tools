"""
Rebalancer Page - Post-Experiment Group Balancing Tool
"""
import streamlit as st

from rebalancer.ui_components import (
    render_data_upload,
    render_configuration,
    render_rebalancing
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
