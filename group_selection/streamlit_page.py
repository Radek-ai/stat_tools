"""
Group Selection Page - Statistical Group Balancing Tool
"""
import streamlit as st

from group_selection.ui_components import (
    render_data_upload,
    render_configuration,
    render_group_balancing
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
