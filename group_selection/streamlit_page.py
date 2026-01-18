"""
Group Selection Page - Statistical Group Balancing Tool
"""
import streamlit as st

from group_selection.ui_components import (
    render_data_upload_and_filtering,
    render_group_balancing
)


def show_group_selection_page():
    """Main function to render the Group Selection page"""
    st.title("🎯 Group Selection Tool")
    st.markdown("Upload data, apply filters, and balance groups for A/B testing experiments")
    
    # Create tabs
    tab_upload, tab_balancing = st.tabs([
        "📤 Data Upload & Filtering",
        "⚖️ Group Balancing"
    ])
    
    with tab_upload:
        render_data_upload_and_filtering()
    
    with tab_balancing:
        render_group_balancing()
