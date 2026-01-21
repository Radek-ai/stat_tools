"""
Results Analysis Page - Treatment Effect Analysis Tool
"""
import streamlit as st

from results_analysis.ui_components import (
    render_data_upload,
    render_configuration,
    render_basic_analysis,
    render_cuped_analysis,
    render_did_analysis
)
from utils.streamlit_artifacts import render_download_artifact_button as _render_download_artifact_button


def render_download_artifact_button():
    """Render the download artifact button"""
    _render_download_artifact_button(
        page_name="results_analysis",
        state_key="results_analysis_artifact",
        file_name="artifact_results_analysis.zip",
        help_disabled="Upload data and perform analysis to create an artifact",
    )


def show_results_analysis_page():
    """Main function to render the Results Analysis page"""
    st.title("📈 Experiment Results Analysis")
    st.markdown("Analyze treatment effects, uplifts, and statistical significance of your A/B test results")
    
    # Create tabs
    tab_upload, tab_config, tab_analysis = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "📊 Analysis"
    ])
    
    with tab_upload:
        render_data_upload()
    
    with tab_config:
        render_configuration()
    
    with tab_analysis:
        # Sub-tabs for different analysis types
        sub_tab_basic, sub_tab_cuped, sub_tab_did = st.tabs([
            "📊 Basic Analysis",
            "🔬 CUPED Analysis",
            "📉 Difference-in-Differences"
        ])
        
        with sub_tab_basic:
            render_basic_analysis()
        
        with sub_tab_cuped:
            render_cuped_analysis()
        
        with sub_tab_did:
            render_did_analysis()
