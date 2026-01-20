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
from utils.artifact_builder import ArtifactBuilder


def render_download_artifact_button():
    """Render the download artifact button"""
    # Initialize artifact builder
    if 'results_analysis_artifact' not in st.session_state:
        st.session_state.results_analysis_artifact = ArtifactBuilder(page_name='results_analysis')
    
    artifact = st.session_state.results_analysis_artifact
    
    # Check if there's anything to download (dataframes, plots, config, or logs)
    has_data = len(artifact.dataframes) > 0 or len(artifact.plots) > 0 or len(artifact.config) > 0 or len(artifact.logs) > 0
    
    if has_data:
        try:
            zip_bytes = artifact.create_zip()
            st.download_button(
                label="💾 Download Artifact",
                data=zip_bytes,
                file_name=f"artifact_results_analysis.zip",
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
            help="Upload data and perform analysis to create an artifact"
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
