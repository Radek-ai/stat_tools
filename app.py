"""
Main Streamlit Application
"""
import streamlit as st
from power_analysis.streamlit_page import show_power_analysis_page
from group_selection.streamlit_page import show_group_selection_page
from rebalancer.streamlit_page import show_rebalancer_page
# Page configuration
st.set_page_config(
    page_title="Statistical Analysis Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# Navigation function
def navigate_to(page_name):
    """Navigate to a specific page"""
    st.session_state.current_page = page_name
    st.rerun()

# Main page content
if st.session_state.current_page == 'main':
    st.title("📊 Statistical Analysis Tool Suite")
    st.markdown("Select an analysis tool to get started:")
    
    # Create 4 columns for the boxes
    col1, col2, col3, col4 = st.columns(4)
    
    # Box 1: Power Analysis
    with col1:
        st.markdown("""
        <div style='border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; height: 100%; background-color: #f9f9f9;'>
            <h2 style='color: #1f77b4; margin-top: 0;'>📊 Power Analysis</h2>
            <p style='color: #666; font-size: 14px; line-height: 1.6;'>
                Comprehensive statistical power analysis tool for A/B testing experiments. Upload your data or provide 
                pre-computed statistics to determine optimal sample sizes across multiple metrics and scenarios. 
                Configure all statistical parameters (uplift, alpha, power) and explore results through interactive 
                visualizations and contour maps.
            </p>
            <hr style='margin: 15px 0;'>
            <p style='color: #888; font-size: 12px;'><strong>Key Capabilities:</strong></p>
            <ul style='color: #888; font-size: 12px; margin: 10px 0; padding-left: 20px; line-height: 1.8;'>
                <li>Upload CSV data or input statistics directly</li>
                <li>Test multiple outlier filtering scenarios</li>
                <li>Analyze multiple metrics simultaneously</li>
                <li>Fully customizable statistical parameters</li>
                <li>Interactive sample size calculator</li>
                <li>Advanced contour maps and visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Power Analysis", key="launch_power", use_container_width=True, type="primary"):
            navigate_to('power_analysis')
    
    # Box 2: Group Selection
    with col2:
        st.markdown("""
        <div style='border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; height: 100%; background-color: #f9f9f9;'>
            <h2 style='color: #1f77b4; margin-top: 0;'>🎯 Group Selection</h2>
            <p style='color: #666; font-size: 14px; line-height: 1.6;'>
                Upload your data and apply filters to remove outliers or filter by specific values before balancing groups 
                for A/B testing experiments. Supports multiple filtering methods including percentile-based and IQR outlier removal.
            </p>
            <hr style='margin: 15px 0;'>
            <p style='color: #888; font-size: 12px;'><strong>Key Capabilities:</strong></p>
            <ul style='color: #888; font-size: 12px; margin: 10px 0; padding-left: 20px; line-height: 1.8;'>
                <li>Upload CSV data files</li>
                <li>Outlier filtering (percentile, IQR)</li>
                <li>Value-based filtering (numeric ranges)</li>
                <li>Categorical value filtering</li>
                <li>Preview filtered data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Group Selection", key="launch_group_selection", use_container_width=True, type="primary"):
            navigate_to('group_selection')
    
    # Box 3: Rebalancer
    with col3:
        st.markdown("""
        <div style='border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; height: 100%; background-color: #f9f9f9;'>
            <h2 style='color: #1f77b4; margin-top: 0;'>⚖️ Group Rebalancer</h2>
            <p style='color: #666; font-size: 14px; line-height: 1.6;'>
                Rebalance existing groups by trimming rows when experiments are already running. Supports multiple groups 
                and uses intelligent middle/odd group detection to minimize loss while balancing.
            </p>
            <hr style='margin: 15px 0;'>
            <p style='color: #888; font-size: 12px;'><strong>Key Capabilities:</strong></p>
            <ul style='color: #888; font-size: 12px; margin: 10px 0; padding-left: 20px; line-height: 1.8;'>
                <li>Upload data with existing groups</li>
                <li>Even size seed search (Basic mode)</li>
                <li>Full iterative rebalancing (Advanced mode)</li>
                <li>Multi-group support with smart pairing</li>
                <li>Balance evaluation and visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Rebalancer", key="launch_rebalancer", use_container_width=True, type="primary"):
            navigate_to('rebalancer')
    
    # Box 4: Placeholder
    with col4:
        st.markdown("""
        <div style='border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; height: 100%; background-color: #f9f9f9; opacity: 0.6;'>
            <h2 style='color: #999; margin-top: 0;'>📊 Coming Soon</h2>
            <p style='color: #999; font-size: 14px;'>Additional analysis tools will be available here.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("🔒 Coming Soon", key="launch_4", use_container_width=True, disabled=True)

# Power Analysis Page
elif st.session_state.current_page == 'power_analysis':
    # Add back button
    if st.button("← Back to Main", key="back_to_main"):
        navigate_to('main')
    show_power_analysis_page()

# Group Selection Page
elif st.session_state.current_page == 'group_selection':
    # Add back button
    if st.button("← Back to Main", key="back_to_main_group"):
        navigate_to('main')
    show_group_selection_page()

# Rebalancer Page
elif st.session_state.current_page == 'rebalancer':
    # Add back button
    if st.button("← Back to Main", key="back_to_main_rebalancer"):
        navigate_to('main')
    show_rebalancer_page()
