"""
Main Streamlit Application
"""
import streamlit as st
from power_analysis.streamlit_page import show_power_analysis_page
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
    
    # Box 2: Placeholder
    with col2:
        st.markdown("""
        <div style='border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; height: 100%; background-color: #f9f9f9; opacity: 0.6;'>
            <h2 style='color: #999; margin-top: 0;'>📈 Coming Soon</h2>
            <p style='color: #999; font-size: 14px;'>Additional analysis tools will be available here.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("🔒 Coming Soon", key="launch_2", use_container_width=True, disabled=True)
    
    # Box 3: Placeholder
    with col3:
        st.markdown("""
        <div style='border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; height: 100%; background-color: #f9f9f9; opacity: 0.6;'>
            <h2 style='color: #999; margin-top: 0;'>📉 Coming Soon</h2>
            <p style='color: #999; font-size: 14px;'>Additional analysis tools will be available here.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("🔒 Coming Soon", key="launch_3", use_container_width=True, disabled=True)
    
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
    # Import and run power analysis page
    
