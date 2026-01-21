"""
Main Streamlit Application
"""
import streamlit as st
from power_analysis.streamlit_page import show_power_analysis_page
from group_selection.streamlit_page import show_group_selection_page
from rebalancer.streamlit_page import show_rebalancer_page
from results_analysis.streamlit_page import show_results_analysis_page
# Page configuration
st.set_page_config(
    page_title="Statistical Analysis Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)
print("Starting the app")
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
    # Header section with better styling
    st.markdown("""
    <div style='text-align: center; padding: 20px 0 30px 0;'>
        <h1 style='color: #1f77b4; margin-bottom: 10px; font-size: 2.5em;'>📊 A/B Testing Experimentation Suite</h1>
        <p style='color: #666; font-size: 1.1em; margin-top: 0;'>Complete workflow from experiment design through group selection to treatment effect analysis rebalancing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create 4 columns for the main tools
    col1, col2, col3, col4 = st.columns(4)
    
    # Box 1: Power Analysis
    with col1:
        st.markdown("""
        <div style='
            border: 1px solid #d0d0d0; 
            border-radius: 12px; 
            padding: 25px; 
            height: 720px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
        '>
            <div style='margin-bottom: 15px;'>
                <h2 style='color: #1f77b4; margin: 0 0 10px 0; font-size: 1.5em; display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.3em;'>📊</span> Power Analysis
                </h2>
                <div style='background: linear-gradient(90deg, #1f77b4, #4a90e2); height: 3px; border-radius: 2px; margin-bottom: 15px;'></div>
            </div>
            <p style='color: #555; font-size: 15px; line-height: 1.7; margin-bottom: 20px; flex-grow: 1; overflow: hidden;'>
                <strong>Design your experiment</strong> by determining the optimal sample sizes needed for your A/B test. 
                Upload your data or provide pre-computed statistics to calculate required sample sizes across multiple metrics 
                and scenarios. Configure statistical parameters (uplift, alpha, power) and explore results through interactive 
                visualizations and contour maps to ensure your experiment is properly powered.
            </p>
            <div style='background-color: #f0f7ff; border-left: 4px solid #1f77b4; padding: 15px; border-radius: 6px; margin-bottom: 0; flex-shrink: 0;'>
                <p style='color: #333; font-size: 13px; font-weight: 600; margin: 0 0 10px 0;'>✨ Key Features:</p>
                <ul style='color: #555; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.9;'>
                    <li>Calculate required sample sizes</li>
                    <li>Test multiple scenarios simultaneously</li>
                    <li>Multi-metric power analysis</li>
                    <li>Fully customizable parameters</li>
                    <li>Interactive visualizations</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Launch Power Analysis", key="launch_power", use_container_width=True, type="primary"):
            navigate_to('power_analysis')
    
    # Box 2: Group Selection
    with col2:
        st.markdown("""
        <div style='
            border: 1px solid #d0d0d0; 
            border-radius: 12px; 
            padding: 25px; 
            height: 720px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
        '>
            <div style='margin-bottom: 15px;'>
                <h2 style='color: #2ca02c; margin: 0 0 10px 0; font-size: 1.5em; display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.3em;'>🎯</span> Group Selection
                </h2>
                <div style='background: linear-gradient(90deg, #2ca02c, #5cb85c); height: 3px; border-radius: 2px; margin-bottom: 15px;'></div>
            </div>
            <p style='color: #555; font-size: 15px; line-height: 1.7; margin-bottom: 20px; flex-grow: 1; overflow: hidden;'>
                <strong>Create perfectly balanced groups</strong> using advanced balancing algorithms with custom objective settings. 
                Set your balancing objectives (target p-values for numeric metrics, imbalance thresholds for categorical variables) 
                and let the algorithm intelligently assign rows to groups to minimize statistical differences. Includes data filtering 
                capabilities to remove outliers before balancing.
            </p>
            <div style='background-color: #f0fff4; border-left: 4px solid #2ca02c; padding: 15px; border-radius: 6px; margin-bottom: 0; flex-shrink: 0;'>
                <p style='color: #333; font-size: 13px; font-weight: 600; margin: 0 0 10px 0;'>✨ Key Features:</p>
                <ul style='color: #555; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.9;'>
                    <li>Custom objective-based balancing</li>
                    <li>Multi-group support</li>
                    <li>Sequential moves & swap algorithms</li>
                    <li>Data filtering & outlier removal</li>
                    <li>Real-time balance evaluation</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Launch Group Selection", key="launch_group_selection", use_container_width=True, type="primary"):
            navigate_to('group_selection')
    
    # Box 3: Rebalancer
    with col3:
        st.markdown("""
        <div style='
            border: 1px solid #d0d0d0; 
            border-radius: 12px; 
            padding: 25px; 
            height: 720px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
        '>
            <div style='margin-bottom: 15px;'>
                <h2 style='color: #ff7f0e; margin: 0 0 10px 0; font-size: 1.5em; display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.3em;'>⚖️</span> Group Rebalancer
                </h2>
                <div style='background: linear-gradient(90deg, #ff7f0e, #ffa64d); height: 3px; border-radius: 2px; margin-bottom: 15px;'></div>
            </div>
            <p style='color: #555; font-size: 15px; line-height: 1.7; margin-bottom: 20px; flex-grow: 1; overflow: hidden;'>
                <strong>Rebalance groups when experiments are already running</strong> and changing group assignments is no longer possible. 
                If you discover your groups are imbalanced after the experiment has started, this tool intelligently trims rows to improve 
                balance while preserving the original group structure. Uses smart middle/odd group detection for multi-group scenarios.
            </p>
            <div style='background-color: #fff8f0; border-left: 4px solid #ff7f0e; padding: 15px; border-radius: 6px; margin-bottom: 0; flex-shrink: 0;'>
                <p style='color: #333; font-size: 13px; font-weight: 600; margin: 0 0 10px 0;'>✨ Key Features:</p>
                <ul style='color: #555; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.9;'>
                    <li>Trim rows from existing groups</li>
                    <li>Preserve original group structure</li>
                    <li>Iterative rebalancing algorithms</li>
                    <li>Multi-group smart pairing</li>
                    <li>Balance evaluation & visualization</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Launch Rebalancer", key="launch_rebalancer", use_container_width=True, type="primary"):
            navigate_to('rebalancer')
    
    # Box 4: Results Analysis
    with col4:
        st.markdown("""
        <div style='
            border: 1px solid #d0d0d0; 
            border-radius: 12px; 
            padding: 25px; 
            height: 720px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
        '>
            <div style='margin-bottom: 15px;'>
                <h2 style='color: #d62728; margin: 0 0 10px 0; font-size: 1.5em; display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.3em;'>📈</span> Results Analysis
                </h2>
                <div style='background: linear-gradient(90deg, #d62728, #ff6b6b); height: 3px; border-radius: 2px; margin-bottom: 15px;'></div>
            </div>
            <p style='color: #555; font-size: 15px; line-height: 1.7; margin-bottom: 20px; flex-grow: 1; overflow: hidden;'>
                <strong>Analyze experiment results</strong> to measure treatment effects, calculate uplifts, and determine statistical significance. 
                Supports basic analysis, CUPED-adjusted effects, and Difference-in-Differences methodology for comprehensive 
                post-experiment evaluation.
            </p>
            <div style='background-color: #fff0f0; border-left: 4px solid #d62728; padding: 15px; border-radius: 6px; margin-bottom: 0; flex-shrink: 0;'>
                <p style='color: #333; font-size: 13px; font-weight: 600; margin: 0 0 10px 0;'>✨ Key Features:</p>
                <ul style='color: #555; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.9;'>
                    <li>Treatment effect analysis</li>
                    <li>Uplift calculations</li>
                    <li>CUPED adjustment</li>
                    <li>Difference-in-Differences</li>
                    <li>Statistical significance</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Launch Results Analysis", key="launch_results", use_container_width=True, type="primary"):
            navigate_to('results_analysis')
    
    # Add footer/info section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #888;'>
        <p style='margin: 5px 0; font-size: 14px;'><strong>Complete A/B Testing Workflow:</strong> Design → Select Groups → (Rebalance) → Analyze Results</p>
        <p style='margin: 5px 0; font-size: 12px; color: #aaa;'>Start with Power Analysis to design your experiment, then use Group Selection to create balanced groups</p>
    </div>
    """, unsafe_allow_html=True)

# Power Analysis Page
elif st.session_state.current_page == 'power_analysis':
    # Add back button and download artifact button in same row
    col_back, col_spacer, col_artifact = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back to Main", key="back_to_main"):
            navigate_to('main')
    with col_artifact:
        # Import here to avoid circular imports
        from power_analysis.streamlit_page import render_download_artifact_button
        render_download_artifact_button()
    show_power_analysis_page()

# Group Selection Page
elif st.session_state.current_page == 'group_selection':
    # Add back button and download artifact button in same row
    col_back, col_spacer, col_artifact = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back to Main", key="back_to_main_group"):
            navigate_to('main')
    with col_artifact:
        # Import here to avoid circular imports
        from group_selection.streamlit_page import render_download_artifact_button
        render_download_artifact_button()
    show_group_selection_page()

# Rebalancer Page
elif st.session_state.current_page == 'rebalancer':
    # Add back button and download artifact button in same row
    col_back, col_spacer, col_artifact = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back to Main", key="back_to_main_rebalancer"):
            navigate_to('main')
    with col_artifact:
        # Import here to avoid circular imports
        from rebalancer.streamlit_page import render_download_artifact_button
        render_download_artifact_button()
    show_rebalancer_page()

# Results Analysis Page
elif st.session_state.current_page == 'results_analysis':
    # Add back button and download artifact button in same row
    col_back, col_spacer, col_artifact = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back to Main", key="back_to_main_results"):
            navigate_to('main')
    with col_artifact:
        # Import here to avoid circular imports
        from results_analysis.streamlit_page import render_download_artifact_button
        render_download_artifact_button()
    show_results_analysis_page()
