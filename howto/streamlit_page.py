"""
How-To Guide Page
"""
import streamlit as st


def show_howto_page():
    """Display the general how-to guide page"""
    st.title("📚 How-To Guide")
    st.markdown("Complete workflow guide for using the A/B Testing Experimentation Suite")
    
    # Create tabs
    tab_overview, tab_tools, tab_practices, tab_artifacts = st.tabs([
        "📋 Overview & Workflow",
        "🛠️ Tools Overview",
        "✨ Best Practices",
        "💾 Saving Your Work"
    ])
    
    # Tab 1: Overview & Workflow
    with tab_overview:
        st.header("🎯 Overview")
        st.markdown("""
        This application provides tools for **designing** and **analyzing** A/B tests. The suite consists of four main tools 
        that support your experimentation workflow:
        
        - **Power Analysis** - Design your experiment and determine required sample sizes
        - **Group Selection** - Create balanced groups from your data
        - **Rebalancer** - Rebalance existing groups if needed (optional)
        - **Results Analysis** - Analyze treatment effects and statistical significance
        """)
        
        st.divider()
        
        st.header("🔄 Complete Workflow")
        st.markdown("""
        The typical A/B testing workflow follows these steps:
        
        1. **📊 Power Analysis** - Determine required sample sizes for your experiment
        2. **🎯 Group Selection** - Create balanced groups from your participant data
        3. **🧪 Run Your Experiment** - Conduct the A/B test (outside this tool)
        4. **⚖️ Rebalancer** (Optional) - If needed, rebalance groups after experiment starts
        5. **📈 Results Analysis** - Analyze treatment effects and statistical significance
        
        Each tool includes detailed step-by-step instructions on its respective page.
        """)
        
        st.divider()
        
        st.header("🚀 Getting Started")
        st.markdown("""
        Ready to start? Here's a quick path:
        
        1. Click "📊 Power Analysis" to design your experiment and determine sample size requirements
        2. Collect your participant data and use "🎯 Group Selection" to create balanced groups
        3. Run your experiment with the balanced group assignments
        4. Return here and use "📈 Results Analysis" to evaluate treatment effects
        
        Each tool page includes detailed step-by-step instructions. Start with Power Analysis to begin your experiment design!
        """)
        
        st.info("💡 **Tip:** Each feature page has its own detailed step-by-step instructions. This guide provides the big picture workflow.")
    
    # Tab 2: Tools Overview
    with tab_tools:
        st.header("🛠️ Tools Overview")
        st.markdown("Brief descriptions of each tool in the suite. For detailed step-by-step instructions, see each tool's page.")
        
        st.divider()
        
        with st.expander("📊 Power Analysis", expanded=False):
            st.markdown("""
            **Purpose:** Determine the optimal sample sizes needed for your A/B test to achieve desired statistical power.
            
            **When to use:** Before starting your experiment, when you need to know how many participants/users you need.
            
            **What it does:** Calculates required sample sizes across multiple scenarios, allowing you to explore trade-offs between 
            statistical parameters (uplift, alpha, power) and resource requirements. Provides interactive visualizations including 
            contour maps and calculators.
            
            **💡 Tip:** See the Power Analysis page for complete step-by-step guidance.
            """)
        
        with st.expander("🎯 Group Selection", expanded=False):
            st.markdown("""
            **Purpose:** Create balanced groups from your data using advanced algorithms that minimize statistical differences.
            
            **When to use:** After determining required sample sizes, when you need to assign participants to treatment groups.
            
            **What it does:** Uses intelligent algorithms to assign participants to groups while minimizing differences in numeric 
            metrics (p-values) and categorical variables (imbalance). Supports data filtering, custom objectives, and multiple 
            balancing algorithms. Outputs balanced group assignments ready for your experiment.
            
            **💡 Tip:** See the Group Selection page for complete step-by-step guidance.
            """)
        
        with st.expander("⚖️ Group Rebalancer (Optional)", expanded=False):
            st.markdown("""
            **Purpose:** Improve balance of existing groups by intelligently trimming rows when group assignments cannot be changed.
            
            **When to use:** If you discover your groups are imbalanced after the experiment has already started and you cannot reassign participants.
            
            **What it does:** Intelligently removes participants from groups to improve balance while preserving the original group 
            structure. Useful when experiments are already running and reassignment is not possible.
            
            **💡 Tip:** See the Rebalancer page for complete step-by-step guidance.
            """)
        
        with st.expander("📈 Results Analysis", expanded=False):
            st.markdown("""
            **Purpose:** Analyze experiment results to measure treatment effects, calculate uplifts, and determine statistical significance.
            
            **When to use:** After your experiment has completed and you have collected results data.
            
            **What it does:** Provides comprehensive analysis including basic treatment effects, CUPED-adjusted effects (for pre-experiment 
            data), and Difference-in-Differences methodology. Generates interactive visualizations and statistical summaries.
            
            **💡 Tip:** See the Results Analysis page for complete step-by-step guidance.
            """)
    
    # Tab 3: Best Practices
    with tab_practices:
        st.header("✨ Best Practices")
        st.markdown("Recommendations for getting the most out of the A/B Testing Experimentation Suite.")
        
        st.divider()
        
        st.subheader("📊 Experiment Design")
        st.markdown("""
        - Always start with Power Analysis to ensure adequate sample sizes
        - Consider multiple scenarios and metrics when planning
        - Account for expected drop-off rates in your sample size calculations
        - Test with dummy data first to understand the workflow
        """)
        
        st.divider()
        
        st.subheader("🎯 Group Balancing")
        st.markdown("""
        - Filter outliers before balancing for better results
        - Balance on all relevant covariates, not just primary metrics
        - Use Advanced mode for critical experiments
        - Review balance reports carefully before proceeding with your experiment
        - The algorithm will show loss history - if algorithm did not converge, continue balancing.
        """)
        
        st.divider()
        
        st.subheader("📈 Results Analysis")
        st.markdown("""
        - Use appropriate analysis methods (CUPED/DiD when applicable)
        - Look at both statistical significance (p-values) and practical significance (uplifts, SMD)
        - Consider multiple metrics to get a complete picture
        - Use CUPED when you have pre-experiment data - it can improve statistical power
        - Use DiD when you want to control for time trends
        """)
        
        st.divider()
        
        st.subheader("💾 General Tips")
        st.markdown("""
        - Save artifacts at each step for reproducibility
        - Review all visualizations and reports before making decisions
        - Document your analysis process using artifacts
        - Share artifacts with team members for collaboration
        """)
    
    # Tab 4: Saving Your Work
    with tab_artifacts:
        st.header("💾 Saving Your Work")
        st.markdown("Learn how to preserve and share your work using artifact downloads.")
        
        st.divider()
        
        st.subheader("What Are Artifacts?")
        st.markdown("""
        Every page includes a "Download Artifact" button that creates a ZIP file containing everything you need to 
        reproduce your analysis or share your work with others.
        """)
        
        st.divider()
        
        st.subheader("What's Included")
        st.markdown("""
        The artifact ZIP file contains:
        
        - **Data Files:** CSV files of uploaded, filtered, and processed data
        - **Plots:** HTML files of all generated visualizations (interactive Plotly charts)
        - **Configuration:** JSON file with all settings and parameters used
        - **Transformation Log:** README.txt file documenting all operations performed during your session
        """)
        
        st.divider()
        
        st.subheader("Why Use Artifacts?")
        st.markdown("""
        Artifacts allow you to:
        
        - **Reproduce your analysis exactly** - All data, settings, and transformations are preserved
        - **Share results with team members** - Complete package of your work
        - **Keep a record** - Document your experiment design and analysis process
        - **Audit and review** - Full transparency of what was done and how
        """)
        
        st.divider()
        
        st.subheader("When to Download")
        st.markdown("""
        **Recommended:** Download artifacts after completing each major step:
        
        - After Power Analysis calculations
        - After Group Selection balancing
        - After Rebalancing (if used)
        - After Results Analysis
        
        This ensures you have a complete record at each stage of your workflow.
        """)
        
        st.divider()
        
        st.subheader("How to Use")
        st.markdown("""
        1. Complete your work on any page (upload data, run analysis, generate plots)
        2. Look for the "💾 Download Artifact" button at the top right of the page
        3. Click the button to download the ZIP file
        4. The file will be named `artifact_[page_name]_[timestamp].zip`
        5. Extract and review the contents, or share with your team
        """)
