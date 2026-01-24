"""
How-To Guide for Results Analysis page
"""
import streamlit as st


def render_guide():
    """Render the Results Analysis how-to guide"""
    st.header("📚 How-To Guide")
    st.markdown("Complete guide for using the Experiment Results Analysis tool")
    
    st.divider()
    
    # Key Concepts
    st.header("🎯 Key Concepts")
    
    st.subheader("What is Results Analysis?")
    st.markdown("""
    Results Analysis helps you **analyze treatment effects** from your A/B test by comparing metrics between groups, 
    calculating uplifts, and assessing statistical significance. The tool provides three analysis methods:
    
    - **Basic Analysis:** Direct comparison of metrics between groups
    - **CUPED Analysis:** Variance reduction using pre-experiment data to improve precision
    - **Difference-in-Differences (DiD):** Compares changes between groups across pre/post periods
    
    **When to Use:** After running your A/B test, upload the results data to analyze treatment effects, 
    calculate uplifts, and determine statistical significance.
    """)
    
    st.subheader("Analysis Methods")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Basic Analysis**
        
        Direct comparison of metrics between groups.
        
        - Compares means, calculates uplifts, SMD, and p-values
        - Simple and straightforward
        - Works with any numeric metrics
        - Best for: Binary outcomes, simple metrics, quick analysis
        """)
    
    with col2:
        st.markdown("""
        **🔬 CUPED Analysis**
        
        Uses pre-experiment data to reduce variance and improve precision.
        
        - Adjusts post-experiment metrics using pre-experiment baseline
        - Reduces variance, improving statistical power
        - Requires pre/post metric pairs (e.g., revenue_pre, revenue_post)
        - Best for: Continuous metrics with baseline data, when you need more precision
        """)
    
    with col3:
        st.markdown("""
        **📉 Difference-in-Differences**
        
        Compares changes between groups across pre/post periods.
        
        - Calculates DiD gap: (Post_Treatment - Pre_Treatment) - (Post_Control - Pre_Control)
        - Accounts for pre-existing differences and trends
        - Requires pre/post metric pairs
        - Best for: When groups may have different baselines, trend analysis
        """)
    
    st.divider()
    
    # Step-by-Step Guide
    st.header("📋 Step-by-Step Guide")
    
    st.subheader("Step 1: Upload Experiment Results Data")
    st.markdown("""
    **Location:** 📤 Data Upload tab
    
    - Upload a CSV file containing your experiment results, OR
    - Click "🎲 Load Dummy Data" to use sample data for testing
    - The data should contain:
      - **Group column:** Column with group assignments (e.g., "control", "treatment")
      - **Metric columns:** Numeric columns with outcome metrics (e.g., revenue, conversion_rate, customer_churned)
      - **Pre/post columns (for CUPED/DiD):** Columns with pre-experiment and post-experiment values (e.g., revenue_pre, revenue_post)
    - ID columns (all unique values) are automatically excluded
    """)
    
    st.subheader("Step 2: Configure Group Column")
    st.markdown("""
    **Location:** ⚙️ Configuration tab
    
    1. **Select Group Column:**
       - Choose the column containing your group assignments
       - The tool will detect all unique groups
       - Group sizes and statistics will be displayed
    
    2. **Review Group Information:**
       - Check that groups are correctly identified
       - Verify group sizes look reasonable
       - Ensure you have at least 2 groups for analysis
    """)
    
    st.subheader("Step 3: Choose Analysis Method")
    st.markdown("""
    **Location:** 📊 Analysis tab
    
    The Analysis tab contains three sub-tabs for different analysis methods:
    
    **📊 Basic Analysis:**
    - Select metric columns to analyze
    - Works with any numeric metrics
    - Shows means, uplifts, SMD, and p-values
    - View results as Summary (table) or Visual Report
    
    **🔬 CUPED Analysis:**
    - Configure suffixes for pre/post columns (default: "_pre" and "_post")
    - Select metrics that have both pre and post versions
    - Automatically detects metrics with matching pre/post pairs
    - Shows CUPED-adjusted means, SMD, and p-values
    - View results as Summary (table) or Visual Report
    
    **📉 Difference-in-Differences:**
    - Configure suffixes for pre/post columns (default: "_pre" and "_post")
    - Select metrics that have both pre and post versions
    - Automatically detects metrics with matching pre/post pairs
    - Shows pre-period means, post-period means, % change, and DiD gap
    - View results as Summary (table) or Visual Report
    """)
    
    st.subheader("Step 4: Review Results")
    st.markdown("""
    **Location:** 📊 Analysis tab → Selected analysis sub-tab
    
    **Summary View:**
    - Text-based table showing:
      - **Basic Analysis:** Means per group, uplifts (%), SMD, p-values, significance
      - **CUPED Analysis:** CUPED-adjusted means, uplifts, CUPED SMD, CUPED p-values
      - **DiD Analysis:** Pre/post means, % change, DiD gap, significance
    - Download summary as CSV for further analysis
    
    **Visual Report:**
    - Interactive plots showing:
      - **Basic Analysis:** 4-column layout per metric (means, uplifts, SMD, p-values)
      - **CUPED Analysis:** 3-column layout per metric (CUPED means, SMD, p-values)
      - **DiD Analysis:** 4-column layout per metric (pre-period means, post-period means, % change, DiD gap heatmap)
    - Download plots as HTML for presentations
    
    **Download Results:**
    - Download summary tables as CSV
    - Download interactive plots as HTML
    - Save complete artifact (data + plots + logs + config)
    """)
    
    st.divider()
    
    # Tips and Best Practices
    st.header("💡 Tips & Best Practices")
    
    st.markdown("""
    **When to Use Each Analysis Method:**
    - **Basic Analysis:** Use for binary outcomes (e.g., customer_churned, new_customer), simple metrics, or when you don't have pre-experiment data
    - **CUPED Analysis:** Use when you have pre-experiment baseline data and want to reduce variance for more precise estimates. Particularly useful for continuous metrics.
    - **DiD Analysis:** Use when groups may have different baselines or when you want to account for pre-existing trends. Good for understanding relative changes.
    
    **Data Requirements:**
    - **Basic Analysis:** Requires only post-experiment metric columns
    - **CUPED/DiD:** Require both pre and post columns with matching suffixes (e.g., revenue_pre, revenue_post)
    - Column naming: Use consistent suffixes (default: "_pre" and "_post")
    - ID columns are automatically excluded
    
    **Metric Selection:**
    - Select all metrics that are relevant to your experiment
    - For CUPED/DiD, only metrics with both pre and post versions will be available
    - Binary metrics (0-1) work well with Basic Analysis
    - Continuous metrics benefit from CUPED adjustment
    
    **Interpreting Results:**
    - **Uplift:** Positive = treatment is better, negative = control is better
    - **P-value < 0.05:** Statistically significant difference (reject null hypothesis)
    - **SMD < 0.1:** Small standardized difference (good balance)
    - **SMD 0.1-0.25:** Moderate difference
    - **SMD > 0.25:** Large difference
    - **DiD Gap:** The treatment effect after accounting for pre-existing differences
    
    **CUPED Benefits:**
    - Reduces variance by using pre-experiment baseline
    - Can detect smaller effects with same sample size
    - Particularly effective when pre/post correlation is high
    - Improves statistical power
    
    **DiD Benefits:**
    - Accounts for pre-existing differences between groups
    - Controls for time trends that affect both groups
    - Provides causal interpretation under parallel trends assumption
    - Good for observational studies or when randomization wasn't perfect
    
    **Workflow Tips:**
    - Start with Basic Analysis for quick overview
    - Use CUPED if you have baseline data and want more precision
    - Use DiD if groups had different baselines or you want to control for trends
    - Compare results across methods for robustness
    - Download summaries and plots for documentation
    - Save artifacts for reproducibility
    """)
    
    st.divider()
    
    # Example Workflow
    st.header("🚀 Example Workflow")
    
    st.markdown("""
    **Common Setup Steps:**
    1. **Upload** your experiment results CSV file (e.g., ab_test_results.csv)
    2. Go to **Configuration** tab
    3. **Select group column** (e.g., "treatment_group")
    4. Review group sizes and verify groups are correct
    5. Go to **Analysis** tab
    
    **For Basic Analysis:**
    1. Select **Basic Analysis** sub-tab
    2. **Select metrics** to analyze (e.g., customer_churned, new_customer, revenue)
    3. Choose **View Mode:** Summary or Visual Report
    4. Review means, uplifts, SMD, and p-values
    5. Download summary or plots as needed
    
    **For CUPED Analysis:**
    1. Select **CUPED Analysis** sub-tab
    2. **Configure suffixes** (default: "_pre" and "_post")
    3. **Select metrics** with pre/post pairs (e.g., revenue, engagement_score)
    4. Choose **View Mode:** Summary or Visual Report
    5. Review CUPED-adjusted means, SMD, and p-values
    6. Compare with Basic Analysis to see variance reduction
    7. Download summary or plots as needed
    
    **For DiD Analysis:**
    1. Select **Difference-in-Differences** sub-tab
    2. **Configure suffixes** (default: "_pre" and "_post")
    3. **Select metrics** with pre/post pairs (e.g., revenue, engagement_score)
    4. Choose **View Mode:** Summary or Visual Report
    5. Review pre/post means, % change, and DiD gap
    6. Check DiD gap heatmap for treatment effects
    7. Download summary or plots as needed
    
    **Final Step:**
    - **Save artifact** for documentation and reproducibility
    """)
    
    st.info("💡 **Remember:** Different analysis methods can provide complementary insights. Use Basic Analysis for quick results, CUPED for precision, and DiD for causal interpretation.")
