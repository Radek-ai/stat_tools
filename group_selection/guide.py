"""
How-To Guide for Group Selection page
"""
import streamlit as st


def render_guide():
    """Render the Group Selection how-to guide"""
    st.header("📚 How-To Guide")
    st.markdown("Complete guide for using the Group Selection tool")
    
    st.divider()
    
    # Key Concepts
    st.header("🎯 Key Concepts")
    
    st.subheader("What is Group Selection & Balancing?")
    st.markdown("""
    Group Selection helps you create **balanced groups** for A/B testing by assigning participants to control and treatment groups 
    in a way that minimizes differences between groups before the experiment begins.
    
    **Why is this important?** Balanced groups ensure that any differences you observe after the experiment are due to the treatment, 
    not pre-existing differences between groups. This increases the reliability and validity of your A/B test results.
    """)
    
    st.subheader("Key Concepts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Stratification**
        
        Dividing your data into similar subgroups (strata) based on numeric or categorical variables, then assigning participants from each stratum to groups.
        
        - Ensures groups are balanced on important characteristics
        - Reduces variance and improves statistical power
        - Example: Balance by age groups, regions, or customer segments
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Iterative Balancing**
        
        Advanced optimization that moves or swaps participants between groups to minimize statistical differences.
        
        - Uses algorithms to find optimal group assignments
        - Minimizes p-values (for numeric) and imbalance (for categorical)
        - Can continue from previous runs to improve balance further
        """)
    
    with col3:
        st.markdown("""
        **📈 Balance Metrics**
        
        Statistical measures to evaluate how well groups are balanced:
        
        - **P-value:** Higher p-values indicate groups are statistically similar
        - **SMD (Standardized Mean Difference):** Lower values indicate good balance
        - **Imbalance %:** For categorical variables, lower percentages indicate better balance
        """)
    
    st.divider()
    
    # Step-by-Step Guide
    st.header("📋 Step-by-Step Guide")
    
    st.subheader("Step 1: Upload Data")
    st.markdown("""
    **Location:** 📤 Data Upload tab
    
    - Upload a CSV file with your participant data, OR
    - Click "🎲 Load Dummy Data" to use sample data for testing
    - The data should contain:
      - **Numeric columns:** Metrics you want to balance (e.g., age, revenue, engagement_score)
      - **Categorical columns:** Variables for stratification (e.g., region, customer_type, device)
      - **ID column:** Unique identifier for each participant (automatically detected)
    """)
    
    st.subheader("Step 2: Configure Filters & Group Settings")
    st.markdown("""
    **Location:** ⚙️ Configuration tab
    
    1. **Data Filtering (Optional):**
       - **Outlier Filtering:** Remove extreme values using percentile, IQR, or winsorization methods
       - **Value-Based Filtering:** Filter numeric columns by min/max ranges
       - **Categorical Filtering:** Include/exclude specific categories
       - Filters help remove problematic data that could affect balancing quality
       - Click "✅ Apply Filters" to use filtered data, or "🔄 Reset Filters" to start over
    
    2. **Review Filtered Data:**
       - Check the summary showing original vs. filtered row counts
       - Preview the filtered data to ensure it looks correct
       - Filtered data will be used for group balancing
    
    **Note:** If no filters are applied, the original uploaded data will be used for balancing.
    """)
    
    st.subheader("Step 3: Group Balancing")
    st.markdown("""
    **Location:** ⚖️ Group Balancing tab
    
    **3.1 Group Settings:**
    - **Number of Groups:** How many groups to create (e.g., 2 = control + treatment)
    - **Group Column Name:** Name for the column storing group assignments (default: "group")
    - **Group Names & Proportions:** Name each group and set their relative sizes
      - Proportions automatically normalize to sum to 1.0
      - Example: control (0.5) + treatment (0.5) for equal split
    
    **3.2 Column Selection:**
    - **Numeric Columns:** Select columns to balance (e.g., revenue, age, engagement)
      - These will be balanced using statistical tests (t-tests, p-values, SMD)
    - **Categorical Columns:** Select columns for stratification (e.g., region, device_type)
      - These will be balanced by ensuring similar distributions across groups
    
    **3.3 Selection Mode:**
    
    **Basic Mode:**
    - Creates stratified initial groups only (no iterative optimization)
    - Fast and simple, good for quick group assignments
    - Uses binning for numeric columns and stratification for categorical columns
    - Settings: Number of bins for numeric columns, random seed for reproducibility
    
    **Advanced Mode:**
    - Creates stratified initial groups, then iteratively optimizes them
    - More thorough balancing, better for critical experiments
    - Can continue from previous runs to improve balance further
    - Settings include:
      - **Balancing Objectives:** Target p-values for numeric columns, max imbalance % for categorical columns
      - **Algorithm:** Sequential Moves (move rows between groups) or Swaps (swap rows between groups)
      - **Batch Mode:** Move/swap groups of rows at once to reduce overfitting
      - **Iteration Settings:** Max iterations, gain threshold, early break options
    
    **3.4 Run Balancing:**
    - Click "🚀 Create Groups" (Basic) or "🚀 Run Balancing" (Advanced)
    - Advanced mode shows progress and loss convergence
    - Results include group assignments and balance evaluation
    """)
    
    st.subheader("Step 4: Review Results")
    st.markdown("""
    **Location:** ⚖️ Group Balancing tab → Results section
    
    **Group Assignment Summary:**
    - Group sizes and proportions
    - Original vs. assigned row counts
    - Final loss value (Advanced mode only)
    
    **Loss History (Advanced Mode):**
    - Plot showing how balance improves over iterations
    - Lower loss = better balance
    - Can show multiple runs if "Continue Balancing" was used
    
    **Balance Evaluation:**
    - **Summary View:** Text-based table showing:
      - P-values and SMD for numeric columns (pairwise comparisons)
      - Imbalance percentages for categorical columns
      - Mean values per group for numeric columns
    - **Visual Report:** Interactive heatmap showing:
      - P-values (higher = better balance, green = good)
      - SMD values (lower = better balance, green = good)
      - Categorical imbalance percentages
      - Color-coded for quick assessment
    
    **Download Results:**
    - Download balanced data as CSV
    - Download balance report plots
    - Save complete artifact (data + plots + logs + config)
    """)
    
    st.divider()
    
    # Tips and Best Practices
    st.header("💡 Tips & Best Practices")
    
    st.markdown("""
    **When to Use Basic vs Advanced Mode:**
    - **Basic Mode:** Quick group assignments, exploratory analysis, when balance requirements are not critical
    - **Advanced Mode:** Production experiments, when balance is critical, when you need to optimize further
    
    **Column Selection:**
    - Include all variables that could affect your outcome metric
    - Balance on confounders (variables that correlate with both treatment and outcome)
    - More columns = more thorough balancing but may take longer
    
    **Filtering:**
    - Remove extreme outliers that could skew group assignments
    - Filter out invalid or missing data before balancing
    - Consider the impact of filtering on your final sample size
    
    **Advanced Mode Settings:**
    - **Target P-values:** Higher targets (e.g., 0.95) are stricter and may take longer
    - **Algorithm Choice:** 
      - Sequential Moves: Faster, preserves groups sizes
      - Swaps: Slower, can balance cases where Sequential fails
    - **Batch Mode:** Reduces overfitting by making larger changes, recommended for large datasets
    - **Continue Balancing:** Use if initial run didn't achieve desired balance - continues from current state
    
    **Workflow Tips:**
    - Start with Basic Mode to get initial groups quickly
    - Switch to Advanced Mode if balance needs improvement
    - Use "Continue Balancing" to iteratively improve results
    - Review balance report carefully before proceeding with experiment
    - Save artifacts for reproducibility and documentation
    """)
    
    st.divider()
    
    # Example Workflow
    st.header("🚀 Example Workflow")
    
    st.markdown("""
    1. **Upload** your participant data CSV file (e.g., customer_data.csv)
    2. Go to **Configuration** tab
    3. **Apply filters** (optional):
       - Remove outliers using percentile filter (1st-99th percentile)
       - Filter out invalid entries
       - Review filtered data summary
    4. Go to **Group Balancing** tab
    5. **Configure groups:**
       - Number of Groups: 2
       - Group names: "control", "treatment"
       - Proportions: 0.5, 0.5 (equal split)
    6. **Select columns:**
       - Numeric: age, revenue, engagement_score
       - Categorical: region, device_type
    7. **Choose mode:**
       - Start with **Basic Mode** for quick initial groups
       - Or use **Advanced Mode** for better balance
    8. **Set Advanced Mode settings** (if using):
       - Target p-values: 0.95 for all numeric columns
       - Max imbalance: 5% for categorical columns
       - Algorithm: Sequential Moves
       - Batch Mode: Enabled
       - Max Iterations: 50
    9. Click **"🚀 Run Balancing"**
    10. **Review results:**
       - Check group sizes match your proportions
       - Review balance evaluation (p-values, SMD, imbalance %)
       - Check loss history plot (Advanced mode)
    11. If balance needs improvement:
       - Enable "Continue Balancing" checkbox
       - Click "🚀 Run Balancing" again to continue optimizing
    12. **Download balanced data** and use it for your A/B test
    13. **Save artifact** for documentation and reproducibility
    """)
    
    st.info("💡 **Remember:** Balanced groups are the foundation of reliable A/B testing. Take time to ensure good balance before running your experiment.")
