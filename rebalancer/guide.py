"""
How-To Guide for Rebalancer page
"""
import streamlit as st


def render_guide():
    """Render the Rebalancer how-to guide"""
    st.header("📚 How-To Guide")
    st.markdown("Complete guide for using the Group Rebalancer tool")
    
    st.divider()
    
    # Key Concepts
    st.header("🎯 Key Concepts")
    
    st.subheader("What is Group Rebalancing?")
    st.markdown("""
    Group Rebalancing helps you **improve balance** in data that already has group assignments by **trimming rows** (removing participants) 
    rather than reassigning them. This is useful when you have existing experimental groups that are imbalanced and you want to improve 
    balance without changing group assignments.
    
    **Key Difference from Group Selection:** 
    - **Group Selection:** Creates balanced groups from scratch by assigning participants to groups
    - **Rebalancer:** Improves balance in existing groups by removing participants (trimming)
    
    **When to Use:** After an experiment has been run and you discover groups are imbalanced.
    """)
    
    st.subheader("Key Concepts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✂️ Trimming (Row Removal)**
        
        The rebalancer improves balance by removing rows from groups, not by reassigning them.
        
        - Preserves existing group assignments
        - Removes participants that contribute most to imbalance
        - Reduces sample size but improves statistical validity
        - Useful when reassignment is not possible
        """)
    
    with col2:
        st.markdown("""
        **📊 Even-Size Seed Search**
        
        A technique that subsamples all groups to the smallest group's size to find optimal starting points.
        
        - Tests multiple random seeds to find best initial subsample
        - Ensures all groups start at equal size
        - Helps find better local optima
        """)
    
    with col3:
        st.markdown("""
        **⚖️ Iterative Rebalancing**
        
        Advanced optimization that iteratively removes rows to minimize statistical differences.
        
        - Removes rows that contribute most to imbalance
        - Uses loss function to guide removal decisions
        - Can continue from previous runs to improve further
        - Slow, but removes less data
        """)
    
    st.divider()
    
    # Step-by-Step Guide
    st.header("📋 Step-by-Step Guide")
    
    st.subheader("Step 1: Upload Data with Existing Groups")
    st.markdown("""
    **Location:** 📤 Data Upload tab
    
    - Upload a CSV file that **already contains group assignments** in one of the columns
    - Or click "🎲 Load Dummy Data" to use sample data for testing
    - Uploaded data should have a column indicating which group each participant belongs to (e.g., "group", "treatment", "variant")
    - Uploaded data should contain:
      - **Group column:** Column with existing group assignments
      - **Numeric columns:** Metrics to balance (e.g., age, revenue, engagement_score)
      - **Categorical columns:** Variables for stratification (e.g., region, device_type)
    """)
    
    st.subheader("Step 2: Configure Group Column & Filters")
    st.markdown("""
    **Location:** ⚙️ Configuration tab
    
    1. **Select Group Column:**
       - Choose the column containing your existing group assignments
       - The tool will detect all unique groups in this column
       - Group sizes and statistics will be displayed
    
    2. **Data Filtering (Optional):**
       - **Outlier Filtering:** Remove extreme values using percentile, IQR, or winsorization methods
       - **Value-Based Filtering:** Filter numeric columns by min/max ranges
       - **Categorical Filtering:** Include/exclude specific categories
       - Filters help remove problematic data before rebalancing
       - Click "✅ Apply Filters" to use filtered data, or "🔄 Reset Filters" to start over
    
    3. **Review Balance Reports:**
       - **Initial Balance Report:** Shows current balance before rebalancing (in expander)
       - **Filtered Data Balance Report:** Shows balance after filtering (if filters applied, in expander)
       - Use these to understand current imbalance and set appropriate targets
    """)
    
    st.subheader("Step 3: Configure Rebalancing")
    st.markdown("""
    **Location:** ⚖️ Rebalancing tab
    
    **3.1 Select Columns to Balance:**
    - **Numeric Columns:** Select columns to balance using statistical tests (t-tests, p-values, SMD)
    - **Categorical Columns:** Select columns for stratification balance
    - These columns will be used to calculate balance and guide row removal
    
    **3.2 Set Balancing Objectives:**
    - **Target P-values:** For each numeric column, set target p-value (higher values = stricter balance requirements)
    - **Max Imbalance %:** For each categorical column, set maximum acceptable imbalance percentage
    - These targets guide the rebalancing algorithm
    
    **3.3 Choose Rebalancing Mode:**
    
    **Basic Mode:**
    - Even-size seed search only
    - Subsamples all groups to smallest group's size
    - Tests multiple random seeds to find best subsample
    - Fast and simple, good for quick rebalancing
    - Settings: Enable seed search, number of seed trials
    
    **Advanced Mode:**
    - Iterative rebalancing with intelligent row removal
    - Uses middle/odd group strategy for multi-group scenarios
    - Slower but performs better and removes less rows
    - Can continue from previous runs
    - Settings include:
      - **Max Removals:** Maximum rows to remove per group
      - **Top K Candidates:** Number of candidates preselected by z-score to consider for trimming
      - **Random Candidates:** Number of random candidates to consider
      - **Gain Threshold:** Minimum improvement required to remove a row
      - **Early Break:** Stop searching once a good move is found
      - **Even Size Seed Search:** Optional initial step to subsample to equal sizes
    
    **3.4 Run Rebalancing:**
    - Click "⚖️ Start Rebalancing"
    - Advanced mode shows progress and loss convergence
    - Results show how many rows were removed and balance improvements
    """)
    
    st.subheader("Step 4: Review Results")
    st.markdown("""
    **Location:** ⚖️ Rebalancing tab → Results section
    
    **Group Size Changes:**
    - Original vs. rebalanced group sizes
    - Shows how many rows were removed from each group
    - Displays size differences and statistics
    
    **Loss History (Advanced Mode):**
    - Plot showing how balance improves over iterations
    - Lower loss = better balance
    - Can show multiple runs if "Continue Rebalancing" was used
    
    **Balance Evaluation:**
    - **Summary View:** Text-based table showing:
      - P-values and SMD for numeric columns (pairwise comparisons)
      - Imbalance percentages for categorical columns
      - Mean values per group for numeric columns
      - Comparison of before vs. after rebalancing
    - **Visual Report:** Interactive heatmap showing:
      - P-values (higher = better balance, green = good)
      - SMD values (lower = better balance, green = good)
      - Categorical imbalance percentages
      - Color-coded for quick assessment
    
    **Download Results:**
    - Download rebalanced data as CSV
    - Download balance report plots
    - Save complete artifact (data + plots + logs + config)
    """)
    
    st.divider()
    
    # Tips and Best Practices
    st.header("💡 Tips & Best Practices")
    
    st.markdown("""
    **When to Use Basic vs Advanced Mode:**
    - **Basic Mode:** Quick rebalancing when perfect balance is not neccesary
    - **Advanced Mode:** When groups must be well balanced and removing as few rows as possible is crucial
    
    **Column Selection:**
    - Include all variables that affect your outcome metric
    - Balance on confounders (variables that correlate with both treatment and outcome)
    - More columns = more thorough balancing but may remove more rows
    
    **Filtering:**
    - Remove extreme outliers that could skew rebalancing decisions
    - Filter out invalid or missing data before rebalancing
    - Consider the impact of filtering on your final sample size
    
    **Balancing Objectives:**
    - **Target P-values:** 
      - Positive values (e.g., 0.95): Algorithm tries to maximize p-value (better balance)
      - Higher positive values indicate stricter balance requirements
    - **Max Imbalance %:** Lower values (e.g., 5%) are stricter for categorical variables
    
    **Advanced Mode Settings:**
    - **Max Removals:** Set based on how much data you can afford to lose. Start conservative (e.g., 100 rows per group)
    - **Top K Candidates:** Higher values consider more z-score based candidates but are slower. Typical: 10-50
    - **Random Candidates:** Adds randomness to prevent getting stuck and overfitting. Typical: 100-500
    - **Gain Threshold:** Lower values allow smaller improvements. Typical: 0.0001-0.001
    - **Even Size Seed Search:** Useful when groups have very different sizes. Tests multiple seeds to find best starting point
    
    **Understanding Results:**
    - Check how many rows were removed - more removals may mean better balance but smaller sample
    - Review balance metrics - p-values should increase, SMD should decrease
    - Compare before/after balance reports to see improvements
    - Consider whether the improved balance is worth the reduced sample size
    
    **Workflow Tips:**
    - Use Advanced Mode if balance is crucial
    - Use "Continue Rebalancing" to iteratively improve results
    - Review initial balance report to set realistic targets
    - Save artifacts for documentation and reproducibility
    """)
    
    st.divider()
    
    # Example Workflow
    st.header("🚀 Example Workflow")
    
    st.markdown("""
    1. **Upload** your experimental data CSV file with existing group assignments (e.g., experiment_results.csv)
    2. Go to **Configuration** tab
    3. **Select group column** (e.g., "treatment_group")
    4. Review group sizes and initial balance report
    5. **Apply filters** (optional):
       - Remove outliers using percentile filter
       - Filter out invalid entries
       - Review filtered data balance report
    6. Go to **Rebalancing** tab
    7. **Select columns to balance:**
       - Numeric: age, revenue, engagement_score
       - Categorical: region, device_type
    8. **Set balancing objectives:**
       - Target p-values: 0.95 for all numeric columns
       - Max imbalance: 5% for categorical columns
    9. **Choose mode:**
       - Start with **Basic Mode** for quick equal-size rebalancing checks
       - Or use **Advanced Mode** for better balance
    10. **Set Advanced Mode settings** (if using):
       - Max Removals: 100 per group
       - Top K Candidates: 10
       - Random Candidates: 200
       - Gain Threshold: 0.0001
       - Enable Even Size Seed Search: Yes (1000 trials)
    11. Click **"⚖️ Start Rebalancing"**
    12. **Review results:**
       - Check how many rows were removed from each group
       - Review balance evaluation (p-values, SMD, imbalance %)
       - Compare before/after balance reports
       - Check loss history plot (Advanced mode)
    13. If balance needs improvement:
       - Enable "Continue Rebalancing" checkbox
       - Click "⚖️ Start Rebalancing" again to continue optimizing
    14. **Download rebalanced data** for your analysis
    15. **Save artifact** for documentation and reproducibility
    """)
    
    st.info("💡 **Remember:** Rebalancing improves balance by removing data. Consider whether the improved balance is worth the reduced sample size for your analysis.")
