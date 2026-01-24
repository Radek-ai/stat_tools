"""
How-To Guide for Power Analysis page
"""
import streamlit as st


def render_guide():
    """Render the Power Analysis how-to guide"""
    st.header("📚 How-To Guide")
    st.markdown("Complete guide for using the Power Analysis tool")
    
    st.divider()
    
    # Key Concepts
    st.header("🎯 Key Concepts")
    
    st.subheader("What is Power Analysis?")
    st.markdown("""
    Power analysis helps you determine the **required sample size** for your A/B test to achieve desired statistical power. 
    It answers the question: "How many participants do I need to reliably detect a treatment effect?"
    """)
    
    st.subheader("Key Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Uplift (Effect Size)**
        
        The expected relative change in your metric.
        
        - Example: 0.02 = 2% increase
        - Smaller uplifts require larger sample sizes to detect
        - Based on your business expectations or historical data
        """)
    
    with col2:
        st.markdown("""
        **🎯 Alpha (Significance Level)**
        
        Probability of Type I error (false positive).
        
        - Common values: 0.05 (5%) or 0.01 (1%)
        - Lower alpha = stricter = requires larger sample sizes
        - Industry standard is typically 0.05
        """)
    
    with col3:
        st.markdown("""
        **⚡ Power (1 - β)**
        
        Probability of detecting a true effect.
        
        - Common values: 0.8 (80%) or 0.9 (90%)
        - Higher power = more reliable = requires larger sample sizes
        - 80% power means 80% chance of detecting a real effect
        """)
    
    st.divider()
    
    # Step-by-Step Guide
    st.header("📋 Step-by-Step Guide")
    
    st.subheader("Step 1: Upload Data")
    st.markdown("""
    **Location:** 📤 Data Upload tab
    
    - Upload a CSV file with your historical data, OR
    - Click "🎲 Load Dummy Data" to use sample data for testing
    - The data should contain numeric columns representing metrics you want to analyze
    - ID columns (all unique values) are automatically excluded
    """)
    
    st.subheader("Step 2: Configure Scenarios")
    st.markdown("""
    **Location:** ⚙️ Configuration tab
    
    1. **Select Metrics:** Choose numeric columns to analyze (e.g., revenue, conversion_rate)
    2. **Add Scenarios:** Create scenarios with different outlier filtering methods:
       - **None:** Keep all data (no filtering)
       - **Percentile:** Remove values outside percentile range (e.g., 1st-99th percentile)
       - **Winsorize:** Clip extreme values instead of removing rows
       - **IQR:** Remove outliers using Interquartile Range method
    3. **Compute Statistics:** Click "🚀 Compute Statistics" to calculate mean and standard deviation for each scenario-metric combination
    
    **Why scenarios?** Different outlier handling methods can affect your statistics. Testing multiple scenarios helps you understand the sensitivity of your sample size requirements.
    """)
    
    st.subheader("Step 3: Set Analysis Parameters")
    st.markdown("""
    **Location:** 🔬 Power Analysis tab → Configuration section
    
    1. **Population Statistics:** 
       - Statistics are auto-filled from Step 2, or
       - Enter manually as JSON (scenario → metric → {mean, std})
    2. **Plot Parameters:** Set ranges for uplift, alpha, and power to explore
    3. **Experiment Settings:**
       - **Number of Groups:** Total number of groups in your experiment. This determines how the total sample size is divided. For example:
         - 2 groups = control + treatment (each gets 50% of total sample)
         - 3 groups = control + treatment A + treatment B (each gets ~33% of total sample)
       - **T-Test Type:** The direction of your alternative hypothesis:
         - **Two-sided:** Detects any difference (increase or decrease) - most common
         - **Larger:** Only detects increases (one-sided test)
         - **Smaller:** Only detects decreases (one-sided test)
    4. **Compute All Plots:** Click "🔍 Compute All Plots" to pre-calculate sample sizes for all parameter combinations
    """)
    
    st.subheader("Step 4: Explore Results")
    st.markdown("""
    **Location:** 🔬 Power Analysis tab → Results tabs
    
    **🧮 Calculator Tab:**
    - Enter specific values for Uplift, Alpha, and Power
    - Get exact sample size requirements instantly
    - Perfect for quick calculations with known parameters
    
    **📈 Single Plots Tab:**
    - View how sample size changes with ONE parameter (uplift, alpha, or power)
    - Use sliders to adjust the other two fixed parameters
    - Three plot types available for different perspectives
    - **📊 Threshold Line:** A red dashed horizontal line shows the maximum achievable sample size per group based on your actual data. This represents the best you can achieve using all available users in your dataset. If the required sample size curve is above this line, you may need more data or to adjust your parameters.
    
    **🗺️ Contour Maps Tab:**
    - Interactive 2D visualizations showing sample size across TWO parameters
    - Three different views (uplift+alpha, alpha+power, uplift+power)
    - Use sliders to explore the third dimension
    - Darker colors indicate larger required sample sizes
    - The threshold line concept also applies here, helping you identify feasible parameter combinations
    """)
    
    st.divider()
    
    # Tips and Best Practices
    st.header("💡 Tips & Best Practices")
    
    st.markdown("""
    **Workflow Tips:**
    - Use the **Calculator** for quick estimates with specific parameters
    - Use **Compute All Plots** once to enable all visualizations (may take a moment)
    - Use **Single Plots** to understand how each parameter individually affects sample size
    - Use **Contour Maps** for comprehensive exploration of the parameter space
    
    **Parameter Selection:**
    - **Uplift:** Base on business goals or historical effect sizes. Test multiple values (e.g., 1%, 2%, 5%)
    - **Alpha:** Typically 0.05. Use 0.01 for stricter requirements
    - **Power:** Typically 0.8 (80%). Use 0.9 (90%) for critical experiments
    
    **Multiple Metrics:**
    - Different metrics may require different sample sizes based on their variability
    - Your final sample size should accommodate the metric requiring the MOST data
    - Review all metrics in the Calculator to find the limiting factor
    
    **Scenarios:**
    - Test multiple outlier filtering methods to understand sensitivity
    - Percentile filtering removes data but may improve balance
    - Winsorization preserves sample size while reducing extreme values
    """)
    
    st.divider()
    
    # Example Workflow
    st.header("🚀 Example Workflow")
    
    st.markdown("""
    1. **Upload** your historical data CSV file
    2. **Select metrics** you want to analyze (e.g., revenue, conversion_rate)
    3. **Add scenarios:**
       - Scenario 1: No filtering
       - Scenario 2: Percentile filter (1st-99th percentile)
    4. **Compute statistics** for all scenario-metric combinations
    5. Go to **Power Analysis** tab
    6. Statistics are auto-filled - verify they look correct
    7. Set parameter ranges (e.g., Uplift: 0.01-0.05, Alpha: 0.01-0.1, Power: 0.7-0.9)
    8. Click **"Compute All Plots"**
    9. Use **Calculator** to find sample size for specific values (e.g., 2% uplift, 0.05 alpha, 0.8 power)
    10. Explore **Single Plots** to see how varying each parameter affects sample size
    11. Check **Contour Maps** to visualize the full parameter space
    12. Use insights to design your experiment with appropriate sample size
    """)
    
