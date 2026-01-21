
"""
UI components for power analysis page
"""
import streamlit as st
import numpy as np

from power_analysis.config import parse_statistics, get_default_stats_json


def render_configuration_section():
    """Render the configuration section and return parsed values"""
    st.header("⚙️ Configuration")
    
    col_set1, col_set2 = st.columns(2)
    
    # Statistics input
    with col_set1:
        st.subheader("Population Statistics")
        
        # Auto-fill from generated JSON if available
        import hashlib
        import uuid
        
        # Check if we have newly generated JSON
        has_new_json = False
        if 'generated_json' in st.session_state and st.session_state.generated_json:
            # Check if this is new JSON that hasn't been used yet
            json_hash = hashlib.md5(st.session_state.generated_json.encode()).hexdigest()
            if 'last_json_hash' not in st.session_state or st.session_state.last_json_hash != json_hash:
                has_new_json = True
                st.session_state.last_json_hash = json_hash
                # Generate a new UUID for the widget key to force a fresh widget
                st.session_state.stats_input_widget_id = str(uuid.uuid4())
                # Store the JSON value
                st.session_state.stats_input_value = st.session_state.generated_json
        
        # Initialize widget ID if not exists
        if 'stats_input_widget_id' not in st.session_state:
            st.session_state.stats_input_widget_id = str(uuid.uuid4())
        
        # Get the value to use - prefer stored value, then generated JSON, then default
        if 'stats_input_value' in st.session_state:
            default_value = st.session_state.stats_input_value
        elif 'generated_json' in st.session_state and st.session_state.generated_json:
            default_value = st.session_state.generated_json
        else:
            default_value = get_default_stats_json()
        
        # Use UUID-based key to ensure uniqueness
        widget_key = f"stats_input_{st.session_state.stats_input_widget_id}"
        
        stats_input = st.text_area(
            "Enter statistics as JSON dictionary (scenario -> metric -> {mean, std, ...}):",
            value=default_value,
            height=300,
            key=widget_key
        )
        
        # Store the current value in a separate key (not the widget key)
        st.session_state.stats_input_value = stats_input
        
        # Show info if auto-filled
        if has_new_json:
            st.info("💡 Statistics auto-filled from Scenarios Design page. You can edit manually if needed.")
        
        # Parse statistics using config module
        stats_valid, result = parse_statistics(stats_input)
        
        if stats_valid:
            scenarios = result
            st.success(f"✅ Valid statistics ({len(scenarios)} scenarios)")
        else:
            scenarios = {}
            error_msg = result if isinstance(result, str) else "Please check the JSON format and required fields"
            st.error(f"❌ Invalid statistics: {error_msg}")
    
    # Parameter ranges
    with col_set2:
        st.subheader("Plot Parameters")
        
        col_a, col_b = st.columns(2)
        with col_a:
            uplift_min = st.number_input("Min Uplift", value=0.005, format="%.4f", key="uplift_min")
            alpha_min = st.number_input("Min Alpha", value=0.1, format="%.3f", key="alpha_min")
            power_min = st.number_input("Min Power", value=0.6, format="%.2f", key="power_min")
        
        with col_b:
            uplift_max = st.number_input("Max Uplift", value=0.1, format="%.4f", key="uplift_max")
            alpha_max = st.number_input("Max Alpha", value=0.4, format="%.3f", key="alpha_max")
            power_max = st.number_input("Max Power", value=0.9, format="%.2f", key="power_max")
        
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            uplift_points = st.number_input("Uplift Points", value=19, min_value=10, max_value=100, key="uplift_points")
        with col_d:
            alpha_points = st.number_input("Alpha Points", value=16, min_value=10, max_value=100, key="alpha_points")
        with col_e:
            power_points = st.number_input("Power Points", value=16, min_value=10, max_value=100, key="power_points")
        
        st.divider()
        
        col_h, col_i = st.columns(2)
        with col_h:
            n_groups = st.number_input("Number of Groups in Experiment", value=2, min_value=1, max_value=10, key="n_groups")
            st.caption("For sample size allocation (e.g., control + treatment)")
        
        with col_i:
            ttest_type = st.selectbox(
                "T-Test Type",
                options=["two-sided", "larger", "smaller"],
                index=0,
                key="ttest_type",
                help="two-sided: H1: μ1 ≠ μ2 | larger: H1: μ1 > μ2 | smaller: H1: μ1 < μ2"
            )
            st.caption("Alternative hypothesis direction")
    
    return {
        'stats_valid': stats_valid,
        'scenarios': scenarios,
        'uplift_min': uplift_min,
        'uplift_max': uplift_max,
        'uplift_points': int(uplift_points),
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'alpha_points': int(alpha_points),
        'power_min': power_min,
        'power_max': power_max,
        'power_points': int(power_points),
        'n_groups': int(n_groups),
        'ttest_type': ttest_type
    }


def render_calculator_tab(stats_valid, power_analysis, groups):
    """Render the calculator tab"""
    st.header("🧮 Sample Size Calculator")
    st.markdown("Enter specific values for all parameters to calculate required sample sizes")
    
    if not stats_valid:
        st.warning("⚠️ Please enter valid statistics above first")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calc_uplift = st.number_input("Assumed Uplift", value=0.02, min_value=0.0001, max_value=1.0, format="%.4f", key="calc_uplift")
        st.caption("Expected relative change (e.g., 0.02 = 2%)")
    
    with col2:
        calc_alpha = st.number_input("Alpha (Significance Level)", value=0.05, min_value=0.01, max_value=0.5, format="%.3f", key="calc_alpha")
        st.caption("Probability of Type I error")
    
    with col3:
        calc_power = st.number_input("Power", value=0.8, min_value=0.5, max_value=0.99, format="%.2f", key="calc_power")
        st.caption("1 - Probability of Type II error")
    
    if st.button("🔢 Calculate Sample Sizes", type="primary", key="calc_button"):
        try:
            st.success("✅ Calculation complete!")
            
            # Calculate for each group
            results = {}
            for group_name, group_stats in groups.items():
                mean = group_stats['mean']
                std = group_stats['std']
                n = power_analysis.calculate_sample_size(
                    calc_uplift, calc_alpha, calc_power,
                    mean, std
                )
                results[group_name] = n
            
            # Display results
            cols = st.columns(len(results))
            for idx, (group_name, n) in enumerate(results.items()):
                with cols[idx]:
                    if not np.isnan(n):
                        st.metric(
                            f"📊 {group_name}",
                            f"{int(np.ceil(n)):,}",
                            help=f"Required sample size for {group_name}"
                        )
                        
                        es = (groups[group_name]['mean'] * calc_uplift) / groups[group_name]['std']
                        st.caption(f"Effect size: {es:.4f}")
                    else:
                        st.error(f"❌ {group_name}: Calculation failed")
            
            # Summary box
            st.info(f"""
            **Summary:**
            - With an assumed uplift of **{calc_uplift*100:.2f}%**
            - At significance level α = **{calc_alpha}**
            - With power (1-β) = **{calc_power}**
            """)
                
        except Exception as e:
            st.error(f"❌ Calculation error: {e}")


def render_instructions_tab():
    """Render the instructions tab"""
    st.header("ℹ️ How to Use This Tool")
    
    st.markdown("""
    ### Overview
    This tool helps you determine the required sample size for A/B testing experiments based on statistical power analysis.
    
    ### Getting Started
    
    #### 1. Configure Population Statistics (Top of Page)
    Enter your population statistics as a JSON dictionary with arbitrary metrics:
    ```json
    {
        "group_a": {"mean": 25.85, "std": 133.89},
        "group_b": {"mean": 18.99, "std": 173.37},
        "group_c": {"mean": 20.5, "std": 150.2}
    }
    ```
    
    - Each **key** represents a metric name (e.g., "group_a", "group_b", "revenue", "conversion_rate")
    - Each metric must have **mean** and **std** (standard deviation) fields
    - You can add as many metrics as needed
    
    #### 2. Set Plot Parameters
    Configure the ranges for uplift, alpha, and power that you want to explore
    
    #### 3. Compute All Plots
    Click the "Compute All Plots" button to pre-calculate all sample sizes
    - This will compute sample sizes for all parameter combinations
    - All plots (Single Plots and Contour Maps) will use this precomputed data
    
    #### 4. Use the Calculator Tab 🧮
    - Enter specific values for **Uplift**, **Alpha**, and **Power**
    - Click "Calculate Sample Sizes" to get exact sample size requirements
    - Perfect for quick calculations with known parameters
    - Works independently from the computed plots
    
    #### 5. Explore Single Plots Tab 📈
    - View how sample size changes with ONE parameter while keeping others fixed
    - Three plot types available:
      - **Sample Size vs Uplift**: See how expected effect size impacts sample size
      - **Sample Size vs Alpha**: Understand the trade-off with significance level
      - **Sample Size vs Power**: Explore the relationship with statistical power
    - Use sliders to adjust the fixed parameters
    - Uses precomputed data for instant updates
    
    #### 6. Analyze Contour Maps Tab 🗺️
    - Three different views showing sample size across TWO parameters simultaneously
    - Use sliders to explore the third dimension
    - Uses precomputed data for smooth interactions
    
    ### Key Concepts
    
    **Uplift (Effect Size)**
    - The expected relative change in your metric
    - Example: 0.02 = 2% increase
    - Smaller uplifts require larger sample sizes to detect
    
    **Alpha (Significance Level)**
    - Probability of Type I error (false positive)
    - Common values: 0.05 (5%) or 0.01 (1%)
    - Lower alpha requires larger sample sizes
    
    **Power (1 - β)**
    - Probability of detecting a true effect
    - Common values: 0.8 (80%) or 0.9 (90%)
    - Higher power requires larger sample sizes
    
    ### Tips
    
    - Start with the **Calculator** for quick estimates
    - Use **Compute All Plots** once to enable all visualizations
    - Use **Single Plots** to understand individual parameter effects
    - Use **Contour Maps** for comprehensive parameter space exploration
    - Darker colors in contour maps indicate larger required sample sizes
    - Different metrics may require different sample sizes based on their variability
    
    ### Example Workflow
    
    1. Enter your historical data statistics at the top
    2. Configure parameter ranges
    3. Click "Compute All Plots"
    4. Go to Calculator tab and try a typical uplift (e.g., 2%)
    5. Check Single Plots to see how varying each parameter affects sample size
    6. Explore Contour Maps to visualize the full parameter space
    7. Use insights to design your experiment with appropriate sample size
    """)

