"""
Scenarios Design Page - CSV upload and scenario/metric configuration
"""
import streamlit as st
import pandas as pd
import numpy as np
import json

from power_analysis.data_processing import compute_all_scenarios


def render_scenarios_design_page():
    """Render the scenarios design page"""
    st.header("📊 Scenarios Design")
    st.markdown("Upload CSV data and configure scenarios with outlier filtering and metrics")
    
    # Initialize session state
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'scenarios_config' not in st.session_state:
        st.session_state.scenarios_config = []
    if 'computed_stats' not in st.session_state:
        st.session_state.computed_stats = None
    
    # Section 1: CSV Upload
    st.subheader("📁 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df
        
        st.success(f"✅ File uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
        
        # Show preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total rows: {len(df)}, Columns: {', '.join(df.columns.tolist())}")
        
        # Show numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.info(f"📊 Numeric columns detected: {', '.join(numeric_cols)}")
        else:
            st.warning("⚠️ No numeric columns found in the dataset")
    
    st.divider()
    
    # Section 2: Metrics Selection (shared across all scenarios)
    st.subheader("📊 Select Metrics")
    st.markdown("Select the numeric columns to use as metrics. These will be applied to all scenarios.")
    
    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a CSV file first")
    else:
        numeric_cols = st.session_state.uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ No numeric columns available")
        else:
            # Initialize selected metrics in session state
            if 'selected_metrics' not in st.session_state:
                st.session_state.selected_metrics = []
            
            # Multi-select for metrics
            selected_metrics = st.multiselect(
                "Select Metrics (columns to analyze)",
                numeric_cols,
                default=st.session_state.selected_metrics,
                key="metrics_selector"
            )
            st.session_state.selected_metrics = selected_metrics
            
            if selected_metrics:
                st.success(f"✅ {len(selected_metrics)} metric(s) selected: {', '.join(selected_metrics)}")
            else:
                st.info("💡 Select at least one metric to continue")
    
    st.divider()
    
    # Section 3: Scenarios Configuration
    st.subheader("⚙️ Configure Scenarios")
    st.markdown("Add scenarios with different outlier filtering methods. All scenarios will use the same metrics selected above.")
    
    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a CSV file first")
    elif not st.session_state.selected_metrics:
        st.warning("⚠️ Please select at least one metric first")
    else:
        # Add new scenario button
        if st.button("➕ Add New Scenario", key="add_new_scenario"):
            new_scenario = {
                'method': 'none',
                'p_low': 1.0,
                'p_high': 99.0,
                'iqr_multiplier': 1.5
            }
            if 'scenarios_config' not in st.session_state:
                st.session_state.scenarios_config = []
            st.session_state.scenarios_config.append(new_scenario)
            st.rerun()
        
        # Display existing scenarios as expanders
        if st.session_state.scenarios_config:
            for idx, scenario in enumerate(st.session_state.scenarios_config):
                # Generate scenario name for display
                if scenario['method'] == 'none':
                    scenario_display_name = "No Outlier Filtering"
                elif scenario['method'] == 'percentile':
                    scenario_display_name = f"Percentile Filter ({scenario['p_low']}-{scenario['p_high']})"
                elif scenario['method'] == 'winsorize':
                    scenario_display_name = f"Winsorize ({scenario['p_low']}-{scenario['p_high']})"
                elif scenario['method'] == 'iqr':
                    scenario_display_name = f"IQR Filter (multiplier: {scenario['iqr_multiplier']})"
                else:
                    scenario_display_name = f"Scenario {idx + 1}"
                
                with st.expander(f"📊 {scenario_display_name}", expanded=True):
                    # Scenario configuration
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        outlier_method = st.selectbox(
                            "Outlier Method",
                            ["none", "percentile", "winsorize", "iqr"],
                            index=["none", "percentile", "winsorize", "iqr"].index(scenario['method']),
                            key=f"scenario_{idx}_method"
                        )
                        scenario['method'] = outlier_method
                    
                    with col2:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                        if st.button("🗑️ Remove Scenario", key=f"remove_scenario_{idx}"):
                            st.session_state.scenarios_config.pop(idx)
                            st.rerun()
                    
                    # Method-specific parameters
                    if outlier_method in ['percentile', 'winsorize']:
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            p_low = st.number_input(
                                "Lower Percentile",
                                min_value=0.0,
                                max_value=100.0,
                                value=scenario.get('p_low', 1.0),
                                step=0.1,
                                format="%.1f",
                                key=f"scenario_{idx}_p_low"
                            )
                            scenario['p_low'] = p_low
                        
                        with col_p2:
                            p_high = st.number_input(
                                "Upper Percentile",
                                min_value=0.0,
                                max_value=100.0,
                                value=scenario.get('p_high', 99.0),
                                step=0.1,
                                format="%.1f",
                                key=f"scenario_{idx}_p_high"
                            )
                            scenario['p_high'] = p_high
                    
                    elif outlier_method == 'iqr':
                        iqr_multiplier = st.number_input(
                            "IQR Multiplier",
                            min_value=0.1,
                            max_value=5.0,
                            value=scenario.get('iqr_multiplier', 1.5),
                            step=0.1,
                            format="%.1f",
                            key=f"scenario_{idx}_iqr_multiplier"
                        )
                        scenario['iqr_multiplier'] = iqr_multiplier
    
    st.divider()
    
    # Section 4: Compute Statistics
    st.subheader("🔢 Compute Statistics")
    
    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a CSV file first")
    elif not st.session_state.selected_metrics:
        st.warning("⚠️ Please select at least one metric")
    elif not st.session_state.scenarios_config:
        st.warning("⚠️ Please add at least one scenario")
    else:
        if st.button("🚀 Compute Statistics", type="primary", use_container_width=True):
            # Prepare scenarios config with shared metrics
            # Convert selected metrics to the format expected by compute_all_scenarios
            scenarios_config_with_metrics = []
            for scenario in st.session_state.scenarios_config:
                scenario_copy = scenario.copy()
                # Add metrics to each scenario (all scenarios share the same metrics)
                scenario_copy['metrics'] = [
                    {'name': col, 'column': col} for col in st.session_state.selected_metrics
                ]
                scenarios_config_with_metrics.append(scenario_copy)
            
            with st.spinner("Computing statistics for all scenario-metric combinations..."):
                computed_stats = compute_all_scenarios(
                    st.session_state.uploaded_df,
                    scenarios_config_with_metrics
                )
                st.session_state.computed_stats = computed_stats
                st.success("✅ Statistics computed successfully!")
        
        # Display computed statistics and JSON
        if st.session_state.computed_stats:
            st.markdown("**Generated JSON (copy this if needed):**")
            json_output = json.dumps(st.session_state.computed_stats, indent=2)
            st.code(json_output, language='json')
            
            # Store in session state for auto-fill on Power Analysis page
            st.session_state.generated_json = json_output
            
            st.success("💡 Statistics are ready! Go to the Power Analysis tab to use them.")
