"""
Scenarios Design Page - CSV upload and scenario/metric configuration
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from power_analysis.data_processing import compute_all_scenarios
from utils.artifact_builder import ArtifactBuilder
from utils.streamlit_upload import render_csv_upload_with_dummy


def render_data_upload():
    """Render the data upload section"""
    # Initialize artifact builder
    if "power_analysis_artifact" not in st.session_state:
        st.session_state.power_analysis_artifact = ArtifactBuilder(page_name="power_analysis")

    render_csv_upload_with_dummy(
        header="📤 Upload Data",
        description="Upload CSV data for power analysis",
        data_state_key="uploaded_df",
        filename_state_key=None,
        uploader_label="Or upload CSV file",
        uploader_key="csv_uploader",
        uploader_help=None,
        dummy_file_path=os.path.join("dummy_data", "power_analysis_dummy.csv"),
        dummy_button_key="power_load_dummy",
        dummy_loaded_filename="power_analysis_dummy.csv",
        artifact=st.session_state.power_analysis_artifact,
        artifact_df_name="uploaded_data",
        artifact_df_description="Original uploaded data",
        artifact_log_category="data_upload",
        show_overview=True,
    )


def render_configuration_page():
    """Render the configuration section"""
    # Initialize session state
    if 'scenarios_config' not in st.session_state:
        st.session_state.scenarios_config = []
    if 'computed_stats' not in st.session_state:
        st.session_state.computed_stats = None
    
    st.header("⚙️ Configuration")
    st.markdown("Configure metrics and scenarios with outlier filtering")
    
    if st.session_state.get('uploaded_df') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    # Section 1: Metrics Selection (shared across all scenarios)
    st.subheader("📊 Select Metrics")
    st.markdown("Select the numeric columns to use as metrics. These will be applied to all scenarios.")
    
    from utils.data_filtering import is_id_column
    
    numeric_cols = st.session_state.uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out ID columns
    numeric_cols = [col for col in numeric_cols if not is_id_column(st.session_state.uploaded_df, col)]
    
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
    
    # Section 2: Scenarios Configuration
    st.subheader("⚙️ Configure Scenarios")
    st.markdown("Add scenarios with different outlier filtering methods. All scenarios will use the same metrics selected above.")
    
    if not st.session_state.selected_metrics:
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
    
    # Section 3: Compute Statistics
    st.subheader("🔢 Compute Statistics")
    
    if not st.session_state.selected_metrics:
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
                
                # Add computed statistics to artifact
                artifact = st.session_state.get('power_analysis_artifact')
                if artifact:
                    artifact.set_config({
                        'computed_stats': computed_stats,
                        'selected_metrics': st.session_state.selected_metrics,
                        'scenarios_config': st.session_state.scenarios_config
                    })
                    artifact.add_log(
                        category='computation',
                        message='Statistics computed for all scenario-metric combinations',
                        details={
                            'n_scenarios': len(st.session_state.scenarios_config),
                            'n_metrics': len(st.session_state.selected_metrics),
                            'total_combinations': len(st.session_state.scenarios_config) * len(st.session_state.selected_metrics)
                        }
                    )
                
                st.success("✅ Statistics computed successfully!")
        
        # Display computed statistics and JSON
        if st.session_state.computed_stats:
            st.markdown("**Generated JSON (copy this if needed):**")
            json_output = json.dumps(st.session_state.computed_stats, indent=2)
            st.code(json_output, language='json')
            
            # Store in session state for auto-fill on Power Analysis page
            st.session_state.generated_json = json_output
            
            st.success("💡 Statistics are ready! Go to the 'Power Analysis' tab to use them.")


def render_scenarios_design_page():
    """Legacy function - kept for backward compatibility, redirects to upload"""
    render_data_upload()
