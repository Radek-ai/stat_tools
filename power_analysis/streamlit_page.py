"""
Power Analysis Page - Statistical Power Analysis Tool
"""
import streamlit as st
import numpy as np

from power_analysis.power_analysis import PowerAnalysis
from power_analysis.plots import (
    create_single_plot_uplift,
    create_single_plot_alpha,
    create_single_plot_power,
    create_contour_plot_power,
    create_contour_plot_uplift,
    create_contour_plot_alpha
)
from power_analysis.ui_components import (
    render_configuration_section,
    render_calculator_tab,
    render_instructions_tab
)
from power_analysis.computation import compute_with_progress, get_max_sample_size
from power_analysis.components import (
    render_plot_with_download,
    render_scenario_selector,
    render_validation_message
)
from power_analysis.scenarios_design import render_data_upload, render_configuration_page
from power_analysis.guide import render_guide
from utils.streamlit_artifacts import render_download_artifact_button as _render_download_artifact_button


def render_download_artifact_button():
    """Render the download artifact button"""
    _render_download_artifact_button(
        page_name="power_analysis",
        state_key="power_analysis_artifact",
        file_name="artifact_power_analysis.zip",
        help_disabled="Upload data and perform analysis to create an artifact",
    )


def show_power_analysis_page():
    # Page title
    st.title("📊 Statistical Power Analysis Tool")
    st.markdown("Analyze required sample sizes for A/B testing based on various statistical parameters")

    # Create main tabs for subpages
    tab_upload, tab_config, tab_analysis, tab_instructions = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "🔬 Power Analysis",
        "📚 How-To Guide"
    ])

    # Tab 1: Data Upload
    with tab_upload:
        render_data_upload()
    
    # Tab 2: Configuration
    with tab_config:
        render_configuration_page()
    
    # Tab 3: How-To Guide
    with tab_instructions:
        render_guide()

    # Tab 4: Power Analysis
    with tab_analysis:
        # Configuration section
        config = render_configuration_section()
        
        # Compute button
        compute_button = st.button("🔍 Compute All Plots", type="primary", use_container_width=True)
        
        st.divider()
        
        # Initialize power analysis with t-test type
        power_analysis = PowerAnalysis(alternative=config.get('ttest_type', 'two-sided'))
        
        # Initialize or update session state
        if 'computed_data' not in st.session_state:
            st.session_state.computed_data = None
        
        # Compute if button clicked
        if compute_button and config['stats_valid']:
            # Reinitialize power analysis with current t-test type
            power_analysis = PowerAnalysis(alternative=config['ttest_type'])
            
            # Compute all scenarios with progress
            computed_data = compute_with_progress(config)
            
            # Store in session state
            st.session_state.computed_data = computed_data
            
            # Ensure artifact is initialized
            if "power_analysis_artifact" not in st.session_state:
                from utils.artifact_builder import ArtifactBuilder
                st.session_state.power_analysis_artifact = ArtifactBuilder(page_name="power_analysis")
            
            artifact = st.session_state.power_analysis_artifact
            if artifact:
                artifact.set_config({
                    'ttest_type': config['ttest_type'],
                    'n_groups': config.get('n_groups', 2),
                    'computed_data_summary': {
                        'n_scenarios': len(computed_data.get('scenarios', [])),
                        'max_sample_size': computed_data.get('max_sample_size', 0)
                    }
                })
                artifact.add_log(
                    category='computation',
                    message=f'Power analysis computation complete using {config["ttest_type"]} t-test',
                    details={
                        'ttest_type': config['ttest_type'],
                        'n_groups': config.get('n_groups', 2)
                    }
                )
                
                # Generate and save all plots to artifact immediately
                # Single plots
                single_plot_configs = [
                    {
                        "name": "sample_size_vs_uplift",
                        "function": create_single_plot_uplift,
                        "description": "Single plot: Sample Size vs Uplift"
                    },
                    {
                        "name": "sample_size_vs_alpha",
                        "function": create_single_plot_alpha,
                        "description": "Single plot: Sample Size vs Alpha"
                    },
                    {
                        "name": "sample_size_vs_power",
                        "function": create_single_plot_power,
                        "description": "Single plot: Sample Size vs Power"
                    }
                ]
                
                plots_generated = []
                for plot_config in single_plot_configs:
                    try:
                        fig = plot_config["function"](
                            computed_data['uplifts'], computed_data['alphas'], computed_data['powers'],
                            computed_data['Z_scenarios'], computed_data['scenarios'], computed_data['max_sample_size']
                        )
                        artifact.add_plot(plot_config["name"], fig, plot_config["description"])
                        plots_generated.append(plot_config["name"])
                    except Exception as e:
                        handle_plot_error(plot_config['name'], e)
                
                
                # Contour plots
                contour_plot_configs = [
                    {
                        "name": "contour_uplift_alpha_fixed_power",
                        "function": create_contour_plot_power,
                        "description": "Contour plot: Sample Size vs Uplift & Alpha (Fixed Power)"
                    },
                    {
                        "name": "contour_alpha_power_fixed_uplift",
                        "function": create_contour_plot_uplift,
                        "description": "Contour plot: Sample Size vs Alpha & Power (Fixed Uplift)"
                    },
                    {
                        "name": "contour_uplift_power_fixed_alpha",
                        "function": create_contour_plot_alpha,
                        "description": "Contour plot: Sample Size vs Uplift & Power (Fixed Alpha)"
                    }
                ]
                
                contour_plots_generated = []
                for plot_config in contour_plot_configs:
                    try:
                        fig = plot_config["function"](
                            computed_data['uplifts'], computed_data['alphas'], computed_data['powers'],
                            computed_data['Z_scenarios'], computed_data['scenarios'],
                            computed_data['max_sample_size'], computed_data['contour_bins'],
                            n_groups=config.get('n_groups', 2)
                        )
                        artifact.add_plot(plot_config["name"], fig, plot_config["description"])
                        contour_plots_generated.append(plot_config["name"])
                    except Exception as e:
                        handle_plot_error(plot_config['name'], e)
                
                # Summary message for all plots
                total_plots = len(plots_generated) + len(contour_plots_generated)

            
            # st.success(f"✅ Computation complete! Using **{config['ttest_type']}** t-test")
        
        # Display current settings if data is computed
        if st.session_state.computed_data:
            ttest_type = st.session_state.computed_data.get('ttest_type', 'two-sided')
            ttest_info = {
                'two-sided': 'H₁: μ₁ ≠ μ₂ (detecting any difference)',
                'larger': 'H₁: μ₁ > μ₂ (detecting increase only)',
                'smaller': 'H₁: μ₁ < μ₂ (detecting decrease only)'
            }
            st.info(f"📊 Current Analysis: **{ttest_type}** t-test — {ttest_info.get(ttest_type, '')}")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "🧮 Calculator",
            "📈 Single Plots",
            "🗺️ Contour Maps"
        ])

        # Tab 1: Calculator
        with tab1:
            selected_scenario, metrics = render_scenario_selector(
                config['scenarios'], 
                key="calculator_scenario"
            )
            
            if selected_scenario and metrics:
                render_calculator_tab(
                    config['stats_valid'],
                    power_analysis,
                    metrics
                )
            elif not config['scenarios']:
                st.warning("No scenarios available. Please configure your data.")

        # Tab 2: Single Plots
        with tab2:
            st.header("📈 Single Parameter Plots")
            st.markdown("Interactive line plots showing sample size variation across one parameter")
            
            if not render_validation_message(config['stats_valid'], st.session_state.computed_data is not None):
                pass  # Validation message already shown
            else:
                data = st.session_state.computed_data
                
                plot_type = st.radio(
                    "Select plot type:",
                    ["Sample Size vs Uplift", "Sample Size vs Alpha", "Sample Size vs Power"],
                    horizontal=True,
                    key="single_plot_type"
                )
                
                # Plot configuration
                plot_configs = {
                    "Sample Size vs Uplift": {
                        "subheader": "Sample Size vs Uplift",
                        "description": "**Use sliders below the plot to adjust fixed Alpha and Power values**",
                        "function": create_single_plot_uplift,
                        "filename": "sample_size_vs_uplift.html"
                    },
                    "Sample Size vs Alpha": {
                        "subheader": "Sample Size vs Alpha",
                        "description": "**Use sliders below the plot to adjust fixed Uplift and Power values**",
                        "function": create_single_plot_alpha,
                        "filename": "sample_size_vs_alpha.html"
                    },
                    "Sample Size vs Power": {
                        "subheader": "Sample Size vs Power",
                        "description": "**Use sliders below the plot to adjust fixed Uplift and Alpha values**",
                        "function": create_single_plot_power,
                        "filename": "sample_size_vs_power.html"
                    }
                }
                
                plot_config = plot_configs[plot_type]
                st.subheader(plot_config["subheader"])
                st.markdown(plot_config["description"])
                
                fig = plot_config["function"](
                    data['uplifts'], data['alphas'], data['powers'],
                    data['Z_scenarios'], data['scenarios'], data['max_sample_size']
                )
                
                # Plots are already added to artifact during computation, just render
                render_plot_with_download(fig, plot_config["filename"])

        # Tab 3: Contour Maps
        with tab3:
            st.header("🗺️ Contour Maps")
            st.markdown("Interactive 2D contour plots showing sample sizes across parameter combinations")
            
            if not render_validation_message(config['stats_valid'], st.session_state.computed_data is not None):
                pass  # Validation message already shown
            else:
                data = st.session_state.computed_data
                
                # Contour plot configurations
                contour_configs = [
                    {
                        "title": "📊 Map 1: Sample Size vs Uplift & Alpha (Fixed Power)",
                        "description": "**Interactive plot showing how sample size changes with uplift and alpha at different power levels**",
                        "function": create_contour_plot_power,
                        "filename": "contour_uplift_alpha_fixed_power.html"
                    },
                    {
                        "title": "📊 Map 2: Sample Size vs Alpha & Power (Fixed Uplift)",
                        "description": "**Interactive plot showing how sample size changes with alpha and power at different uplift levels**",
                        "function": create_contour_plot_uplift,
                        "filename": "contour_alpha_power_fixed_uplift.html"
                    },
                    {
                        "title": "📊 Map 3: Sample Size vs Uplift & Power (Fixed Alpha)",
                        "description": "**Interactive plot showing how sample size changes with uplift and power at different alpha levels**",
                        "function": create_contour_plot_alpha,
                        "filename": "contour_uplift_power_fixed_alpha.html"
                    }
                ]
                
                for plot_config in contour_configs:
                    with st.expander(plot_config["title"]):
                        st.markdown(plot_config["description"])
                        fig = plot_config["function"](
                            data['uplifts'], data['alphas'], data['powers'],
                            data['Z_scenarios'], data['scenarios'],
                            data['max_sample_size'], data['contour_bins'],
                            n_groups=config['n_groups']
                        )
                        
                        # Plots are already added to artifact during computation, just render
                        render_plot_with_download(fig, plot_config["filename"])


