"""
Computation logic for power analysis
"""
import numpy as np
import streamlit as st
from time import sleep

from power_analysis.power_analysis import PowerAnalysis


def compute_all_scenarios(
    config: dict,
    progress_callback=None
) -> dict:
    """
    Compute sample sizes for all scenarios and parameter combinations.
    
    Args:
        config: Configuration dictionary with scenarios and parameters
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary containing computed data
    """
    # Initialize power analysis with t-test type
    power_analysis = PowerAnalysis(alternative=config['ttest_type'])
    
    # Create parameter arrays
    uplifts = np.linspace(config['uplift_min'], config['uplift_max'], config['uplift_points'])
    alphas = np.linspace(config['alpha_min'], config['alpha_max'], config['alpha_points'])
    powers = np.linspace(config['power_min'], config['power_max'], config['power_points'])
    
    # Compute Z_groups for each scenario
    Z_scenarios = {}
    total_scenarios = len(config['scenarios'])
    
    for scenario_idx, (scenario_name, metrics) in enumerate(config['scenarios'].items()):
        def scenario_progress(value):
            if progress_callback:
                overall_progress = (scenario_idx + value) / total_scenarios
                progress_callback(overall_progress)
        
        Z_groups = power_analysis.compute_all_sample_sizes(
            uplifts, alphas, powers,
            metrics,
            progress_callback=scenario_progress
        )
        Z_scenarios[scenario_name] = Z_groups
    
    # Calculate max_sample_size as 3x the maximum user-provided sample size across all scenarios
    max_sample_size = get_max_sample_size(config)
    contour_bins = 50  # Fixed number of contour bins
    
    return {
        'uplifts': uplifts,
        'alphas': alphas,
        'powers': powers,
        'Z_scenarios': Z_scenarios,
        'scenarios': config['scenarios'],
        'max_sample_size': max_sample_size,
        'contour_bins': contour_bins,
        'ttest_type': config['ttest_type']
    }


def get_max_sample_size(config: dict) -> int:
    """
    Calculate maximum sample size for plots.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Maximum sample size as integer
    """
    max_user_sample_size = max(
        metric_data.get('sample_size', 0) 
        for scenario in config['scenarios'].values()
        for metric_data in scenario.values()
    ) / config['n_groups']
    return int(max_user_sample_size * 3) if max_user_sample_size > 0 else 200000


def compute_with_progress(config: dict) -> dict:
    """
    Compute all scenarios with progress bar.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing computed data
    """
    progress_text = st.empty()
    progress_text.info("⏳ Computing sample sizes for all scenarios...")
    progress_bar = st.progress(0)
    
    def update_progress(value):
        progress_bar.progress(value)
    
    sleep(0.5)
    
    computed_data = compute_all_scenarios(config, progress_callback=update_progress)
    
    progress_bar.empty()
    progress_text.empty()
    
    return computed_data

