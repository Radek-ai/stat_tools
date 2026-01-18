"""
Data processing utilities for scenarios and metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Import shared filtering utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_filtering import (
    remove_nans,
    filter_outliers_percentile,
    filter_outliers_iqr,
    winsorize_percentile
)


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Calculate statistics for a given column.
    
    Args:
        df: Input dataframe
        column: Column name
        
    Returns:
        Dictionary with mean, std, and sample_size
    """
    if len(df) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'sample_size': 0
        }
    
    return {
        'mean': float(df[column].mean()),
        'std': float(df[column].std()),
        'sample_size': len(df)
    }


def calculate_retention(df_before: pd.DataFrame, df_after: pd.DataFrame, column: str) -> Tuple[float, float]:
    """
    Calculate rows_retained and metric_retained.
    
    Args:
        df_before: Dataframe before outlier filtering (after NaN removal)
        df_after: Dataframe after outlier filtering
        column: Column name
        
    Returns:
        Tuple of (rows_retained, metric_retained) as ratios
    """
    if len(df_before) == 0:
        return 1.0, 1.0
    
    rows_retained = len(df_after) / len(df_before) if len(df_before) > 0 else 1.0
    
    metric_before = df_before[column].sum()
    metric_after = df_after[column].sum()
    metric_retained = metric_after / metric_before if metric_before > 0 else 1.0
    
    return rows_retained, metric_retained


def generate_scenario_name(outlier_method: str, p_low: float = None, p_high: float = None, 
                           iqr_multiplier: float = None) -> str:
    """
    Generate automatic scenario name based on outlier method and parameters.
    
    Args:
        outlier_method: Method name ("none", "percentile", "winsorize", or "iqr")
        p_low: Lower percentile (if percentile/winsorize method)
        p_high: Upper percentile (if percentile/winsorize method)
        iqr_multiplier: IQR multiplier (if iqr method)
        
    Returns:
        Scenario name string
    """
    if outlier_method == "none":
        return "No Outlier Filtering"
    elif outlier_method == "percentile":
        return f"Percentile_{p_low}_{p_high}"
    elif outlier_method == "winsorize":
        return f"Winsorize_{p_low}_{p_high}"
    elif outlier_method == "iqr":
        return f"IQR_{iqr_multiplier}"
    else:
        return f"Scenario_{outlier_method}"


def compute_all_scenarios(
    df: pd.DataFrame,
    scenarios_config: List[Dict]
) -> Dict:
    """
    Compute statistics for all scenario-metric combinations.
    
    Args:
        df: Raw dataframe (with all data)
        scenarios_config: List of scenario configs, each with:
            {
                "method": "percentile"/"winsorize"/"iqr"/"none",
                "p_low": float (for percentile/winsorize),
                "p_high": float (for percentile/winsorize),
                "iqr_multiplier": float (for iqr),
                "metrics": [{"name": "revenue", "column": "revenue"}, ...]
            }
        
    Returns:
        Dictionary in the format expected by power analysis: {scenario_name: {metric_name: {stats}}}
    """
    results = {}
    
    for scenario_config in scenarios_config:
        scenario_name = generate_scenario_name(
            scenario_config['method'],
            scenario_config.get('p_low'),
            scenario_config.get('p_high'),
            scenario_config.get('iqr_multiplier')
        )
        
        scenario_results = {}
        
        for metric_config in scenario_config.get('metrics', []):
            metric_name = metric_config['name']
            column = metric_config['column']
            
            # Step 1: Remove NaNs
            df_no_nan = remove_nans(df, column)
            base_sample_size = len(df_no_nan)
            
            if base_sample_size == 0:
                # No data after NaN removal
                scenario_results[metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'sample_size': 0,
                    'rows_retained': 1.0,
                    'metric_retained': 1.0
                }
                continue
            
            # Step 2: Apply outlier filtering
            if scenario_config['method'] == 'none':
                df_filtered = df_no_nan.copy()
            elif scenario_config['method'] == 'percentile':
                df_filtered = filter_outliers_percentile(
                    df_no_nan,
                    column,
                    scenario_config['p_low'],
                    scenario_config['p_high']
                )
            elif scenario_config['method'] == 'winsorize':
                # Winsorization clips values but doesn't remove rows
                df_filtered = winsorize_percentile(
                    df_no_nan,
                    column,
                    scenario_config['p_low'],
                    scenario_config['p_high']
                )
            elif scenario_config['method'] == 'iqr':
                df_filtered = filter_outliers_iqr(
                    df_no_nan,
                    column,
                    scenario_config.get('iqr_multiplier', 1.5)
                )
            else:
                df_filtered = df_no_nan.copy()
            
            # Step 3: Calculate statistics
            stats = calculate_statistics(df_filtered, column)
            
            # Step 4: Calculate retention
            # For winsorization, no rows are removed, so retention is 1.0
            if scenario_config['method'] == 'winsorize':
                rows_retained = 1.0
                # Metric retained might be < 1.0 if values were clipped downward
                metric_before = df_no_nan[column].sum()
                metric_after = df_filtered[column].sum()
                metric_retained = metric_after / metric_before if metric_before > 0 else 1.0
            else:
                rows_retained, metric_retained = calculate_retention(
                    df_no_nan, df_filtered, column
                )
            
            scenario_results[metric_name] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'sample_size': stats['sample_size'],
                'rows_retained': rows_retained,
                'metric_retained': metric_retained
            }
        
        results[scenario_name] = scenario_results
    
    return results

