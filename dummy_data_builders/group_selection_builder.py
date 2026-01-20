"""
Generate dummy data for Group Selection page
Creates data without group assignments (groups will be created by the tool)
"""
import pandas as pd
import numpy as np
from dummy_data_builders.power_analysis_builder import generate_power_analysis_data


def generate_group_selection_data(
    n_rows: int = 10000,
    n_numeric_metrics: int = 4,
    n_categorical: int = 2,
    include_outliers: bool = True,
    missing_pct: float = 0.0,  # No missing values for group selection
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate dummy data for group selection.
    Same as power analysis data (no groups assigned yet).
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    n_numeric_metrics : int
        Number of numeric metric columns
    n_categorical : int
        Number of categorical columns
    include_outliers : bool
        Whether to include outliers in numeric columns
    missing_pct : float
        Percentage of missing values (0-1)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame: Generated data without group assignments
    """
    return generate_power_analysis_data(
        n_rows=n_rows,
        n_numeric_metrics=n_numeric_metrics,
        n_categorical=n_categorical,
        include_outliers=include_outliers,
        missing_pct=missing_pct,
        random_seed=random_seed
    )
