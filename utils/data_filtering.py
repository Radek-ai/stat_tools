"""
Data filtering utilities for outlier removal and value-based filtering
Shared between power_analysis and group_selection modules
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def remove_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove rows with NaN values in the specified column.
    
    Args:
        df: Input dataframe
        column: Column name to check for NaNs
        
    Returns:
        Dataframe with NaNs removed
    """
    return df.dropna(subset=[column])


def filter_outliers_percentile(df: pd.DataFrame, column: str, p_low: float, p_high: float) -> pd.DataFrame:
    """
    Filter outliers using percentile-based method.
    Rows with values outside [p_low, p_high] percentiles are removed.
    
    Args:
        df: Input dataframe
        column: Column name to process
        p_low: Lower percentile (0-100)
        p_high: Upper percentile (0-100)
        
    Returns:
        Dataframe with outliers removed
    """
    lower_bound = df[column].quantile(p_low / 100.0)
    upper_bound = df[column].quantile(p_high / 100.0)
    
    # Keep only rows within the percentile range
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    return filtered_df


def winsorize_percentile(df: pd.DataFrame, column: str, p_low: float, p_high: float) -> pd.DataFrame:
    """
    Winsorize (clip) outliers using percentile-based method.
    Values outside [p_low, p_high] percentiles are clipped to those boundaries.
    
    Args:
        df: Input dataframe
        column: Column name to process
        p_low: Lower percentile (0-100)
        p_high: Upper percentile (0-100)
        
    Returns:
        Dataframe with clipped values (same number of rows)
    """
    df = df.copy()
    lower_bound = df[column].quantile(p_low / 100.0)
    upper_bound = df[column].quantile(p_high / 100.0)
    
    # Clip values to boundaries
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def filter_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Filter outliers using IQR (Interquartile Range) method.
    Rows with values outside Q1 - multiplier*IQR and Q3 + multiplier*IQR are removed.
    
    Args:
        df: Input dataframe
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Dataframe with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Keep only rows within the IQR range
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    return filtered_df


def filter_by_value_range(df: pd.DataFrame, column: str, min_value: float = None, max_value: float = None) -> pd.DataFrame:
    """
    Filter rows based on value range.
    
    Args:
        df: Input dataframe
        column: Column name to filter on
        min_value: Minimum value (inclusive), None to skip
        max_value: Maximum value (inclusive), None to skip
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    if min_value is not None:
        filtered_df = filtered_df[filtered_df[column] >= min_value]
    if max_value is not None:
        filtered_df = filtered_df[filtered_df[column] <= max_value]
    
    return filtered_df


def filter_by_categorical_values(df: pd.DataFrame, column: str, keep_values: List = None, exclude_values: List = None) -> pd.DataFrame:
    """
    Filter rows based on categorical values.
    
    Args:
        df: Input dataframe
        column: Column name to filter on
        keep_values: List of values to keep (None to skip)
        exclude_values: List of values to exclude (None to skip)
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    if keep_values is not None:
        filtered_df = filtered_df[filtered_df[column].isin(keep_values)]
    if exclude_values is not None:
        filtered_df = filtered_df[~filtered_df[column].isin(exclude_values)]
    
    return filtered_df


def winsorize_percentile(df: pd.DataFrame, column: str, p_low: float, p_high: float) -> pd.DataFrame:
    """
    Winsorize (clip) outliers using percentile-based method.
    Values outside [p_low, p_high] percentiles are clipped to those boundaries.
    
    Args:
        df: Input dataframe
        column: Column name to process
        p_low: Lower percentile (0-100)
        p_high: Upper percentile (0-100)
        
    Returns:
        Dataframe with clipped values (same number of rows)
    """
    df = df.copy()
    lower_bound = df[column].quantile(p_low / 100.0)
    upper_bound = df[column].quantile(p_high / 100.0)
    
    # Clip values to boundaries
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df
