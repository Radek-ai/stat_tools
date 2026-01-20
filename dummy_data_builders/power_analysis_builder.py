"""
Generate dummy data for Power Analysis page
Creates data similar to group selection but without group assignments
"""
import pandas as pd
import numpy as np


def generate_power_analysis_data(
    n_rows: int = 10000,
    n_numeric_metrics: int = 4,
    n_categorical: int = 2,
    include_outliers: bool = True,
    missing_pct: float = 0.05,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate dummy data for power analysis.
    
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
    np.random.seed(random_seed)
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_rows + 1)]
    
    # Generate numeric metrics with realistic distributions
    numeric_data = {}
    metric_names = ['revenue', 'clicks', 'conversions', 'session_duration', 'page_views',
                    'bounce_rate', 'time_on_site', 'engagement_score', 'purchase_value', 'ad_impressions']
    
    for i in range(n_numeric_metrics):
        metric_name = metric_names[i % len(metric_names)]
        
        if metric_name == 'revenue':
            # Right-skewed lognormal
            values = np.random.lognormal(mean=3.5, sigma=1.2, size=n_rows)
            if include_outliers:
                outlier_indices = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
                values[outlier_indices] *= np.random.uniform(5, 20, size=len(outlier_indices))
            values = np.round(values, 2)
            
        elif metric_name == 'clicks':
            # Negative binomial
            values = np.random.negative_binomial(n=10, p=0.3, size=n_rows)
            values = np.maximum(values, 0).astype(float)
            
        elif metric_name == 'conversions':
            # Binomial-like
            clicks = np.random.negative_binomial(n=10, p=0.3, size=n_rows)
            conversion_rate = np.random.beta(5, 95, size=n_rows)
            values = np.random.binomial(clicks.astype(int), conversion_rate).astype(float)
            
        elif metric_name == 'session_duration':
            # Normal with some outliers
            values = np.random.normal(loc=15, scale=8, size=n_rows)
            values = np.maximum(values, 0)
            if include_outliers:
                long_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
                values[long_indices] += np.random.uniform(30, 120, size=len(long_indices))
            values = np.round(values, 1)
            
        elif metric_name == 'page_views':
            # Lognormal
            values = np.random.lognormal(mean=2.5, sigma=0.8, size=n_rows)
            values = np.maximum(values, 1).astype(float)
            
        elif metric_name == 'bounce_rate':
            # Beta (0-1)
            values = np.random.beta(2, 3, size=n_rows)
            values = np.round(values, 3).astype(float)
            
        elif metric_name == 'time_on_site':
            # Correlated with session duration
            base = np.random.normal(loc=15, scale=8, size=n_rows)
            values = base * 60 + np.random.normal(0, 300, size=n_rows)
            values = np.maximum(values, 0)
            values = np.round(values, 0).astype(float)
            
        elif metric_name == 'engagement_score':
            # Normal
            values = np.random.normal(loc=50, scale=15, size=n_rows)
            values = np.clip(values, 0, 100)
            values = np.round(values, 2)
            
        elif metric_name == 'purchase_value':
            # Lognormal
            values = np.random.lognormal(mean=4.0, sigma=1.0, size=n_rows)
            values = np.round(values, 2)
            
        else:  # ad_impressions
            # Poisson-like
            values = np.random.poisson(lam=100, size=n_rows).astype(float)
        
        # Add missing values
        if missing_pct > 0:
            nan_indices = np.random.choice(n_rows, size=int(n_rows * missing_pct), replace=False)
            values = values.astype(float)
            values[nan_indices] = np.nan
        
        numeric_data[metric_name] = values
    
    # Generate categorical columns
    categorical_data = {}
    cat_names = ['region', 'device_type', 'channel', 'campaign', 'segment']
    cat_options = [
        ['North', 'South', 'East', 'West'],
        ['Desktop', 'Mobile', 'Tablet'],
        ['Organic', 'Paid', 'Direct', 'Referral'],
        ['Campaign A', 'Campaign B', 'Campaign C'],
        ['High Value', 'Medium Value', 'Low Value']
    ]
    
    for i in range(n_categorical):
        cat_name = cat_names[i % len(cat_names)]
        options = cat_options[i % len(cat_options)]
        probs = np.random.dirichlet(np.ones(len(options)))
        categorical_data[cat_name] = np.random.choice(options, size=n_rows, p=probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        **numeric_data,
        **categorical_data
    })
    
    return df
