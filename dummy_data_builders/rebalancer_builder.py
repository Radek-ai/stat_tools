"""
Generate dummy data for Rebalancer page
Creates data with existing group assignments that are imbalanced
"""
import pandas as pd
import numpy as np


def generate_rebalancer_data(
    n_rows: int = 10000,
    n_groups: int = 3,
    group_names: list = None,
    imbalance_level: float = 0.05,  # Small imbalance for rebalancer (was 0.3)
    n_numeric_metrics: int = 4,
    n_categorical: int = 2,
    include_outliers: bool = True,
    missing_pct: float = 0.0,  # No missing values for rebalancer
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate dummy data for rebalancer with slightly imbalanced groups.
    Groups are mostly balanced (similar counts, means, categorical distribution)
    with small imbalances that the rebalancer can fix.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    n_groups : int
        Number of groups to create
    group_names : list
        Optional list of group names (if None, generates A, B, C, etc.)
    imbalance_level : float
        Level of imbalance (0-1), higher = more imbalanced (default: 0.05 for slight imbalance)
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
    pd.DataFrame: Generated data with slightly imbalanced group assignments
    """
    np.random.seed(random_seed)
    
    # Generate group names
    if group_names is None:
        group_names = [f"Group_{chr(65+i)}" for i in range(n_groups)]
    else:
        n_groups = len(group_names)
    
    # Create imbalanced group sizes
    base_size = n_rows // n_groups
    sizes = [base_size] * n_groups
    
    # Add imbalance
    total_extra = int(n_rows * imbalance_level)
    for i in range(total_extra):
        # Randomly add to groups to create imbalance
        group_idx = np.random.randint(0, n_groups)
        sizes[group_idx] += 1
    
    # Adjust to match total
    total = sum(sizes)
    if total != n_rows:
        sizes[0] += (n_rows - total)
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_rows + 1)]
    
    # Assign groups (imbalanced)
    group_assignments = []
    for i, size in enumerate(sizes):
        group_assignments.extend([group_names[i]] * size)
    
    # Shuffle to mix groups
    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    group_assignments = [group_assignments[i] for i in indices]
    customer_ids = [customer_ids[i] for i in indices]
    
    # Generate numeric metrics with group-specific distributions (to create imbalance)
    numeric_data = {}
    metric_names = ['revenue', 'clicks', 'conversions', 'session_duration', 'page_views',
                    'bounce_rate', 'time_on_site', 'engagement_score', 'purchase_value', 'ad_impressions']
    
    for i in range(n_numeric_metrics):
        metric_name = metric_names[i % len(metric_names)]
        values = np.zeros(n_rows)
        
        for group_idx, group_name in enumerate(group_names):
            group_mask = np.array([g == group_name for g in group_assignments])
            n_group = np.sum(group_mask)
            
            if n_group == 0:
                continue
            
            # Create small group-specific means (mostly balanced with slight differences)
            # Use smaller shift to keep groups more balanced
            group_mean_shift = (group_idx - n_groups/2) * 0.05 * imbalance_level
            
            if metric_name == 'revenue':
                base = np.random.lognormal(mean=3.5 + group_mean_shift, sigma=1.2, size=n_group)
                if include_outliers:
                    outlier_count = max(1, int(n_group * 0.01))
                    outlier_indices = np.random.choice(n_group, size=outlier_count, replace=False)
                    base[outlier_indices] *= np.random.uniform(5, 20, size=len(outlier_indices))
                values[group_mask] = np.round(base, 2)
                
            elif metric_name == 'clicks':
                lam = 10 + group_mean_shift * 5
                values[group_mask] = np.maximum(np.random.poisson(lam=max(1, lam), size=n_group), 0).astype(float)
                
            elif metric_name == 'conversions':
                clicks = np.random.poisson(lam=10, size=n_group)
                cr = np.random.beta(5 + group_mean_shift * 2, 95, size=n_group)
                values[group_mask] = np.random.binomial(clicks, cr).astype(float)
                
            elif metric_name == 'session_duration':
                base = np.random.normal(loc=15 + group_mean_shift * 2, scale=8, size=n_group)
                values[group_mask] = np.maximum(np.round(base, 1), 0)
                
            elif metric_name == 'page_views':
                base = np.random.lognormal(mean=2.5 + group_mean_shift * 0.1, sigma=0.8, size=n_group)
                values[group_mask] = np.maximum(base, 1).astype(float)
                
            elif metric_name == 'bounce_rate':
                base = np.random.beta(2, 3, size=n_group)
                values[group_mask] = np.round(base, 3).astype(float)
                
            elif metric_name == 'time_on_site':
                base = np.random.normal(loc=15 + group_mean_shift * 2, scale=8, size=n_group)
                values[group_mask] = np.maximum(np.round(base * 60, 0), 0).astype(float)
                
            elif metric_name == 'engagement_score':
                base = np.random.normal(loc=50 + group_mean_shift * 5, scale=15, size=n_group)
                values[group_mask] = np.clip(np.round(base, 2), 0, 100)
                
            elif metric_name == 'purchase_value':
                base = np.random.lognormal(mean=4.0 + group_mean_shift * 0.1, sigma=1.0, size=n_group)
                values[group_mask] = np.round(base, 2)
                
            else:  # ad_impressions
                lam = 100 + group_mean_shift * 20
                values[group_mask] = np.maximum(np.random.poisson(lam=max(1, lam), size=n_group), 0).astype(float)
        
        # Add missing values
        if missing_pct > 0:
            nan_indices = np.random.choice(n_rows, size=int(n_rows * missing_pct), replace=False)
            values = values.astype(float)
            values[nan_indices] = np.nan
        
        numeric_data[metric_name] = values
    
    # Generate categorical columns with balanced distribution across groups
    categorical_data = {}
    cat_names = ['region', 'device_type', 'channel', 'campaign', 'segment']
    cat_options = [
        ['North', 'South', 'East', 'West'],
        ['Desktop', 'Mobile', 'Tablet'],
        ['Organic', 'Paid', 'Direct', 'Referral'],
        ['Campaign A', 'Campaign B', 'Campaign C'],
        ['High Value', 'Medium Value', 'Low Value']
    ]
    
    # Create array to store categorical values in correct order
    for i in range(n_categorical):
        cat_name = cat_names[i % len(cat_names)]
        options = cat_options[i % len(cat_options)]
        n_options = len(options)
        
        # Initialize array
        cat_values = np.empty(n_rows, dtype=object)
        
        # Generate values per group to maintain balance
        for group_idx, group_name in enumerate(group_names):
            group_mask = np.array([g == group_name for g in group_assignments])
            n_group = np.sum(group_mask)
            
            if n_group == 0:
                continue
            
            # Create mostly balanced distribution with slight variations
            # Base probabilities are uniform (balanced)
            base_probs = np.ones(n_options) / n_options
            
            # Slight variation from base probabilities per group
            # Use dirichlet with small alpha to create slight imbalance
            alpha = np.ones(n_options) * (1.0 + imbalance_level * 2)
            # Add small group-specific bias for slight imbalance
            alpha[group_idx % n_options] += imbalance_level * 5
            probs = np.random.dirichlet(alpha)
            
            # Blend with base probabilities to keep mostly balanced
            probs = 0.8 * base_probs + 0.2 * probs
            probs = probs / probs.sum()  # Normalize
            
            # Generate values for this group
            group_cat_values = np.random.choice(options, size=n_group, p=probs)
            cat_values[group_mask] = group_cat_values
        
        categorical_data[cat_name] = cat_values.tolist()
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'group': group_assignments,
        **numeric_data,
        **categorical_data
    })
    
    return df
