"""
Generate dummy data for Results Analysis page
Creates realistic experiment data with pre/post columns showing treatment effects
"""
import pandas as pd
import numpy as np


def generate_results_analysis_data(
    n_rows: int = 10000,
    n_groups: int = 2,
    group_names: list = None,
    treatment_effect: float = 0.15,  # 15% uplift for treatment group
    cuped_suffix_pre: str = "_pre",
    cuped_suffix_post: str = "_post",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic experiment results data with pre/post columns.
    
    Creates data with:
    - ID column (customer_id)
    - 3 numeric columns with pre/post versions:
      1. Both groups increase, but treatment increases more
      2. Similar pre values, but in post one goes up, other stays same
      3. One group increases, other decreases (interesting contrast)
    - 2 numeric columns without pre/post:
      1. customer_churned (0-1): lower in treatment group
      2. new_customer (0-1): higher in treatment group
    - 2 categorical columns (region, device_type)
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    n_groups : int
        Number of groups (typically 2 for A/B test)
    group_names : list
        Optional list of group names (if None, generates Control, Treatment, etc.)
    treatment_effect : float
        Treatment effect as multiplier (0.15 = 15% uplift)
    cuped_suffix_pre : str
        Suffix for pre columns (used for both CUPED and DiD)
    cuped_suffix_post : str
        Suffix for post columns (used for both CUPED and DiD)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame: Generated data with groups and metrics (pre/post versions)
    """
    np.random.seed(random_seed)
    
    # Generate group names
    if group_names is None:
        if n_groups == 2:
            group_names = ['Control', 'Treatment']
        else:
            group_names = [f"Group_{chr(65+i)}" for i in range(n_groups)]
    else:
        n_groups = len(group_names)
    
    # Balanced group sizes
    base_size = n_rows // n_groups
    sizes = [base_size] * n_groups
    sizes[0] += (n_rows - sum(sizes))  # Add remainder to first group
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_rows + 1)]
    
    # Assign groups
    group_assignments = []
    for i, size in enumerate(sizes):
        group_assignments.extend([group_names[i]] * size)
    
    # Shuffle to randomize
    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    group_assignments = [group_assignments[i] for i in indices]
    customer_ids = [customer_ids[i] for i in indices]
    
    data = {
        'customer_id': customer_ids,
        'group': group_assignments
    }
    
    # ===== METRIC 1: Both groups increase, but treatment increases more =====
    # Column: revenue
    metric1_pre = np.zeros(n_rows)
    metric1_post = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        # Pre-period: Both groups have similar baseline (with some variation)
        pre_base = np.random.lognormal(mean=3.5, sigma=1.2, size=n_group)
        metric1_pre[group_mask] = np.round(pre_base, 2)
        
        # Post-period: Both increase, but treatment increases more
        if group_idx == 0:  # Control
            # Control: small increase (5-8%)
            increase = np.random.normal(loc=1.06, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.03, 1.10)
        else:  # Treatment
            # Treatment: larger increase (15-20%)
            increase = np.random.normal(loc=1.17, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.13, 1.22)
        
        metric1_post[group_mask] = np.round(pre_base * increase, 2)
    
    data['revenue'] = metric1_post
    data['revenue' + cuped_suffix_pre] = metric1_pre
    data['revenue' + cuped_suffix_post] = metric1_post
    
    # ===== METRIC 2: Similar pre values, but in post one goes up, other stays same =====
    # Column: engagement_score
    metric2_pre = np.zeros(n_rows)
    metric2_post = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        # Pre-period: Very similar values for both groups
        pre_base = np.random.normal(loc=50, scale=5, size=n_group)  # Tight distribution
        pre_base = np.clip(pre_base, 0, 100)
        metric2_pre[group_mask] = np.round(pre_base, 2)
        
        # Post-period: Treatment goes up, Control stays almost same
        if group_idx == 0:  # Control
            # Control: stays almost the same (slight variation ±2%)
            variation = np.random.normal(loc=1.0, scale=0.01, size=n_group)
            variation = np.clip(variation, 0.98, 1.02)
            metric2_post[group_mask] = np.round(pre_base * variation, 2)
        else:  # Treatment
            # Treatment: significant increase (20-25%)
            increase = np.random.normal(loc=1.22, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.18, 1.27)
            metric2_post[group_mask] = np.round(pre_base * increase, 2)
            metric2_post[group_mask] = np.clip(metric2_post[group_mask], 0, 100)
    
    data['engagement_score'] = metric2_post
    data['engagement_score' + cuped_suffix_pre] = metric2_pre
    data['engagement_score' + cuped_suffix_post] = metric2_post
    
    # ===== METRIC 3: One group increases, other decreases (interesting contrast) =====
    # Column: support_tickets
    metric3_pre = np.zeros(n_rows)
    metric3_post = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        # Pre-period: Both groups start with similar baseline
        pre_base = np.random.poisson(lam=3, size=n_group).astype(float)
        metric3_pre[group_mask] = np.round(pre_base, 2)
        
        # Post-period: Treatment decreases (good!), Control increases slightly
        if group_idx == 0:  # Control
            # Control: slight increase (bad - more tickets)
            increase = np.random.normal(loc=1.08, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.05, 1.12)
            metric3_post[group_mask] = np.round(pre_base * increase, 2)
        else:  # Treatment
            # Treatment: decreases (good - fewer tickets, better service)
            decrease = np.random.normal(loc=0.75, scale=0.03, size=n_group)
            decrease = np.clip(decrease, 0.70, 0.82)
            metric3_post[group_mask] = np.round(pre_base * decrease, 2)
            metric3_post[group_mask] = np.maximum(metric3_post[group_mask], 0)  # No negatives
    
    data['support_tickets'] = metric3_post
    data['support_tickets' + cuped_suffix_pre] = metric3_pre
    data['support_tickets' + cuped_suffix_post] = metric3_post
    
    # ===== METRIC 4: customer_churned (0-1) - lower in treatment =====
    churned_values = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        if group_idx == 0:  # Control
            # Control: higher churn rate (0.15-0.20)
            churn_rate = np.random.beta(a=2, b=8, size=n_group)  # Mean ~0.2
        else:  # Treatment
            # Treatment: lower churn rate (0.08-0.12)
            churn_rate = np.random.beta(a=2, b=15, size=n_group)  # Mean ~0.12
        
        # Convert to binary (0 or 1)
        churned_values[group_mask] = (np.random.random(n_group) < churn_rate).astype(int)
    
    data['customer_churned'] = churned_values.astype(int)
    
    # ===== METRIC 5: new_customer (0-1) - higher in treatment =====
    new_customer_values = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        if group_idx == 0:  # Control
            # Control: lower new customer rate (0.10-0.15)
            new_rate = np.random.beta(a=2, b=12, size=n_group)  # Mean ~0.14
        else:  # Treatment
            # Treatment: higher new customer rate (0.20-0.25)
            new_rate = np.random.beta(a=3, b=10, size=n_group)  # Mean ~0.23
        
        # Convert to binary (0 or 1)
        new_customer_values[group_mask] = (np.random.random(n_group) < new_rate).astype(int)
    
    data['new_customer'] = new_customer_values.astype(int)
    
    # Add categorical columns (2 as per standard)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_rows)
    data['device_type'] = np.random.choice(['Desktop', 'Mobile', 'Tablet'], size=n_rows, p=[0.5, 0.4, 0.1])
    
    df = pd.DataFrame(data)
    
    return df
