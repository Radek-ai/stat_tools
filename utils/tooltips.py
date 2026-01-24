"""
Centralized tooltips registry for all application pages
Consolidates tooltips to avoid duplication and ensure consistency
"""
from typing import Dict

# Shared tooltips (used across multiple pages)
SHARED_TOOLTIPS: Dict[str, str] = {
    # Group/Column Selection (used in multiple pages)
    "group_column": "Select the column containing group assignments. This column should have values indicating which group each participant belongs to (e.g., 'control', 'treatment', 'variant_A').",
    "numeric_columns": "Numeric columns to balance/analyze. These will be balanced using statistical tests (t-tests, p-values, SMD). Select columns that could affect your outcome metric or are confounders.",
    "categorical_columns": "Categorical columns for stratification/balancing. These will be balanced by ensuring similar distributions across groups. Select columns like region, device type, customer segment, etc.",
    
    # Balancing Objectives (used in Group Selection and Rebalancer)
    "target_p_value": "Target p-value for t-test between groups for this numeric column. Higher values (e.g., 0.95) indicate stricter balance requirements - the algorithm will try to achieve this p-value or higher (maximizes p). Higher positive values indicate better balance.",
    "max_imbalance_percent": "Maximum acceptable total imbalance percentage for this categorical column. Lower values (e.g., 5%) are stricter. The algorithm will try to keep imbalance below this threshold.",
    
    # Algorithm Settings (used in Group Selection and Rebalancer)
    "gain_threshold": "Minimum improvement (gain) required to accept a move/swap/removal. Lower values allow smaller improvements. If no move improves by at least this amount, the process stops. Typical: 0.0001-0.01.",
    "early_break": "Stop searching candidates once a good move is found. Speeds up processing but may miss better moves. Recommended: Enabled for faster processing, Disabled for thorough search.",
    "top_k_candidates": "Number of top candidates to consider based on their contribution to imbalance. Higher values consider more candidates but are slower. Typical: 10-50 candidates.",
    "k_random_candidates": "Number of random candidates to consider in addition to top K. Adds randomness to prevent getting stuck in local optima. Higher values explore more but are slower. Typical: 10-500 candidates depending on context.",
    
    # Filtering (used in multiple pages)
    "outlier_method": "Method to handle outliers: none (keep all), percentile (remove by percentile), winsorize (clip values), iqr (remove by IQR)",
    "p_low": "Lower percentile threshold (0-100). Values below this percentile will be filtered/winsorized",
    "p_high": "Upper percentile threshold (0-100). Values above this percentile will be filtered/winsorized",
    "iqr_multiplier": "Multiplier for IQR method. Standard: 1.5 (removes values beyond 1.5×IQR from quartiles)",
    
    # Common Settings
    "n_groups": "Total number of groups in your experiment (e.g., 2 = control + treatment, 3 = control + treatment1 + treatment2)",
    "random_seed": "Random seed for reproducible results. Use the same seed to get identical results. Set to None for truly random assignments.",
}

# Power Analysis specific tooltips
POWER_ANALYSIS_TOOLTIPS: Dict[str, str] = {
    "uplift_min": "Minimum expected relative change (effect size). Example: 0.005 = 0.5% increase",
    "uplift_max": "Maximum expected relative change (effect size). Example: 0.1 = 10% increase",
    "alpha_min": "Minimum significance level (Type I error rate). Common: 0.05 (5%)",
    "alpha_max": "Maximum significance level. Higher values allow more false positives",
    "power_min": "Minimum statistical power (1 - Type II error). Common: 0.8 (80%)",
    "power_max": "Maximum statistical power. Higher power requires larger sample sizes",
    "uplift_points": "Number of data points to calculate for uplift range (affects plot resolution)",
    "alpha_points": "Number of data points to calculate for alpha range (affects plot resolution)",
    "power_points": "Number of data points to calculate for power range (affects plot resolution)",
    "ttest_type": "Direction of alternative hypothesis: two-sided (any difference), larger (increase only), smaller (decrease only)",
}

# Group Selection specific tooltips
GROUP_SELECTION_TOOLTIPS: Dict[str, str] = {
    "group_column_name": "Name of the column that will store group assignments in the output data. This column will be added to your dataset with values matching the group names you specify.",
    "group_name": "Name for this group (e.g., 'control', 'treatment', 'variant_A'). Group names must be unique and will appear in the output data.",
    "group_proportion": "Relative size of this group as a proportion (0.0 to 1.0). Proportions are automatically normalized to sum to 1.0. Example: 0.5 = 50% of participants, 0.33 = 33% of participants.",
    "selection_mode": "Basic: Creates stratified initial groups only (fast, no iterative optimization). Advanced: Creates initial groups then iteratively optimizes them for better balance (slower but more thorough).",
    "n_bins": "Number of bins to create for numeric column stratification. More bins = finer stratification but may create smaller groups. Typical range: 3-5 bins.",
    "algorithm": "Balancing algorithm: Sequential Moves (move rows between groups, more flexible) or Swaps (swap rows between groups, preserves exact group sizes).",
    "batch_mode": "Enable to move/swap groups of rows at once instead of single rows. Reduces overfitting by making larger, more robust changes. Recommended for large datasets.",
    "max_iterations": "Maximum number of balancing iterations to run. Each iteration attempts to improve balance. More iterations may improve balance but take longer. Typical: 50-200 iterations.",
    "batch_size": "Number of rows to move/swap in each batch operation. Larger batches make bigger changes but may be less precise. Typical: 3-10 rows per batch.",
    "random_samples": "Number of random batch samples to try in each iteration. More samples = better chance of finding good moves but slower. Typical: 10-50 samples.",
    "continue_balancing": "Continue balancing from the previously balanced groups. Uses the current balanced state as the starting point, allowing you to iteratively improve balance across multiple runs. Advanced mode only.",
}

# Rebalancer specific tooltips
REBALANCER_TOOLTIPS: Dict[str, str] = {
    "rebalancing_mode": "Basic: Even-size seed search only - subsamples all groups to smallest group's size. Advanced: Iterative rebalancing with intelligent row removal - more thorough but may remove more rows.",
    "enable_seed_search": "Enable to subsample all groups to the smallest group's size before rebalancing. Tests multiple random seeds to find the best initial subsample. Helps find better starting points for rebalancing.",
    "even_size_trials": "Number of random seeds to try for even-size subsampling. More trials = better chance of finding optimal subsample but slower. Typical: 1000-10000 trials.",
    "enable_seed_search_advanced": "Enable to first subsample all groups to smallest size before iterative rebalancing. Useful when groups have very different sizes. Tests multiple random seeds to find best starting point.",
    "even_size_trials_advanced": "Number of random seeds to try for even-size subsampling in Advanced mode. More trials = better chance of finding optimal starting point but slower. Typical: 1000-10000 trials.",
    "max_removals": "Maximum number of rows to remove per group during rebalancing. Set based on how much data you can afford to lose. Start conservative (e.g., 100 rows per group) and increase if needed. Typical: 50-500 rows per group.",
    "continue_rebalancing": "Continue rebalancing from the previously rebalanced groups. Uses the current rebalanced state as the starting point, allowing you to iteratively improve balance across multiple runs. Useful when initial run didn't achieve desired balance.",
}

# Results Analysis specific tooltips
RESULTS_ANALYSIS_TOOLTIPS: Dict[str, str] = {
    "metric_columns": "Select numeric metric columns to analyze for treatment effects. These should be outcome metrics from your experiment (e.g., revenue, conversion_rate, customer_churned). Pre/post columns (ending with _pre, _post) are excluded as they should be analyzed via CUPED/DiD.",
    "cuped_suffix_pre": "Suffix for pre-experiment metric columns. The tool will look for columns like 'metric{suffix_pre}' (e.g., 'revenue_pre' if suffix is '_pre'). Default: '_pre'",
    "cuped_suffix_post": "Suffix for post-experiment metric columns. The tool will look for columns like 'metric{suffix_post}' (e.g., 'revenue_post' if suffix is '_post'). Default: '_post'",
    "cuped_metrics": "Select metrics that have both pre and post versions. Only metrics with matching pre/post column pairs will be available. CUPED uses pre-experiment data to reduce variance and improve precision.",
    "did_suffix_pre": "Suffix for pre-period metric columns. The tool will look for columns like 'metric{suffix_pre}' (e.g., 'revenue_pre' if suffix is '_pre'). Default: '_pre'",
    "did_suffix_post": "Suffix for post-period metric columns. The tool will look for columns like 'metric{suffix_post}' (e.g., 'revenue_post' if suffix is '_post'). Default: '_post'",
    "did_metrics": "Select metrics that have both pre and post versions. Only metrics with matching pre/post column pairs will be available. DiD compares changes between groups across pre/post periods.",
}

# Combined tooltip dictionaries for each page
def get_power_analysis_tooltips() -> Dict[str, str]:
    """Get all tooltips for Power Analysis page"""
    return {**SHARED_TOOLTIPS, **POWER_ANALYSIS_TOOLTIPS}

def get_group_selection_tooltips() -> Dict[str, str]:
    """Get all tooltips for Group Selection page"""
    return {**SHARED_TOOLTIPS, **GROUP_SELECTION_TOOLTIPS}

def get_rebalancer_tooltips() -> Dict[str, str]:
    """Get all tooltips for Rebalancer page"""
    return {**SHARED_TOOLTIPS, **REBALANCER_TOOLTIPS}

def get_results_analysis_tooltips() -> Dict[str, str]:
    """Get all tooltips for Results Analysis page"""
    return {**SHARED_TOOLTIPS, **RESULTS_ANALYSIS_TOOLTIPS}

# For backward compatibility, export as PARAMETER_TOOLTIPS for each module
# This allows existing imports to continue working
PARAMETER_TOOLTIPS = SHARED_TOOLTIPS
