

import numpy as np
import pandas as pd
from typing import List
from scipy.stats import ttest_ind

def evaluate_balance(
    df: pd.DataFrame,
    value_columns: List[str],
    strat_columns: List[str],
    group_column: str,
    group1_name: str,
    group2_name: str,
):
    """
    Evaluate the balance of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    value_columns (List[str]): List of numeric columns to evaluate.
    strat_columns (List[str]): List of categorical columns to evaluate.
    group_column (str): The name of the column containing group labels.
    group1_name (str): The name of the first group.
    group2_name (str): The name of the second group.
    """
    g1 = df[df[group_column] == group1_name]
    g2 = df[df[group_column] == group2_name]

    n1, n2 = len(g1), len(g2)
    group_diff = abs(n1 - n2) / (n1 + n2) if (n1 + n2) > 0 else np.nan

    print("=" * 100)
    print("🔍 Balance Evaluation Summary")
    print("=" * 100)

    # Group size stats
    print(f"\n📊 Group Sizes:")
    print(f"G1 = '{group1_name}' : {n1}")
    print(f"G2 = '{group2_name}' : {n2}")
    print(f"Group Size Difference: {group_diff:.4f}")

    # Numeric stats
    if value_columns:
        print("\n📈 Numeric Balance:")
        print(f"{'Column':<40}{'Mean G1':>12}{'Mean G2':>12}{'SMD':>12}{'p-value':>12}{'Δ(G2−G1)%':>14}")
        print("-" * 110)
        for col in value_columns:
            x1, x2 = g1[col].dropna(), g2[col].dropna()
            if len(x1) < 2 or len(x2) < 2:
                print(f"{col:<40}{'N/A':>12}{'N/A':>12}{'N/A':>12}{'N/A':>12}{'N/A':>14}")
                continue
            m1, m2 = x1.mean(), x2.mean()
            s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
            pooled = np.sqrt((s1**2 + s2**2) / 2)
            smd = abs(m1 - m2) / pooled if pooled else np.nan
            _, p = ttest_ind(x1, x2, equal_var=False)
            rel_diff = ((m2 - m1) / m1) * 100 if m1 != 0 else np.nan
            rel_diff_str = f"{rel_diff:+6.1f}%" if np.isfinite(rel_diff) else "   N/A "

            print(f"{col:<40}{m1:12.3f}{m2:12.3f}{smd:12.3f}{p:12.5f}{rel_diff_str:>14}")

    # Categorical stats
    if strat_columns:
        print("\n📊 Categorical Balance (Total % Imbalance):")
        print(f"{'Column':<40}{'Imbalance (%)':>20}")
        print("-" * 60)
        for col in strat_columns:
            df_copy = df[[group_column, col]].copy()
            df_copy[col] = df_copy[col].fillna("__MISSING__")
            ct = pd.crosstab(df_copy[group_column], df_copy[col], normalize='index')
            for grp in [group1_name, group2_name]:
                if grp not in ct.index:
                    ct.loc[grp] = 0.0
            ct = ct.fillna(0).sort_index()
            abs_diffs = (ct.loc[group1_name] - ct.loc[group2_name]).abs() * 100
            total_imbalance = abs_diffs.sum()
            print(f"{col:<40}{total_imbalance:20.2f}")

    print("=" * 100)


import numpy as np
import pandas as pd
from typing import List
from scipy.stats import ttest_ind

def treatment_effect_cuped(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_col: str,
    group1: str,
    group2: str,
    suffix_pre: str = "_pre",
    suffix_post: str = "_post",
    warn_missing_pre: bool = False,
):
    """
    Compare balance on post-period metrics, using CUPED with matching prefixes + suffixes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with group column, and both pre and post suffix columns for each metric.
    base_metrics : List[str]
        List of metric names (without suffix), e.g. ["revenue", "click_rate"].
    group_col : str
        Column name containing assignment: two groups.
    group1, group2 : str
        Labels for treatment groups (e.g. "control", "treatment").
    suffix_pre, suffix_post : str
        Suffixes to find baseline ("pre") and outcome ("post") columns.
    warn_missing_pre : bool
        If True, print warning if pre-period values missing (will prevent CUPED).
    """

    g1 = df[df[group_col] == group1]
    g2 = df[df[group_col] == group2]
    n1, n2 = len(g1), len(g2)
    size_diff = abs(n1 - n2) / (n1 + n2) if (n1 + n2) > 0 else np.nan

    print("=" * 96)
    print("🔍 Numeric Balance (with CUPED adjustment by suffix)")
    print("=" * 96)
    print(f"\n📊 Group sizes:  {group1:<10} {n1:>7n}   {group2:<10} {n2:>7n}")
    print(f"Group size difference rate = {size_diff:.4f}")
    print(f"G1 = '{group1}', G2 = '{group2}'")

    print("\n" + f"{'Metric':<30}{'Mean G1':>10}{'Mean G2':>10}{'SMD':>10}{'p-val':>12}"
          + f"{' r':>8}{'VarRed%':>10}{'Δ(G2−G1)%':>14}")
    print("-" * 104)

    for metric in base_metrics:
        post = metric + suffix_post
        pre = metric + suffix_pre
        if post not in df.columns:
            print(f"{metric:<30} -post-missing")
            continue

        x1 = g1[post].dropna()
        x2 = g2[post].dropna()

        if len(x1) < 2 or len(x2) < 2:
            print(f"{metric:<30} insufficient-post")
            continue

        m1, m2 = x1.mean(), x2.mean()
        s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
        pooled = np.sqrt((s1**2 + s2**2) / 2)
        smd = abs(m1 - m2) / pooled if pooled else np.nan
        _, pval = ttest_ind(x1, x2, equal_var=False)

        # Initialize optional output columns
        r = var_red = ""
        rel_diff = np.nan
        if m1 != 0:
            rel_diff = ((m2 - m1) / m1) * 100

        if pre in df.columns:
            pre_vals = df[pre]
            if warn_missing_pre and pre_vals.isna().any():
                print(f"⚠ warning: missing in pre for '{metric}'")

            df_sub = df[[pre, post]].dropna()
            X, Y = df_sub[pre].values, df_sub[post].values
            if len(X) >= 2:
                cov = np.cov(Y, X, ddof=1)[0, 1]
                var_x = np.var(X, ddof=1)
                theta = cov / var_x if var_x else 0.0
                x_bar = X.mean()
                Y_c = df[post] - theta * (df[pre] - x_bar)

                y1c = Y_c[df[group_col] == group1].dropna()
                y2c = Y_c[df[group_col] == group2].dropna()

                m1c, m2c = y1c.mean(), y2c.mean()
                s1c, s2c = y1c.std(ddof=1), y2c.std(ddof=1)
                pooled_c = np.sqrt((s1c**2 + s2c**2) / 2)
                smd_c = abs(m1c - m2c) / pooled_c if pooled_c else np.nan
                _, pval_c = ttest_ind(y1c, y2c, equal_var=False)

                r = np.corrcoef(X, Y)[0, 1]
                if np.isfinite(r):
                    var_red = (1 - r**2) * 100

                # override with adjusted stats
                m1, m2, smd, pval = m1c, m2c, smd_c, pval_c
                rel_diff = ((m2 - m1) / m1) * 100 if m1 != 0 else np.nan

        # Format output line
        rel_diff_str = f"{rel_diff:+6.1f}%" if np.isfinite(rel_diff) else "   N/A "

        print(f"{metric:<30}{m1:10.3f}{m2:10.3f}{smd:10.3f}{pval:12.4g}"
              + (f"{r:8.3f}{var_red:10.0f}%" if r != "" else " " * 18)
              + f"{rel_diff_str:>14}")

    print("=" * 96)


def treatment_effect_did(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_col: str,
    group1: str,
    group2: str,
    suffix_pre: str = "_aa",
    suffix_post: str = "_ab",
):
    """
    For each metric, compute the difference between groups in pre and post periods, and the change in that difference.
    Prints a summary table.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with group column, and columns for each metric+suffix.
    base_metrics : List[str]
        List of metric names (without suffix).
    group_col : str
        Column name containing assignment: two groups.
    group1, group2 : str
        Labels for treatment groups (e.g. "control", "treatment").
    suffix_pre, suffix_post : str
        Suffixes for pre and post columns (e.g. "_aa", "_ab").
    """
    g1 = df[df[group_col] == group1]
    g2 = df[df[group_col] == group2]
    n1, n2 = len(g1), len(g2)
    size_diff = abs(n1 - n2) / (n1 + n2) if (n1 + n2) > 0 else np.nan

    print("=" * 120)
    print("🔍 Group Gap Change Analysis (Difference of Group Differences)")
    print("=" * 120)
    print(f"\n📊 Group sizes:  {group1:<10} {n1:>7n}   {group2:<10} {n2:>7n}")
    print(f"Group size difference rate = {size_diff:.4f}")
    print(f"G1 = '{group1}', G2 = '{group2}'")

    print("\n" + f"{'Metric':<40}{'G2-G1 pre':>14}{'G2-G1 post':>14}{'Δ gap':>12}{'p-val':>12}"
          + f"{'G1 pre':>10}{'G1 post':>10}{'G2 pre':>10}{'G2 post':>10}")
    print("-" * 120)

    for metric in base_metrics:
        pre_col = metric + suffix_pre
        post_col = metric + suffix_post

        # Check columns exist
        for col in [pre_col, post_col]:
            if col not in df.columns:
                print(f"{metric:<40} missing column: {col}")
                continue

        g1_pre_mean = g1[pre_col].mean() if pre_col in g1.columns else np.nan
        g1_post_mean = g1[post_col].mean() if post_col in g1.columns else np.nan
        g2_pre_mean = g2[pre_col].mean() if pre_col in g2.columns else np.nan
        g2_post_mean = g2[post_col].mean() if post_col in g2.columns else np.nan

        diff_pre = g2_pre_mean - g1_pre_mean if np.isfinite(g2_pre_mean) and np.isfinite(g1_pre_mean) else np.nan
        diff_post = g2_post_mean - g1_post_mean if np.isfinite(g2_post_mean) and np.isfinite(g1_post_mean) else np.nan
        delta_gap = diff_post - diff_pre if np.isfinite(diff_post) and np.isfinite(diff_pre) else np.nan

        # For p-value, use the difference scores (post - pre) for each group, then compare those between groups
        g1_change = (g1[post_col] - g1[pre_col]).dropna()
        g2_change = (g2[post_col] - g2[pre_col]).dropna()
        if len(g1_change) > 1 and len(g2_change) > 1:
            _, pval = ttest_ind(g2_change, g1_change, equal_var=False)
        else:
            pval = np.nan

        print(f"{metric:<40}{diff_pre:14.3f}{diff_post:14.3f}{delta_gap:12.3f}{pval:12.4g}"
              f"{g1_pre_mean:10.3f}{g1_post_mean:10.3f}{g2_pre_mean:10.3f}{g2_post_mean:10.3f}")

    print("=" * 120)