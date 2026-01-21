
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List
from scipy.stats import ttest_ind
from utils.stats import smd as _smd, cuped_adjust as _cuped_adjust


# ============================================================
# -------------------- Common helpers ------------------------
# ============================================================
def _pairwise_matrix(groups, fn):
    """
    Full pairwise matrix (no masking).
    Entry (i, j) = fn(i, j)
    """
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat


def _plot_heatmap(mat, title, cmap, ax):
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        ax=ax,
        # cbar=True,
    )
    ax.set_title(title)


def _plot_bar(series, title, ax, ylabel=None):
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Group")
    if ylabel:
        ax.set_ylabel(ylabel)

# ============================================================
# -------- Multigroup balance (numeric + categorical) --------
# ============================================================
def multigroup_balance_plots(
    df: pd.DataFrame,
    value_columns: List[str],
    strat_columns: List[str],
    group_column: str,
    figsize=(16, 5),
):
    groups = df[group_column].unique()

    # =========================
    # Numeric columns
    # =========================
    for col in value_columns:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # ---- Bar chart: means
        means = df.groupby(group_column)[col].mean()
        _plot_bar(means, f"{col}: Mean by group", axes[0], ylabel="Mean")

        # ---- Heatmap: delta mean %
        def delta_mean(g1, g2):
            m1 = df[df[group_column] == g1][col].mean()
            m2 = df[df[group_column] == g2][col].mean()
            return ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan

        delta_mat = _pairwise_matrix(groups, delta_mean)
        _plot_heatmap(delta_mat, f"{col}: Δ Mean (%)", "RdBu_r", axes[1])

        # ---- Heatmap: p-values
        def pval(g1, g2):
            x1 = df[df[group_column] == g1][col].dropna()
            x2 = df[df[group_column] == g2][col].dropna()
            if len(x1) < 2 or len(x2) < 2:
                return np.nan
            return ttest_ind(x1, x2, equal_var=False).pvalue

        pval_mat = _pairwise_matrix(groups, pval)
        _plot_heatmap(pval_mat, f"{col}: p-value", "viridis", axes[2])

        plt.tight_layout()
        plt.show()

    # =========================
    # Categorical columns
    # =========================
    for col in strat_columns:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        tmp = df[[group_column, col]].copy()
        tmp[col] = tmp[col].fillna("__MISSING__")
        ct = pd.crosstab(tmp[group_column], tmp[col], normalize="index").fillna(0)
        overall = ct.mean(axis=0)

        imbalance = (ct.sub(overall, axis=1).abs().sum(axis=1)) * 100
        _plot_bar(imbalance, f"{col}: Total imbalance (%)", ax, ylabel="%")

        plt.tight_layout()
        plt.show()


# ============================================================
# -------------------- CUPED diagnostics ---------------------
# ============================================================
def multigroup_cuped_plots(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_col: str,
    suffix_pre="_pre",
    suffix_post="_post",
    figsize=(16, 5),
):
    groups = df[group_col].unique()

    for metric in base_metrics:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        pre = metric + suffix_pre
        post = metric + suffix_post

        if pre not in df.columns or post not in df.columns:
            plt.close(fig)
            continue

        # ---- CUPED-adjusted means
        adj = _cuped_adjust(df[pre], df[post])
        if adj is None:
            plt.close(fig)
            continue

        means = (
            pd.DataFrame({group_col: df[group_col], "Yc": adj})
            .groupby(group_col)["Yc"]
            .mean()
        )
        _plot_bar(means, f"{metric}: CUPED mean", axes[0], ylabel="Adjusted mean")

        # ---- CUPED SMD heatmap
        def cuped_smd(g1, g2):
            y1 = adj[df[group_col] == g1].dropna()
            y2 = adj[df[group_col] == g2].dropna()
            return _smd(y1, y2)

        smd_mat = _pairwise_matrix(groups, cuped_smd)
        _plot_heatmap(smd_mat, f"{metric}: CUPED SMD", "Reds", axes[1])

        # ---- CUPED p-value heatmap
        def cuped_pval(g1, g2):
            y1 = adj[df[group_col] == g1].dropna()
            y2 = adj[df[group_col] == g2].dropna()
            if len(y1) < 2 or len(y2) < 2:
                return np.nan
            return ttest_ind(y1, y2, equal_var=False).pvalue

        pval_mat = _pairwise_matrix(groups, cuped_pval)
        _plot_heatmap(pval_mat, f"{metric}: CUPED p-value", "viridis", axes[2])

        plt.tight_layout()
        plt.show()


# ============================================================
# ---------------- Difference-in-Differences -----------------
# ============================================================
def multigroup_did_heatmaps(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_col: str,
    suffix_pre="_aa",
    suffix_post="_ab",
    figsize=(6, 5),
):
    groups = df[group_col].unique()

    for metric in base_metrics:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        def did_gap(g1, g2):
            g1df = df[df[group_col] == g1]
            g2df = df[df[group_col] == g2]
            d_pre = g2df[metric + suffix_pre].mean() - g1df[metric + suffix_pre].mean()
            d_post = g2df[metric + suffix_post].mean() - g1df[metric + suffix_post].mean()
            return d_post - d_pre

        mat = _pairwise_matrix(groups, did_gap)
        _plot_heatmap(mat, f"{metric}: Δ DiD gap", "RdBu_r", ax)

        plt.tight_layout()
        plt.show()

# ============================================================
# ------------------- 2-Group Numeric Balance ----------------
# ============================================================
def evaluate_balance(
    df: pd.DataFrame,
    value_columns: List[str],
    strat_columns: List[str],
    group_column: str,
    group1_name: str,
    group2_name: str,
):
    """
    Evaluate numeric and categorical balance for two groups using standardized helpers.
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
            smd_val = _smd(x1, x2)
            _, p = ttest_ind(x1, x2, equal_var=False)
            rel_diff = ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan
            rel_diff_str = f"{rel_diff:+6.1f}%" if np.isfinite(rel_diff) else "   N/A "
            print(f"{col:<40}{m1:12.3f}{m2:12.3f}{smd_val:12.3f}{p:12.5f}{rel_diff_str:>14}")

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


# ============================================================
# ------------------- 2-Group CUPED Diagnostics -------------
# ============================================================
def treatment_effect_cuped(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_col: str,
    group1: str,
    group2: str,
    suffix_pre: str = "_pre",
    suffix_post: str = "_post",
):
    """
    Compute CUPED-adjusted means, SMD, p-values for two groups using shared helpers.
    """
    g1 = df[df[group_col] == group1]
    g2 = df[df[group_col] == group2]

    n1, n2 = len(g1), len(g2)
    size_diff = abs(n1 - n2) / (n1 + n2) if (n1 + n2) > 0 else np.nan

    print("=" * 96)
    print("🔍 Numeric Balance (with CUPED adjustment)")
    print("=" * 96)
    print(f"\n📊 Group sizes:  {group1:<10} {n1:>7n}   {group2:<10} {n2:>7n}")
    print(f"Group size difference rate = {size_diff:.4f}")
    print(f"G1 = '{group1}', G2 = '{group2}'")
    print("\n" + f"{'Metric':<30}{'Mean G1':>10}{'Mean G2':>10}{'SMD':>10}{'p-val':>12}{'Δ(G2−G1)%':>14}")
    print("-" * 104)

    for metric in base_metrics:
        post = metric + suffix_post
        pre = metric + suffix_pre

        if post not in df.columns:
            print(f"{metric:<30} -post-missing")
            continue

        # Raw post metrics
        x1 = g1[post].dropna()
        x2 = g2[post].dropna()
        if len(x1) < 2 or len(x2) < 2:
            print(f"{metric:<30} insufficient-post")
            continue

        m1_raw, m2_raw = x1.mean(), x2.mean()
        smd_raw = _smd(x1, x2)
        _, pval_raw = ttest_ind(x1, x2, equal_var=False)

        # CUPED-adjusted
        if pre in df.columns:
            adj = _cuped_adjust(df[pre], df[post])
            if adj is not None:
                y1 = adj[df[group_col] == group1].dropna()
                y2 = adj[df[group_col] == group2].dropna()
                if len(y1) >= 2 and len(y2) >= 2:
                    m1, m2 = y1.mean(), y2.mean()
                    smd_val = _smd(y1, y2)
                    _, pval = ttest_ind(y1, y2, equal_var=False)
                else:
                    m1, m2, smd_val, pval = m1_raw, m2_raw, smd_raw, pval_raw
            else:
                m1, m2, smd_val, pval = m1_raw, m2_raw, smd_raw, pval_raw
        else:
            m1, m2, smd_val, pval = m1_raw, m2_raw, smd_raw, pval_raw

        rel_diff = ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan
        rel_diff_str = f"{rel_diff:+6.1f}%" if np.isfinite(rel_diff) else "   N/A "

        print(f"{metric:<30}{m1:10.3f}{m2:10.3f}{smd_val:10.3f}{pval:12.4g}{rel_diff_str:>14}")

    print("=" * 96)


# ============================================================
# ------------------- 2-Group DiD Diagnostics ----------------
# ============================================================
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
    Compute difference-in-differences for two groups using shared helpers.
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
          f"{'G1 pre':>10}{'G1 post':>10}{'G2 pre':>10}{'G2 post':>10}")
    print("-" * 120)

    for metric in base_metrics:
        pre_col = metric + suffix_pre
        post_col = metric + suffix_post

        g1_pre_mean = g1[pre_col].mean() if pre_col in g1.columns else np.nan
        g1_post_mean = g1[post_col].mean() if post_col in g1.columns else np.nan
        g2_pre_mean = g2[pre_col].mean() if pre_col in g2.columns else np.nan
        g2_post_mean = g2[post_col].mean() if post_col in g2.columns else np.nan

        diff_pre = g2_pre_mean - g1_pre_mean
        diff_post = g2_post_mean - g1_post_mean
        delta_gap = diff_post - diff_pre

        # P-value using difference scores
        g1_change = (g1[post_col] - g1[pre_col]).dropna()
        g2_change = (g2[post_col] - g2[pre_col]).dropna()
        pval = ttest_ind(g2_change, g1_change, equal_var=False).pvalue if len(g1_change) > 1 and len(g2_change) > 1 else np.nan

        print(f"{metric:<40}{diff_pre:14.3f}{diff_post:14.3f}{delta_gap:12.3f}{pval:12.4g}"
              f"{g1_pre_mean:10.3f}{g1_post_mean:10.3f}{g2_pre_mean:10.3f}{g2_post_mean:10.3f}")

    print("=" * 120)