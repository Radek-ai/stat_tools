"""
Plotly visualizations for treatment effect analysis
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict
from scipy.stats import ttest_ind
from utils.stats import smd as _smd, cuped_adjust as _cuped_adjust

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def _pairwise_matrix(groups, fn):
    """Create pairwise matrix"""
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat


def _precompute_group_data(df: pd.DataFrame, group_column: str, columns: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Pre-compute group data arrays for efficient access.
    This avoids repeated DataFrame filtering operations.
    
    Args:
        df: DataFrame with group assignments
        group_column: Name of the column containing group assignments
        columns: List of column names to pre-compute
        
    Returns:
        dict: {group_name: {column_name: numpy_array}}
    """
    groups = sorted(df[group_column].unique())
    group_data = {}
    
    # Pre-compute group indices once
    group_indices = {}
    for g in groups:
        mask = df[group_column] == g
        group_indices[g] = df.index[mask]
    
    # Pre-extract data for each group and column
    for g in groups:
        group_data[g] = {}
        for col in columns:
            if col in df.columns:
                # Use pre-computed indices for fast access
                group_data[g][col] = df.loc[group_indices[g], col].dropna().values
    
    return group_data


def _create_basic_analysis_plotly_impl(
    df: pd.DataFrame,
    value_columns: List[str],
    group_column: str,
    title: str = "Treatment Effect Analysis"
) -> go.Figure:
    """
    Create Plotly visualization for basic treatment effect analysis.
    Shows means, uplifts, SMD, and p-values for each metric.
    """
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 groups for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Filter valid columns
    valid_columns = [col for col in value_columns if col in df.columns]
    
    if len(valid_columns) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid metric columns selected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Create subplots: 4 columns per metric (means, uplifts, SMD, p-values)
    n_rows = len(valid_columns)
    fig = make_subplots(
        rows=n_rows, cols=4,
        subplot_titles=[f"{col}: {subtitle}" for col in valid_columns 
                       for subtitle in ["Mean by Group", "Uplift (%)", "SMD", "p-value"]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    row_idx = 1
    
    # Pre-compute group data for all columns (Issue #1 fix)
    group_data_cache = _precompute_group_data(df, group_column, valid_columns)
    
    for col in valid_columns:
        # Calculate statistics for all groups using cached data
        group_means = {}
        group_data = {}
        
        for g in groups:
            if g in group_data_cache and col in group_data_cache[g]:
                data_array = group_data_cache[g][col]
                if len(data_array) >= 2:
                    group_means[g] = np.mean(data_array)
                    group_data[g] = data_array
        
        if len(group_means) < 2:
            row_idx += 1
            continue
        
        # Plot 1: Bar chart - Means
        fig.add_trace(
            go.Bar(
                x=list(group_means.keys()),
                y=list(group_means.values()),
                name=f"{col} Mean",
                marker_color='steelblue',
                showlegend=False
            ),
            row=row_idx, col=1
        )
        
        # Plot 2: Heatmap - Uplift %
        def uplift(g1, g2):
            if g1 in group_means and g2 in group_means and group_means[g1] != 0:
                return ((group_means[g2] - group_means[g1]) / group_means[g1] * 100)
            return np.nan
        
        uplift_mat = _pairwise_matrix(groups, uplift)
        
        fig.add_trace(
            go.Heatmap(
                z=uplift_mat.values,
                x=uplift_mat.columns,
                y=uplift_mat.index,
                colorscale='RdBu_r',
                text=uplift_mat.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=2
        )
        
        # Plot 3: Heatmap - SMD
        def smd_fn(g1, g2):
            if g1 in group_data and g2 in group_data:
                return _smd(group_data[g1], group_data[g2])
            return np.nan
        
        smd_mat = _pairwise_matrix(groups, smd_fn)
        
        fig.add_trace(
            go.Heatmap(
                z=smd_mat.values,
                x=smd_mat.columns,
                y=smd_mat.index,
                colorscale='Reds',
                text=smd_mat.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=3
        )
        
        # Plot 4: Heatmap - p-values
        def pval(g1, g2):
            if g1 in group_data and g2 in group_data:
                x1, x2 = group_data[g1], group_data[g2]
                if len(x1) < 2 or len(x2) < 2:
                    return np.nan
                try:
                    _, p = ttest_ind(x1, x2, equal_var=False)
                    return p
                except:
                    return np.nan
            return np.nan
        
        pval_mat = _pairwise_matrix(groups, pval)
        pval_mat_clipped = pval_mat.clip(0, 1)
        
        fig.add_trace(
            go.Heatmap(
                z=pval_mat_clipped.values,
                x=pval_mat_clipped.columns,
                y=pval_mat_clipped.index,
                colorscale='Viridis',
                text=pval_mat.values,
                texttemplate='%{text:.4f}',
                textfont={"size": 10},
                zmin=0,
                zmax=1,
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=4
        )
        
        row_idx += 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f77b4')
        ),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Group", row=i, col=1)
        fig.update_yaxes(title_text="Mean", row=i, col=1)
        for j in [2, 3, 4]:
            fig.update_xaxes(title_text="Group", row=i, col=j)
            fig.update_yaxes(title_text="Group", row=i, col=j)
    
    return fig


# Cached version for Streamlit (Issue #4 fix)
if STREAMLIT_AVAILABLE:
    @st.cache_data(show_spinner=False)
    def _create_basic_analysis_plotly_cached(
        df: pd.DataFrame,
        value_cols: tuple,
        group_col: str,
        plot_title: str
    ) -> go.Figure:
        """Cached version that Streamlit can use."""
        return _create_basic_analysis_plotly_impl(df, list(value_cols), group_col, plot_title)


def create_basic_analysis_plotly(
    df: pd.DataFrame,
    value_columns: List[str],
    group_column: str,
    title: str = "Treatment Effect Analysis",
    _use_cache: bool = True
) -> go.Figure:
    """
    Create Plotly visualization for basic treatment effect analysis.
    Shows means, uplifts, SMD, and p-values for each metric.
    
    Args:
        df: DataFrame with group assignments and metrics
        value_columns: List of numeric columns to analyze
        group_column: Name of the column containing group assignments
        title: Overall title for the plot
        _use_cache: Whether to use Streamlit caching (default: True)
        
    Returns:
        Plotly Figure object
    """
    # Use caching if Streamlit is available and caching is enabled
    if STREAMLIT_AVAILABLE and _use_cache:
        return _create_basic_analysis_plotly_cached(
            df,
            tuple(value_columns),
            group_column,
            title
        )
    else:
        # No caching, call directly
        return _create_basic_analysis_plotly_impl(df, value_columns, group_column, title)


def _create_cuped_analysis_plotly_impl(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    suffix_pre: str = "_pre",
    suffix_post: str = "_post",
    title: str = "CUPED-Adjusted Treatment Effect Analysis"
) -> go.Figure:
    """
    Create Plotly visualization for CUPED-adjusted analysis.
    """
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 groups for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Filter valid metrics (must have both pre and post columns
    valid_metrics = []
    for metric in base_metrics:
        pre_col = metric + suffix_pre
        post_col = metric + suffix_post
        if pre_col in df.columns and post_col in df.columns:
            valid_metrics.append(metric)
    
    if len(valid_metrics) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid metrics found. Expected columns with suffixes '{suffix_pre}' and '{suffix_post}'",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Create subplots: 3 columns per metric (CUPED means, SMD, p-values)
    n_rows = len(valid_metrics)
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f"{metric}: {subtitle}" for metric in valid_metrics 
                       for subtitle in ["CUPED Mean", "CUPED SMD", "CUPED p-value"]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    row_idx = 1
    
    for metric in valid_metrics:
        pre_col = metric + suffix_pre
        post_col = metric + suffix_post
        
        # CUPED adjustment
        adj = _cuped_adjust(df[pre_col], df[post_col])
        if adj is None:
            row_idx += 1
            continue
        
        # Calculate CUPED-adjusted statistics by group
        cuped_means = {}
        cuped_data = {}
        
        # Pre-compute group indices for CUPED data
        group_indices = {}
        for g in groups:
            mask = df[group_column] == g
            group_indices[g] = df.index[mask]
        
        for g in groups:
            # Use pre-computed indices for fast access
            data_array = adj.loc[group_indices[g]].dropna().values
            if len(data_array) >= 2:
                cuped_means[g] = np.mean(data_array)
                cuped_data[g] = data_array
        
        if len(cuped_means) < 2:
            row_idx += 1
            continue
        
        # Plot 1: Bar chart - CUPED-adjusted means
        fig.add_trace(
            go.Bar(
                x=list(cuped_means.keys()),
                y=list(cuped_means.values()),
                name=f"{metric} CUPED Mean",
                marker_color='green',
                showlegend=False
            ),
            row=row_idx, col=1
        )
        
        # Plot 2: Heatmap - CUPED SMD
        def cuped_smd(g1, g2):
            if g1 in cuped_data and g2 in cuped_data:
                return _smd(cuped_data[g1], cuped_data[g2])
            return np.nan
        
        smd_mat = _pairwise_matrix(groups, cuped_smd)
        
        fig.add_trace(
            go.Heatmap(
                z=smd_mat.values,
                x=smd_mat.columns,
                y=smd_mat.index,
                colorscale='Reds',
                text=smd_mat.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=2
        )
        
        # Plot 3: Heatmap - CUPED p-values
        def cuped_pval(g1, g2):
            if g1 in cuped_data and g2 in cuped_data:
                y1, y2 = cuped_data[g1], cuped_data[g2]
                if len(y1) < 2 or len(y2) < 2:
                    return np.nan
                try:
                    _, p = ttest_ind(y1, y2, equal_var=False)
                    return p
                except:
                    return np.nan
            return np.nan
        
        pval_mat = _pairwise_matrix(groups, cuped_pval)
        pval_mat_clipped = pval_mat.clip(0, 1)
        
        fig.add_trace(
            go.Heatmap(
                z=pval_mat_clipped.values,
                x=pval_mat_clipped.columns,
                y=pval_mat_clipped.index,
                colorscale='Viridis',
                text=pval_mat.values,
                texttemplate='%{text:.4f}',
                textfont={"size": 10},
                zmin=0,
                zmax=1,
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=3
        )
        
        row_idx += 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f77b4')
        ),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Group", row=i, col=1)
        fig.update_yaxes(title_text="CUPED Mean", row=i, col=1)
        for j in [2, 3]:
            fig.update_xaxes(title_text="Group", row=i, col=j)
            fig.update_yaxes(title_text="Group", row=i, col=j)
    
    return fig


# Cached version for Streamlit (Issue #4 fix)
if STREAMLIT_AVAILABLE:
    @st.cache_data(show_spinner=False)
    def _create_cuped_analysis_plotly_cached(
        df: pd.DataFrame,
        base_metrics: tuple,
        group_col: str,
        suffix_pre: str,
        suffix_post: str,
        plot_title: str
    ) -> go.Figure:
        """Cached version that Streamlit can use."""
        return _create_cuped_analysis_plotly_impl(df, list(base_metrics), group_col, suffix_pre, suffix_post, plot_title)


def create_cuped_analysis_plotly(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    suffix_pre: str = "_pre",
    suffix_post: str = "_post",
    title: str = "CUPED-Adjusted Treatment Effect Analysis",
    _use_cache: bool = True
) -> go.Figure:
    """
    Create Plotly visualization for CUPED-adjusted analysis.
    
    Args:
        df: DataFrame with group assignments and pre/post metrics
        base_metrics: List of base metric names (without suffixes)
        group_column: Name of the column containing group assignments
        suffix_pre: Suffix for pre-experiment columns
        suffix_post: Suffix for post-experiment columns
        title: Overall title for the plot
        _use_cache: Whether to use Streamlit caching (default: True)
        
    Returns:
        Plotly Figure object
    """
    # Use caching if Streamlit is available and caching is enabled
    if STREAMLIT_AVAILABLE and _use_cache:
        return _create_cuped_analysis_plotly_cached(
            df,
            tuple(base_metrics),
            group_column,
            suffix_pre,
            suffix_post,
            title
        )
    else:
        # No caching, call directly
        return _create_cuped_analysis_plotly_impl(df, base_metrics, group_column, suffix_pre, suffix_post, title)


def _create_did_analysis_plotly_impl(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    suffix_pre: str = "_aa",
    suffix_post: str = "_ab",
    title: str = "Difference-in-Differences Analysis"
) -> go.Figure:
    """
    Create Plotly visualization for DiD analysis.
    """
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 groups for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Filter valid metrics
    valid_metrics = []
    for metric in base_metrics:
        pre_col = metric + suffix_pre
        post_col = metric + suffix_post
        if pre_col in df.columns and post_col in df.columns:
            valid_metrics.append(metric)
    
    if len(valid_metrics) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid metrics found. Expected columns with suffixes '{suffix_pre}' and '{suffix_post}'",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Create subplots: 4 columns per metric
    # Column 1: Pre-period means (bar chart)
    # Column 2: Post-period means (bar chart)
    # Column 3: % Change (bar chart)
    # Column 4: DiD gap heatmap
    n_rows = len(valid_metrics)
    fig = make_subplots(
        rows=n_rows, cols=4,
        subplot_titles=[f"{metric}: {subtitle}" for metric in valid_metrics 
                       for subtitle in ["Pre-Period Mean", "Post-Period Mean", "% Change", "Δ DiD Gap"]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    row_idx = 1
    
    for metric in valid_metrics:
        pre_col = metric + suffix_pre
        post_col = metric + suffix_post
        
        # Calculate means by group
        pre_means = {}
        post_means = {}
        pct_changes = {}
        
        # Pre-compute group indices once
        group_indices = {}
        for g in groups:
            mask = df[group_column] == g
            group_indices[g] = df.index[mask]
        
        for g in groups:
            # Use pre-computed indices for fast access
            g_indices = group_indices[g]
            pre_means[g] = df.loc[g_indices, pre_col].mean()
            post_means[g] = df.loc[g_indices, post_col].mean()
            # Calculate % change
            if pre_means[g] != 0:
                pct_changes[g] = ((post_means[g] - pre_means[g]) / pre_means[g]) * 100
            else:
                pct_changes[g] = 0
        
        # Column 1: Pre-period means (bar chart)
        fig.add_trace(
            go.Bar(
                x=list(pre_means.keys()),
                y=list(pre_means.values()),
                name=f"{metric} Pre",
                marker_color='lightblue',
                showlegend=False
            ),
            row=row_idx, col=1
        )
        
        # Column 2: Post-period means (bar chart)
        fig.add_trace(
            go.Bar(
                x=list(post_means.keys()),
                y=list(post_means.values()),
                name=f"{metric} Post",
                marker_color='lightgreen',
                showlegend=False
            ),
            row=row_idx, col=2
        )
        
        # Column 3: % Change (bar chart)
        fig.add_trace(
            go.Bar(
                x=list(pct_changes.keys()),
                y=list(pct_changes.values()),
                name=f"{metric} % Change",
                marker_color='orange',
                showlegend=False
            ),
            row=row_idx, col=3
        )
        
        # Column 4: DiD gap heatmap (using pre-computed indices)
        def did_gap(g1, g2):
            g1_indices = group_indices[g1]
            g2_indices = group_indices[g2]
            
            d_pre = df.loc[g2_indices, pre_col].mean() - df.loc[g1_indices, pre_col].mean()
            d_post = df.loc[g2_indices, post_col].mean() - df.loc[g1_indices, post_col].mean()
            return d_post - d_pre
        
        did_mat = _pairwise_matrix(groups, did_gap)
        
        fig.add_trace(
            go.Heatmap(
                z=did_mat.values,
                x=did_mat.columns,
                y=did_mat.index,
                colorscale='RdBu_r',
                text=did_mat.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                showscale=(row_idx == 1),  # Only show scale for first row
                showlegend=False
            ),
            row=row_idx, col=4
        )
        
        row_idx += 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f77b4')
        ),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes
    for i in range(1, n_rows + 1):
        # Column 1: Pre-period means
        fig.update_xaxes(title_text="Group", row=i, col=1)
        fig.update_yaxes(title_text="Pre-Period Mean", row=i, col=1)
        
        # Column 2: Post-period means
        fig.update_xaxes(title_text="Group", row=i, col=2)
        fig.update_yaxes(title_text="Post-Period Mean", row=i, col=2)
        
        # Column 3: % Change
        fig.update_xaxes(title_text="Group", row=i, col=3)
        fig.update_yaxes(title_text="% Change", row=i, col=3)
        
        # Column 4: DiD gap heatmap
        fig.update_xaxes(title_text="Group", row=i, col=4)
        fig.update_yaxes(title_text="Group", row=i, col=4)
    
    return fig


# Cached version for Streamlit (Issue #4 fix)
if STREAMLIT_AVAILABLE:
    @st.cache_data(show_spinner=False)
    def _create_did_analysis_plotly_cached(
        df: pd.DataFrame,
        base_metrics: tuple,
        group_col: str,
        suffix_pre: str,
        suffix_post: str,
        plot_title: str
    ) -> go.Figure:
        """Cached version that Streamlit can use."""
        return _create_did_analysis_plotly_impl(df, list(base_metrics), group_col, suffix_pre, suffix_post, plot_title)


def create_did_analysis_plotly(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    suffix_pre: str = "_aa",
    suffix_post: str = "_ab",
    title: str = "Difference-in-Differences Analysis",
    _use_cache: bool = True
) -> go.Figure:
    """
    Create Plotly visualization for DiD analysis.
    
    Args:
        df: DataFrame with group assignments and pre/post metrics
        base_metrics: List of base metric names (without suffixes)
        group_column: Name of the column containing group assignments
        suffix_pre: Suffix for pre-experiment columns
        suffix_post: Suffix for post-experiment columns
        title: Overall title for the plot
        _use_cache: Whether to use Streamlit caching (default: True)
        
    Returns:
        Plotly Figure object
    """
    # Use caching if Streamlit is available and caching is enabled
    if STREAMLIT_AVAILABLE and _use_cache:
        return _create_did_analysis_plotly_cached(
            df,
            tuple(base_metrics),
            group_column,
            suffix_pre,
            suffix_post,
            title
        )
    else:
        # No caching, call directly
        return _create_did_analysis_plotly_impl(df, base_metrics, group_column, suffix_pre, suffix_post, title)
