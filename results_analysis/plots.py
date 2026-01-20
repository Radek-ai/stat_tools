"""
Plotly visualizations for treatment effect analysis
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
from scipy.stats import ttest_ind


def _smd(x1, x2):
    """Calculate Standardized Mean Difference"""
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    pooled = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return abs(x1.mean() - x2.mean()) / pooled if pooled else np.nan


def _cuped_adjust(pre, post):
    """CUPED adjustment"""
    X = pre.values
    Y = post.values
    var_x = np.var(X, ddof=1)
    if var_x == 0:
        return None
    theta = np.cov(Y, X, ddof=1)[0, 1] / var_x
    return post - theta * (pre - X.mean())


def _pairwise_matrix(groups, fn):
    """Create pairwise matrix"""
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat


def create_basic_analysis_plotly(
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
    
    for col in valid_columns:
        # Calculate statistics for all groups
        group_means = {}
        group_data = {}
        
        for g in groups:
            data = df[df[group_column] == g][col].dropna()
            if len(data) >= 2:
                group_means[g] = data.mean()
                group_data[g] = data
        
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


def create_cuped_analysis_plotly(
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
        
        for g in groups:
            data = adj[df[group_column] == g].dropna()
            if len(data) >= 2:
                cuped_means[g] = data.mean()
                cuped_data[g] = data
        
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


def create_did_analysis_plotly(
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
        
        for g in groups:
            gdf = df[df[group_column] == g]
            pre_means[g] = gdf[pre_col].mean()
            post_means[g] = gdf[post_col].mean()
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
        
        # Column 4: DiD gap heatmap
        def did_gap(g1, g2):
            g1df = df[df[group_column] == g1]
            g2df = df[df[group_column] == g2]
            
            d_pre = g2df[pre_col].mean() - g1df[pre_col].mean()
            d_post = g2df[post_col].mean() - g1df[post_col].mean()
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
