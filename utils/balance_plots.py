"""
Plotly-based balance report visualization for Streamlit
Similar style to power analysis plots with interactive features
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
from scipy.stats import ttest_ind
from utils.data_filtering import is_id_column
from utils.stats import smd as _smd


def _pairwise_matrix(groups, fn):
    """Create pairwise matrix"""
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat


def create_balance_report_plotly(
    df: pd.DataFrame,
    value_columns: List[str],
    strat_columns: List[str],
    group_column: str,
    title: str = "Balance Report"
) -> go.Figure:
    """
    Create a comprehensive interactive Plotly balance report.
    
    Args:
        df: DataFrame with group assignments
        value_columns: List of numeric columns to analyze
        strat_columns: List of categorical columns to analyze
        group_column: Name of the column containing group assignments
        title: Overall title for the report
        
    Returns:
        Plotly Figure object
    """
    # Validate inputs
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataframe")
    
    if len(df) == 0:
        raise ValueError("Dataframe is empty")
    
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups, found {n_groups}")
    
    # Calculate number of rows needed
    # Each numeric column gets 1 row with 4 plots
    # Each categorical column gets 1 row with 1 plot
    n_numeric_rows = sum(1 for col in value_columns if col in df.columns)
    n_categorical_rows = sum(1 for col in strat_columns if col in df.columns)
    total_rows = n_numeric_rows + n_categorical_rows
    
    if total_rows == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No columns selected for balance analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Determine max columns (4 for numeric, 1 for categorical)
    max_cols = 4 if n_numeric_rows > 0 else 1
    
    # Build specs and titles
    specs = []
    subplot_titles = []
    
    # Add numeric column rows (4 columns each)
    for col in value_columns:
        if col not in df.columns:
            continue
        specs.append([{"type": "bar"}, {"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}])
        subplot_titles.extend([
            f"{col}: Mean by Group",
            f"{col}: Δ Mean (%)",
            f"{col}: SMD",
            f"{col}: p-value"
        ])
    
    # Add categorical column rows (stacked bar spanning 3 cols + heatmap in 4th col)
    for col in strat_columns:
        if col not in df.columns:
            continue
        # Stacked bar chart spans 3 columns, heatmap in 4th column
        row_spec = [
            {"type": "bar", "colspan": 3},
            None,  # Placeholder for colspan
            None,  # Placeholder for colspan
            {"type": "heatmap"}
        ]
        specs.append(row_spec)
        subplot_titles.extend([
            f"{col}: Distribution by Group",
            "",  # Placeholder for colspan
            "",  # Placeholder for colspan
            f"{col}: Pairwise Imbalance (%)"
        ])
    
    # Create figure with subplots
    fig = make_subplots(
        rows=total_rows,
        cols=max_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.1,
        horizontal_spacing=0.12
    )
    
    row_idx = 1
    
    # =========================
    # Numeric columns
    # =========================
    for col in value_columns:
        if col not in df.columns:
            continue
        
        # Skip ID columns (all values unique)
        if is_id_column(df, col):
            continue
        
        # Filter valid groups (with sufficient data)
        valid_groups = []
        group_means = {}
        for g in groups:
            group_data = df[df[group_column] == g][col].dropna()
            if len(group_data) >= 2:
                valid_groups.append(g)
                group_means[g] = group_data.mean()
        
        if len(valid_groups) < 2:
            row_idx += 1
            continue
        
        # Plot 1: Bar chart - means
        fig.add_trace(
            go.Bar(
                x=valid_groups,
                y=[group_means[g] for g in valid_groups],
                name=f"{col} Mean",
                marker_color='steelblue',
                showlegend=False
            ),
            row=row_idx, col=1
        )
        
        # Plot 2: Heatmap - delta mean %
        def delta_mean(g1, g2):
            m1 = group_means.get(g1, np.nan)
            m2 = group_means.get(g2, np.nan)
            return ((m2 - m1) / m1 * 100) if m1 != 0 and not np.isnan(m1) and not np.isnan(m2) else np.nan
        
        delta_mat = _pairwise_matrix(valid_groups, delta_mean)
        
        fig.add_trace(
            go.Heatmap(
                z=delta_mat.values,
                x=delta_mat.columns,
                y=delta_mat.index,
                colorscale='RdBu',
                reversescale=True,
                text=delta_mat.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=2
        )
        
        # Plot 3: Heatmap - SMD
        def smd_func(g1, g2):
            x1 = df[df[group_column] == g1][col].dropna()
            x2 = df[df[group_column] == g2][col].dropna()
            return _smd(x1, x2)
        
        smd_mat = _pairwise_matrix(valid_groups, smd_func)
        
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
            x1 = df[df[group_column] == g1][col].dropna()
            x2 = df[df[group_column] == g2][col].dropna()
            if len(x1) < 2 or len(x2) < 2:
                return np.nan
            try:
                _, p = ttest_ind(x1, x2, equal_var=False)
                return p
            except (ValueError, RuntimeWarning):
                # Handle cases where t-test fails (e.g., all values same, insufficient variance)
                return np.nan
        
        pval_mat = _pairwise_matrix(valid_groups, pval)
        
        # Ensure p-values are in [0, 1] range for consistent color scale
        pval_mat_clipped = pval_mat.clip(0, 1)
        
        fig.add_trace(
            go.Heatmap(
                z=pval_mat_clipped.values,
                x=pval_mat_clipped.columns,
                y=pval_mat_clipped.index,
                colorscale='Viridis',
                text=pval_mat.values,  # Show original values in text
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
    
    # =========================
    # Categorical columns (Stacked bar + Heatmap)
    # =========================
    for col in strat_columns:
        if col not in df.columns:
            continue
        
        # Skip ID columns (all values unique)
        if is_id_column(df, col):
            continue
        
        try:
            tmp = df[[group_column, col]].copy()
            # Convert to string to handle any data type issues
            tmp[col] = tmp[col].astype(str).fillna("__MISSING__")
            tmp[group_column] = tmp[group_column].astype(str)
            
            # Check if we have valid data
            if len(tmp) == 0:
                continue
            
            ct = pd.crosstab(tmp[group_column], tmp[col], normalize="index").fillna(0)
            
            if len(ct) == 0:
                continue
            
            groups = sorted(ct.index.tolist())
            n_groups = len(groups)
            categories = sorted(ct.columns.tolist())
            
            if n_groups < 2:
                continue
            
            # ===== Stacked Bar Chart (col 1, spanning 3 columns) =====
            # Normalize by column (category) so each bar totals 100%
            # This shows what percentage of each category belongs to each group
            ct_normalized = pd.crosstab(tmp[group_column], tmp[col], normalize="columns").fillna(0)
            
            # Create stacked bar chart - one trace per group
            for group in groups:
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=ct_normalized.loc[group].values * 100,  # Convert to percentage
                        name=str(group),
                        showlegend=(row_idx == 1),  # Only show legend for first categorical row
                        hovertemplate=f'<b>{group}</b><br>Category: %{{x}}<br>Percentage: %{{y:.2f}}%<extra></extra>'
                    ),
                    row=row_idx, col=1
                )
            
            # ===== Heatmap (col 4) =====
            # Calculate pairwise imbalance matrix
            imbalance_matrix = pd.DataFrame(0.0, index=groups, columns=groups)
            
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    if i != j:
                        # Pairwise difference: |g1_distribution - g2_distribution|.sum() * 100
                        diff = (ct.loc[g1] - ct.loc[g2]).abs().sum() * 100
                        imbalance_matrix.loc[g1, g2] = diff
            
            # Set diagonal to NaN for cleaner visualization
            for g in groups:
                imbalance_matrix.loc[g, g] = np.nan
            
            fig.add_trace(
                go.Heatmap(
                    z=imbalance_matrix.values,
                    x=imbalance_matrix.columns,
                    y=imbalance_matrix.index,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Imbalance (%)", len=0.4, y=0.5),
                    text=imbalance_matrix.values,
                    texttemplate='%{text:.1f}%',
                    textfont={"size": 10},
                    hovertemplate='%{y} vs %{x}<br>Imbalance: %{z:.2f}%<extra></extra>'
                ),
                row=row_idx, col=4
            )
            
            row_idx += 1
        except Exception as e:
            # Skip this categorical column if there's an error
            continue
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f77b4')
        ),
        height=300 * row_idx,  # Dynamic height based on number of rows
        showlegend=True,  # Show legend for stacked bars
        template='plotly_white',
        barmode='stack'  # Stack bars for categorical columns
    )
    
    # Update axes labels
    current_row = 1
    
    # Update numeric column rows
    for col in value_columns:
        if col not in df.columns:
            continue
        
        # Check if we have valid groups for this column
        valid_groups = []
        for g in groups:
            group_data = df[df[group_column] == g][col].dropna()
            if len(group_data) >= 2:
                valid_groups.append(g)
        
        if len(valid_groups) < 2:
            continue
        
        # Bar chart (col 1)
        fig.update_xaxes(title_text="Group", row=current_row, col=1)
        fig.update_yaxes(title_text="Mean", row=current_row, col=1)
        
        # Heatmaps (cols 2-4)
        fig.update_xaxes(title_text="Group", row=current_row, col=2)
        fig.update_yaxes(title_text="Group", row=current_row, col=2)
        fig.update_xaxes(title_text="Group", row=current_row, col=3)
        fig.update_yaxes(title_text="Group", row=current_row, col=3)
        fig.update_xaxes(title_text="Group", row=current_row, col=4)
        fig.update_yaxes(title_text="Group", row=current_row, col=4)
        
        current_row += 1
    
    # Update categorical column rows (stacked bar + heatmap)
    for col in strat_columns:
        if col not in df.columns:
            continue
        
        # Stacked bar chart (spans cols 1-3)
        fig.update_xaxes(title_text="Category", row=current_row, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=current_row, col=1)
        
        # Heatmap (col 4)
        fig.update_xaxes(title_text="Group", row=current_row, col=4)
        fig.update_yaxes(title_text="Group", row=current_row, col=4)
        
        current_row += 1
    
    return fig
