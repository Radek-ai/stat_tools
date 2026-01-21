"""
Results display for rebalancer page (plots, evaluation, downloads).
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ttest_ind
from utils.balance_plots import create_balance_report_plotly
from utils.data_filtering import is_id_column
from utils.stats import smd as _smd
from utils.streamlit_errors import handle_error


def render_rebalancing_results(original_df: pd.DataFrame, group_column: str, groups: list) -> None:
    """Render the rebalancing results section"""
    st.divider()
    st.header("📊 Rebalancing Results")
    
    rebalanced_df = st.session_state.rebalanced_data
    config = st.session_state.rebalancing_config
    
    # Get value_columns and strat_columns from config
    value_columns = config.get('value_columns', []) if config else []
    strat_columns = config.get('strat_columns', []) if config else []
    
    # Show group size changes
    st.subheader("📈 Group Size Changes")
    original_sizes = original_df[group_column].value_counts()
    rebalanced_sizes = rebalanced_df[group_column].value_counts()
    
    col_change1, col_change2 = st.columns(2)
    with col_change1:
        st.write("**Original Sizes:**")
        for group_name in groups:
            size = original_sizes.get(group_name, 0)
            st.metric(str(group_name), f"{size:,}")
    
    with col_change2:
        st.write("**Rebalanced Sizes:**")
        for group_name in groups:
            size = rebalanced_sizes.get(group_name, 0)
            change = size - original_sizes.get(group_name, 0)
            st.metric(str(group_name), f"{size:,}", f"{change:+,}")
    
    # Show middle and odd groups (Advanced mode only)
    if config.get('mode') == "Advanced" and (config.get('middle_group') or config.get('odd_group')):
        st.subheader("🎯 Group Strategy")
        col_strat1, col_strat2 = st.columns(2)
        with col_strat1:
            if config.get('middle_group'):
                st.info(f"**Middle Group:** {config['middle_group']}")
        with col_strat2:
            if config.get('odd_group'):
                st.info(f"**Odd Group:** {config['odd_group']}")
    
    # Loss history plot (if available)
    loss_history = config.get('loss_history', [])
    loss_history_runs = config.get('loss_history_runs', [])
    
    if loss_history and len(loss_history) > 1:
        st.subheader("📉 Loss History")
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # If we have separate runs, plot them with different colors and annotations
            if loss_history_runs and len(loss_history_runs) > 1:
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                cumulative_iter = 0
                
                for run_idx, run in enumerate(loss_history_runs):
                    if not run or len(run) == 0:
                        continue
                    
                    color = colors[run_idx % len(colors)]
                    run_iterations = list(range(cumulative_iter, cumulative_iter + len(run)))
                    
                    # Plot this run
                    fig.add_trace(go.Scatter(
                        x=run_iterations,
                        y=run,
                        mode='lines+markers',
                        name=f'Run {run_idx + 1}',
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        showlegend=True
                    ))
                    
                    # Add annotation at end of each run (except last)
                    if run_idx < len(loss_history_runs) - 1 and len(run) > 0:
                        end_iter = cumulative_iter + len(run) - 1
                        end_loss = run[-1]
                        fig.add_annotation(
                            x=end_iter,
                            y=end_loss,
                            text=f"Run {run_idx + 1}<br>End: {end_loss:.4f}",
                            showarrow=False,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor=color,
                            borderwidth=1,
                            font=dict(size=10, color=color)
                        )
                    
                    cumulative_iter += len(run)
                
                # Add vertical lines between runs
                cumulative_iter = 0
                for run_idx, run in enumerate(loss_history_runs[:-1]):  # All but last
                    if not run or len(run) == 0:
                        continue
                    cumulative_iter += len(run)
                    fig.add_vline(
                        x=cumulative_iter - 0.5,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5,
                        annotation_text=f"Run {run_idx + 1} → {run_idx + 2}",
                        annotation_position="top"
                    )
                
                # Add initial loss annotation
                if len(loss_history_runs) > 0 and len(loss_history_runs[0]) > 0:
                    initial_loss = loss_history_runs[0][0]
                    fig.add_annotation(
                        x=0,
                        y=initial_loss,
                        text=f"Initial: {initial_loss:.4f}",
                        showarrow=False,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='green',
                        borderwidth=1,
                        font=dict(size=10, color='green')
                    )
                
                fig.update_layout(
                    title=f"Rebalancing Loss Convergence ({len(loss_history_runs)} runs)",
                    xaxis_title="Iteration",
                    yaxis_title="Total Loss",
                    hovermode='x unified',
                    template='plotly_white',
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
            else:
                # Single run or old format - simple plot
                fig.add_trace(go.Scatter(
                    x=list(range(len(loss_history))),
                    y=loss_history,
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title="Rebalancing Loss Convergence",
                    xaxis_title="Iteration",
                    yaxis_title="Loss",
                    height=400,
                    template='plotly_white'
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Loss metrics
            col_loss1, col_loss2, col_loss3 = st.columns(3)
            with col_loss1:
                st.metric("Initial Loss", f"{loss_history[0]:.4f}")
            with col_loss2:
                st.metric("Final Loss", f"{loss_history[-1]:.4f}")
            with col_loss3:
                improvement_pct = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100) if loss_history[0] > 0 else 0
                num_runs = len(loss_history_runs) if loss_history_runs else 1
                st.metric("Improvement", f"{improvement_pct:.1f}%", f"({num_runs} run{'s' if num_runs > 1 else ''})")
            
        except Exception as e:
            handle_error(e, "Could not plot loss history", show_traceback=True)
    
    # Balance evaluation
    st.divider()
    st.subheader("📊 Rebalanced Group Balance Report")
    
    # View switcher
    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="rebalanced_balance_view_mode"
    )
    
    if view_mode == "Summary":
        # Summary view
        st.markdown("**Numeric Balance Summary:**")
        if value_columns:
            eval_data = []
            groups_sorted = sorted(rebalanced_df[group_column].unique())
            
            for col in value_columns[:10]:  # Limit for performance
                groups_data = {}
                for group_name in groups_sorted:
                    group_df = rebalanced_df[rebalanced_df[group_column] == group_name]
                    groups_data[group_name] = group_df[col].dropna()
                
                # Calculate pairwise statistics
                pairs = []
                for i, g1 in enumerate(groups_sorted):
                    for g2 in groups_sorted[i+1:]:
                        x1, x2 = groups_data[g1], groups_data[g2]
                        if len(x1) > 1 and len(x2) > 1:
                            _, p = ttest_ind(x1, x2, equal_var=False)
                            smd_val = _smd(x1, x2)
                            pairs.append({
                                'Column': col,
                                'Pair': f"{g1} vs {g2}",
                                'p-value': p,
                                'SMD': smd_val,
                                f'Mean {g1}': x1.mean(),
                                f'Mean {g2}': x2.mean()
                            })
                
                if pairs:
                    eval_data.extend(pairs)
            
            if eval_data:
                eval_df = pd.DataFrame(eval_data)
                st.dataframe(eval_df, use_container_width=True, hide_index=True)
            else:
                st.info("No valid numeric comparisons available")
        else:
            st.info("No numeric columns selected")
        
        st.markdown("**Categorical Balance Summary:**")
        if strat_columns:
            for col in strat_columns[:5]:  # Limit for performance
                try:
                    tmp = rebalanced_df[[group_column, col]].copy()
                    tmp[col] = tmp[col].astype(str).fillna("__MISSING__")
                    tmp[group_column] = tmp[group_column].astype(str)
                    ct = pd.crosstab(tmp[group_column], tmp[col], normalize="index").fillna(0)
                    
                    # Calculate pairwise imbalance matrix
                    groups_sorted = sorted(ct.index.tolist())
                    n_groups = len(groups_sorted)
                    
                    if n_groups < 2:
                        st.info(f"Need at least 2 groups for {col}")
                        continue
                    
                    # Create pairwise imbalance matrix
                    imbalance_matrix = pd.DataFrame(0.0, index=groups_sorted, columns=groups_sorted)
                    
                    for i, g1 in enumerate(groups_sorted):
                        for j, g2 in enumerate(groups_sorted):
                            if i != j:
                                # Pairwise difference: |g1_distribution - g2_distribution|.sum() * 100
                                diff = (ct.loc[g1] - ct.loc[g2]).abs().sum() * 100
                                imbalance_matrix.loc[g1, g2] = diff
                    
                    # Set diagonal to NaN for cleaner display
                    for g in groups_sorted:
                        imbalance_matrix.loc[g, g] = np.nan
                    
                    st.write(f"**{col}:**")
                    st.dataframe(imbalance_matrix.round(2), use_container_width=True)
                except Exception:
                    st.warning(f"Could not compute imbalance for {col}")
        else:
            st.info("No categorical columns selected")
    
    else:  # Visual Report
        balance_fig = create_balance_report_plotly(
            rebalanced_df,
            value_columns=value_columns[:5] if len(value_columns) > 5 else value_columns,
            strat_columns=strat_columns[:3] if len(strat_columns) > 3 else strat_columns,
            group_column=group_column,
            title="Rebalanced Group Balance Analysis"
        )
        
        # Add plot to artifact
        artifact = st.session_state.get('rebalancer_artifact')
        if artifact:
            artifact.add_plot('balance_report', balance_fig, 'Rebalanced group balance visualization report')
        
        st.plotly_chart(balance_fig, use_container_width=True)
    
    # Download section
    st.divider()
    st.subheader("💾 Download Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        # Excel summary
        try:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Group sizes
                sizes_df = pd.DataFrame({
                    'Group': groups,
                    'Original Size': [original_sizes.get(g, 0) for g in groups],
                    'Rebalanced Size': [rebalanced_sizes.get(g, 0) for g in groups],
                    'Change': [rebalanced_sizes.get(g, 0) - original_sizes.get(g, 0) for g in groups]
                })
                sizes_df.to_excel(writer, sheet_name='Group Sizes', index=False)
                
                # Loss history
                if loss_history:
                    loss_df = pd.DataFrame({
                        'Iteration': range(len(loss_history)),
                        'Loss': loss_history
                    })
                    loss_df.to_excel(writer, sheet_name='Loss History', index=False)
            
            buffer.seek(0)
            st.download_button(
                label="📊 Download Summary (Excel)",
                data=buffer.getvalue(),
                file_name="rebalancing_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.download_button(
                label="📊 Download Summary (Excel)",
                data=b"",
                file_name="",
                disabled=True
            )
    
    with col_dl2:
        # HTML plots
        try:
            balance_fig = create_balance_report_plotly(
                rebalanced_df,
                value_columns=value_columns[:5] if len(value_columns) > 5 else value_columns,
                strat_columns=strat_columns[:3] if len(strat_columns) > 3 else strat_columns,
                group_column=group_column,
                title="Rebalanced Group Balance Analysis"
            )
            
            # Add plot to artifact (if not already added)
            artifact = st.session_state.get('rebalancer_artifact')
            if artifact and 'balance_report' not in artifact.plots:
                artifact.add_plot('balance_report', balance_fig, 'Rebalanced group balance visualization report')
            
            html_buffer = balance_fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📈 Download Plots (HTML)",
                data=html_buffer,
                file_name="rebalancing_balance_report.html",
                mime="text/html"
            )
        except Exception as e:
            st.download_button(
                label="📈 Download Plots (HTML)",
                data=b"",
                file_name="",
                disabled=True
            )
    
    with col_dl3:
        # CSV data
        csv_buffer = rebalanced_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Data (CSV)",
            data=csv_buffer,
            file_name="rebalanced_data.csv",
            mime="text/csv"
        )
