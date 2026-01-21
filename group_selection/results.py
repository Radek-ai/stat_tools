"""
Results display for group selection page (plots, evaluation, downloads).
"""

import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import ttest_ind

from utils.balance_plots import create_balance_report_plotly
from utils.stats import smd as _smd
from utils.streamlit_errors import handle_error


def render_balancing_results(original_df: pd.DataFrame) -> None:
    """Render the balancing results section (plots, evaluation, downloads)"""
    st.divider()
    results_title = "📊 Group Assignment Results" if st.session_state.balancing_config.get('mode') == "Basic" else "📊 Balancing Results"
    st.subheader(results_title)
    
    balanced_df = st.session_state.balanced_data
    config = st.session_state.balancing_config
    
    # Group sizes
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.write("**Group Sizes:**")
        group_sizes = balanced_df[config['group_column']].value_counts()
        for group_name in config['group_names']:
            size = group_sizes.get(group_name, 0)
            st.metric(group_name, f"{size:,}", f"{size/len(balanced_df)*100:.1f}%")
    
    with col_res2:
        st.write("**Data Summary:**")
        original_size = len(original_df)
        balanced_size = len(balanced_df)
        st.metric("Original Rows", f"{original_size:,}")
        st.metric("Assigned Rows", f"{balanced_size:,}")
        if config.get('loss_history') and config.get('mode') == "Advanced":
            final_loss = config['loss_history'][-1] if config['loss_history'] else None
            if final_loss is not None:
                st.metric("Final Loss", f"{final_loss:.4f}")
    
    # Loss History Plot (only for Advanced mode)
    if config.get('loss_history') and config.get('mode') == "Advanced" and len(config['loss_history']) > 1:
        st.divider()
        st.subheader("📉 Loss History")
        
        try:
            import plotly.graph_objects as go
            
            loss_history = config['loss_history']
            loss_history_runs = config.get('loss_history_runs', [])
            
            # Create loss history plot
            fig_loss = go.Figure()
            
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
                    fig_loss.add_trace(go.Scatter(
                        x=run_iterations,
                        y=run,
                        mode='lines+markers',
                        name=f'Run {run_idx + 1}',
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        legendgroup=f'run_{run_idx}'
                    ))
                    
                    # Add annotation at the end of each run (except the last one)
                    if run_idx < len(loss_history_runs) - 1 and len(run) > 0:
                        end_iter = cumulative_iter + len(run) - 1
                        end_loss = run[-1]
                        fig_loss.add_annotation(
                            x=end_iter,
                            y=end_loss,
                            text=f"Run {run_idx + 1}<br>End: {end_loss:.4f}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=color,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor=color,
                            borderwidth=1,
                            xshift=30
                        )
                    
                    cumulative_iter += len(run)
                
                # Add vertical lines to separate runs
                cumulative_iter = 0
                for run_idx, run in enumerate(loss_history_runs[:-1]):  # All but last
                    if not run or len(run) == 0:
                        continue
                    cumulative_iter += len(run)
                    fig_loss.add_vline(
                        x=cumulative_iter - 0.5,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5,
                        annotation_text=f"Run {run_idx + 1} → {run_idx + 2}",
                        annotation_position="top"
                    )
                
                # Add initial and final annotations
                if len(loss_history_runs) > 0 and len(loss_history_runs[0]) > 0:
                    initial_loss = loss_history_runs[0][0]
                    fig_loss.add_annotation(
                        x=0,
                        y=initial_loss,
                        text=f"Initial: {initial_loss:.4f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='green',
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='green',
                        borderwidth=1
                    )
                    
                    if len(loss_history) > 0:
                        final_loss = loss_history[-1]
                        initial_loss = loss_history[0]
                        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                        fig_loss.add_annotation(
                            x=len(loss_history) - 1,
                            y=final_loss,
                            text=f"Final: {final_loss:.4f}<br>Total Improvement: {improvement:.1f}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor='red',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='red',
                            borderwidth=1,
                            xshift=-30
                        )
                    
                    fig_loss.update_layout(
                        title=f"Loss Convergence Over Iterations ({len(loss_history_runs)} runs)",
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
                iterations = list(range(len(loss_history)))
                
                fig_loss.add_trace(go.Scatter(
                    x=iterations,
                    y=loss_history,
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                
                # Add initial and final loss annotations
                if len(loss_history) > 0:
                    initial_loss = loss_history[0]
                    final_loss = loss_history[-1]
                    improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                    
                    fig_loss.add_annotation(
                        x=0,
                        y=initial_loss,
                        text=f"Initial: {initial_loss:.4f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='green',
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='green'
                    )
                    
                    if len(loss_history) > 1:
                        fig_loss.add_annotation(
                            x=len(loss_history) - 1,
                            y=final_loss,
                            text=f"Final: {final_loss:.4f}<br>Improvement: {improvement:.1f}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor='red',
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='red'
                        )
                    
                    fig_loss.update_layout(
                        title="Loss Convergence Over Iterations",
                        xaxis_title="Iteration",
                        yaxis_title="Total Loss",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400,
                        showlegend=False
                    )
            
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Show summary stats
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
    
    # Evaluation with View Switcher
    st.divider()
    st.subheader("📈 Balance Evaluation")
    
    # Generate plot immediately (for download button)
    try:
        balance_fig = create_balance_report_plotly(
            balanced_df,
            value_columns=config['value_columns'],
            strat_columns=config['strat_columns'],
            group_column=config['group_column'],
            title="Group Balance Analysis Report"
        )
        st.session_state.balance_report_fig = balance_fig
        
        # Add plot to artifact
        artifact = st.session_state.get('group_selection_artifact')
        if artifact:
            artifact.add_plot('balance_report', balance_fig, 'Group balance visualization report')
    except Exception as e:
        st.session_state.balance_report_fig = None
        if st.session_state.get('balance_view_mode', 'Summary') == "Visual Report":
            st.error(f"❌ Error generating balance report: {str(e)}")
    
    # View switcher
    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="balance_view_mode"
    )
    
    if view_mode == "Summary":
        # Simple text-based evaluation
        # Create evaluation summary
        st.markdown("**Numeric Balance Summary:**")
        if config['value_columns']:
            eval_data = []
            for col in config['value_columns']:
                groups_data = {}
                for group_name in config['group_names']:
                    group_df = balanced_df[balanced_df[config['group_column']] == group_name]
                    groups_data[group_name] = group_df[col].dropna()
                
                # Calculate pairwise statistics
                pairs = []
                for i, g1 in enumerate(config['group_names']):
                    for g2 in config['group_names'][i+1:]:
                        x1, x2 = groups_data[g1], groups_data[g2]
                        if len(x1) > 1 and len(x2) > 1:
                            _, p = ttest_ind(x1, x2, equal_var=False)
                            smd_val = _smd(x1, x2)
                            pairs.append({
                                'Column': col,
                                'Pair': f"{g1} vs {g2}",
                                'p-value': p,
                                'SMD': smd_val,
                                'Mean ' + g1: x1.mean(),
                                'Mean ' + g2: x2.mean()
                            })
                
                if pairs:
                    eval_df = pd.DataFrame(pairs)
                    st.dataframe(eval_df, use_container_width=True, hide_index=True)
        
        # Categorical balance
        if config['strat_columns']:
            st.markdown("**Categorical Balance Summary:**")
            for col in config['strat_columns']:
                tmp = balanced_df[[config['group_column'], col]].copy()
                tmp[col] = tmp[col].fillna("__MISSING__")
                ct = pd.crosstab(tmp[config['group_column']], tmp[col], normalize="index").fillna(0)
                
                # Calculate pairwise imbalance matrix
                groups = sorted(ct.index.tolist())
                n_groups = len(groups)
                
                if n_groups < 2:
                    st.info(f"Need at least 2 groups for {col}")
                    continue
                
                # Create pairwise imbalance matrix
                imbalance_matrix = pd.DataFrame(0.0, index=groups, columns=groups)
                
                for i, g1 in enumerate(groups):
                    for j, g2 in enumerate(groups):
                        if i != j:
                            # Pairwise difference: |g1_distribution - g2_distribution|.sum() * 100
                            diff = (ct.loc[g1] - ct.loc[g2]).abs().sum() * 100
                            imbalance_matrix.loc[g1, g2] = diff
                
                # Set diagonal to NaN for cleaner display
                for g in groups:
                    imbalance_matrix.loc[g, g] = np.nan
                
                st.write(f"**{col}:**")
                st.dataframe(imbalance_matrix.round(2), use_container_width=True)
    
    else:  # Visual Report
        st.markdown("### 📊 Interactive Balance Report")
        
        if st.session_state.get('balance_report_fig') is not None:
            # Display plot
            st.plotly_chart(st.session_state.balance_report_fig, use_container_width=True)
        else:
            st.warning("⚠️ Could not generate balance report. Please check your data and column selections.")
    
    # Download section
    st.divider()
    st.subheader("💾 Downloads")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        # Download Summary as Excel
        try:
            from io import BytesIO
            import openpyxl
            
            # Create Excel file with summary data
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Numeric balance summary
                if config['value_columns']:
                    all_pairs = []
                    for col in config['value_columns']:
                        groups_data = {}
                        for group_name in config['group_names']:
                            group_df = balanced_df[balanced_df[config['group_column']] == group_name]
                            groups_data[group_name] = group_df[col].dropna()
                        
                        for i, g1 in enumerate(config['group_names']):
                            for g2 in config['group_names'][i+1:]:
                                x1, x2 = groups_data[g1], groups_data[g2]
                                if len(x1) > 1 and len(x2) > 1:
                                    _, p = ttest_ind(x1, x2, equal_var=False)
                                    smd_val = _smd(x1, x2)
                                    all_pairs.append({
                                        'Column': col,
                                        'Pair': f"{g1} vs {g2}",
                                        'p-value': p,
                                        'SMD': smd_val,
                                        'Mean ' + g1: x1.mean(),
                                        'Mean ' + g2: x2.mean()
                                    })
                    
                    if all_pairs:
                        numeric_df = pd.DataFrame(all_pairs)
                        numeric_df.to_excel(writer, sheet_name='Numeric Balance', index=False)
                
                # Categorical balance summary (pairwise matrix)
                if config['strat_columns']:
                    for col in config['strat_columns']:
                        tmp = balanced_df[[config['group_column'], col]].copy()
                        tmp[col] = tmp[col].fillna("__MISSING__")
                        ct = pd.crosstab(tmp[config['group_column']], tmp[col], normalize="index").fillna(0)
                        
                        # Calculate pairwise imbalance matrix
                        groups = sorted(ct.index.tolist())
                        n_groups = len(groups)
                        
                        if n_groups >= 2:
                            # Create pairwise imbalance matrix
                            imbalance_matrix = pd.DataFrame(0.0, index=groups, columns=groups)
                            
                            for i, g1 in enumerate(groups):
                                for j, g2 in enumerate(groups):
                                    if i != j:
                                        # Pairwise difference: |g1_distribution - g2_distribution|.sum() * 100
                                        diff = (ct.loc[g1] - ct.loc[g2]).abs().sum() * 100
                                        imbalance_matrix.loc[g1, g2] = diff
                            
                            # Set diagonal to NaN for cleaner display
                            for g in groups:
                                imbalance_matrix.loc[g, g] = np.nan
                            
                            # Sanitize sheet name (Excel doesn't allow :, /, \, ?, *, [])
                            sheet_name = f'Categorical {col[:30]}'.replace(':', '_').replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')
                            imbalance_matrix.to_excel(writer, sheet_name=sheet_name)
                
                # Group sizes
                group_sizes = balanced_df[config['group_column']].value_counts()
                sizes_df = pd.DataFrame({
                    'Group': group_sizes.index,
                    'Count': group_sizes.values,
                    'Percentage': (group_sizes.values / len(balanced_df) * 100).round(2)
                })
                sizes_df.to_excel(writer, sheet_name='Group Sizes', index=False)
            
            excel_buffer.seek(0)
            
            st.download_button(
                label="📊 Download Summary (Excel)",
                data=excel_buffer.getvalue(),
                file_name=f"balance_summary_{st.session_state.get('uploaded_filename', 'data').replace('.csv', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating Excel: {str(e)}")
            # Provide empty data for disabled button
            st.download_button(
                label="📊 Download Summary (Excel)",
                data=b"",
                file_name="",
                disabled=True,
                use_container_width=True
            )
    
    with col_dl2:
        # Download Plots as HTML
        if st.session_state.get('balance_report_fig') is not None:
            fig = st.session_state.balance_report_fig
            html_buffer = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📈 Download Plots (HTML)",
                data=html_buffer,
                file_name="balance_report.html",
                mime="text/html",
                use_container_width=True
            )
        else:
            st.download_button(
                label="📈 Download Plots (HTML)",
                data="",
                file_name="",
                disabled=True,
                use_container_width=True,
                help="Generate visual report first"
            )
    
    with col_dl3:
        # Download Data as CSV
        csv = balanced_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name=f"balanced_{st.session_state.get('uploaded_filename', 'data.csv')}",
            mime="text/csv",
            use_container_width=True
        )
