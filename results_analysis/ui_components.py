"""
UI components for results analysis page
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results_analysis.plots import (
    create_basic_analysis_plotly,
    create_cuped_analysis_plotly,
    create_did_analysis_plotly
)
from power_analysis.components import render_download_button, render_plot_with_download
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


def render_data_upload():
    """Render the data upload section"""
    # Initialize session state
    if 'results_uploaded_data' not in st.session_state:
        st.session_state.results_uploaded_data = None
    
    st.header("📤 Upload Experiment Results Data")
    st.markdown("Upload a CSV file containing your experiment results with group assignments and metric values")
    
    # Dummy data loader expander
    with st.expander("🎲 Load Dummy Data", expanded=False):
        st.markdown("Load pre-generated sample experiment results data for testing")
        
        if st.button("🎲 Load Dummy Data", key="results_load_dummy", type="primary"):
            import os
            dummy_file = os.path.join("dummy_data", "results_analysis_dummy.csv")
            if os.path.exists(dummy_file):
                df = pd.read_csv(dummy_file)
                st.session_state.results_uploaded_data = df
                st.session_state.results_filename = "dummy_results_data.csv"
                st.success(f"✅ Dummy data loaded! ({len(df)} rows, {len(df.columns)} columns)")
                st.rerun()
            else:
                st.error(f"❌ Dummy data file not found: {dummy_file}")
                st.info("💡 Run 'python dummy_data_builders/generate_all_dummy_data.py' to generate the files")
    
    uploaded_file = st.file_uploader(
        "Or choose a CSV file",
        type=['csv'],
        key="results_file_upload",
        help="Upload a CSV file with experiment results"
    )
    
    # Check if we have data (either from upload or dummy)
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.results_uploaded_data = df
            st.session_state.results_filename = uploaded_file.name
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            df = None
    
    # Also check if data was loaded from dummy
    if df is None and st.session_state.results_uploaded_data is not None:
        df = st.session_state.results_uploaded_data
        if 'results_filename' not in st.session_state:
            st.session_state.results_filename = "dummy_results_data.csv"
    
    if df is not None:
        # Basic statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Rows", f"{len(df):,}")
        with col_stat2:
            st.metric("Total Columns", len(df.columns))
        with col_stat3:
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_count)
        
        # Display preview
        with st.expander("📋 Data Preview", expanded=True):
            preview_df = df.head(20).copy()
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"Showing first 20 rows of {len(df):,} total rows")
        
        # Display column info
        with st.expander("📊 Column Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Numeric Columns:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write(", ".join(numeric_cols))
                else:
                    st.write("None found")
            
            with col2:
                st.write("**Categorical Columns:**")
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    st.write(", ".join(categorical_cols))
                else:
                    st.write("None found")
            
            with col3:
                st.write("**All Columns:**")
                st.write(", ".join(df.columns.tolist()))
    elif st.session_state.results_uploaded_data is None:
        st.info("ℹ️ Please upload a CSV file or load dummy data to begin analysis")


def render_configuration():
    """Render the configuration section"""
    if st.session_state.get('results_uploaded_data') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    df = st.session_state.results_uploaded_data.copy()
    
    st.header("⚙️ Configuration")
    
    # Group column selection
    st.subheader("📋 Select Group Column")
    group_column = st.selectbox(
        "Group Column",
        options=[""] + df.columns.tolist(),
        index=0,  # Default to empty
        key="results_group_column",
        help="Select the column containing group assignments"
    )
    
    if group_column and group_column != "":
        groups = sorted(df[group_column].unique())
        n_groups = len(groups)
        st.info(f"📊 Found {n_groups} groups: {', '.join(map(str, groups))}")
        
        # Show group sizes
        group_sizes = df[group_column].value_counts()
        st.write("**Group Sizes:**")
        cols = st.columns(min(n_groups, 4))
        for idx, group_name in enumerate(groups):
            with cols[idx % len(cols)]:
                size = group_sizes.get(group_name, 0)
                st.metric(str(group_name), f"{size:,}", f"{size/len(df)*100:.1f}%")
        
        # Store groups list (group_column is already stored by the widget with key="results_group_column")
        st.session_state.results_groups = groups


def render_basic_analysis():
    """Render basic treatment effect analysis"""
    st.header("📊 Basic Treatment Effect Analysis")
    st.markdown("Analyze treatment effects, uplifts, and statistical significance")
    
    # Check if data is available
    if 'results_uploaded_data' not in st.session_state or st.session_state.results_uploaded_data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    df = st.session_state.results_uploaded_data.copy()
    
    if 'results_group_column' not in st.session_state or not st.session_state.results_group_column:
        st.warning("⚠️ Please select a group column in the 'Configuration' tab")
        return
    
    group_column = st.session_state.results_group_column
    groups = st.session_state.results_groups
    
    st.divider()
    
    # Column selection
    st.subheader("📊 Select Metrics to Analyze")
    
    from utils.data_filtering import is_id_column
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out group column and ID columns
    numeric_cols = [col for col in numeric_cols if col != group_column and not is_id_column(df, col)]
    
    # Also filter out pre/post columns (they should be analyzed via CUPED/DiD)
    # Keep only base columns (not ending with _pre, _post, _aa, _ab)
    base_numeric_cols = [col for col in numeric_cols if not col.endswith(('_pre', '_post', '_aa', '_ab'))]
    
    if not base_numeric_cols:
        st.error("❌ No numeric columns found for analysis")
        return
    
    # Prioritize customer_churned and new_customer in default selection
    priority_cols = ['customer_churned', 'new_customer']
    other_cols = [col for col in base_numeric_cols if col not in priority_cols]
    
    # Build default selection: priority cols first, then others
    default_cols = [col for col in priority_cols if col in base_numeric_cols]
    default_cols.extend([col for col in other_cols if col not in default_cols][:max(0, 5 - len(default_cols))])
    
    value_columns = st.multiselect(
        "Metric Columns",
        options=base_numeric_cols,
        default=default_cols if default_cols else base_numeric_cols[:min(5, len(base_numeric_cols))],
        key="results_basic_metrics",
        help="Select numeric metrics to analyze for treatment effects"
    )
    
    if not value_columns:
        st.info("ℹ️ Please select at least one metric column")
        return
    
    st.divider()
    
    # View switcher
    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="basic_analysis_view_mode"
    )
    
    if view_mode == "Summary":
        # Summary tables
        st.subheader("📋 Treatment Effect Summary")
        
        summary_data = []
        for col in value_columns:
            group_data = {}
            group_means = {}
            
            for g in groups:
                data = df[df[group_column] == g][col].dropna()
                if len(data) >= 2:
                    group_data[g] = data
                    group_means[g] = data.mean()
            
            if len(group_means) < 2:
                continue
            
            # Calculate pairwise statistics
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    if g1 in group_data and g2 in group_data:
                        x1, x2 = group_data[g1], group_data[g2]
                        m1, m2 = group_means[g1], group_means[g2]
                        
                        # Uplift
                        uplift_pct = ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan
                        
                        # SMD
                        smd_val = _smd(x1, x2)
                        
                        # p-value
                        try:
                            _, p = ttest_ind(x1, x2, equal_var=False)
                        except:
                            p = np.nan
                        
                        summary_data.append({
                            'Metric': col,
                            'Group 1': str(g1),
                            'Group 2': str(g2),
                            'Mean G1': f"{m1:.3f}",
                            'Mean G2': f"{m2:.3f}",
                            'Uplift (%)': f"{uplift_pct:+.2f}" if np.isfinite(uplift_pct) else "N/A",
                            'SMD': f"{smd_val:.3f}" if np.isfinite(smd_val) else "N/A",
                            'p-value': f"{p:.4f}" if np.isfinite(p) else "N/A",
                            'Significant': "Yes" if np.isfinite(p) and p < 0.05 else "No"
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Summary as CSV",
                data=csv,
                file_name="basic_analysis_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No valid data for summary")
    
    else:
        # Visual report
        st.subheader("📈 Treatment Effect Visualizations")
        
        try:
            fig = create_basic_analysis_plotly(
                df,
                value_columns,
                group_column,
                title="Basic Treatment Effect Analysis"
            )
            
            render_plot_with_download(
                fig,
                "basic_analysis_report.html",
                "Download interactive treatment effect analysis report"
            )
        except Exception as e:
            st.error(f"❌ Error generating visualizations: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def render_cuped_analysis():
    """Render CUPED-adjusted analysis"""
    st.header("🔬 CUPED-Adjusted Analysis")
    st.markdown("Analyze treatment effects using CUPED (Controlled-experiment Using Pre-Experiment Data) adjustment")
    
    # Check if data is available
    if 'results_uploaded_data' not in st.session_state or st.session_state.results_uploaded_data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    df = st.session_state.results_uploaded_data.copy()
    
    if 'results_group_column' not in st.session_state or not st.session_state.results_group_column:
        st.warning("⚠️ Please select a group column in the 'Configuration' tab")
        return
    
    group_column = st.session_state.results_group_column
    groups = st.session_state.results_groups
    
    st.divider()
    
    # Suffix configuration
    st.subheader("⚙️ CUPED Configuration")
    col_suffix1, col_suffix2 = st.columns(2)
    
    with col_suffix1:
        suffix_pre = st.text_input(
            "Pre-experiment suffix",
            value="_pre",
            key="cuped_suffix_pre",
            help="Suffix for pre-experiment metric columns (e.g., 'revenue_pre')"
        )
    
    with col_suffix2:
        suffix_post = st.text_input(
            "Post-experiment suffix",
            value="_post",
            key="cuped_suffix_post",
            help="Suffix for post-experiment metric columns (e.g., 'revenue_post')"
        )
    
    # Find base metrics (columns that have both pre and post versions)
    all_cols = df.columns.tolist()
    from utils.data_filtering import is_id_column
    
    base_metrics = []
    for col in all_cols:
        if col.endswith(suffix_post):
            base = col[:-len(suffix_post)]
            pre_col = base + suffix_pre
            if pre_col in all_cols:
                # Skip if either pre or post column is an ID column
                if not is_id_column(df, pre_col) and not is_id_column(df, col):
                    base_metrics.append(base)
    
    if not base_metrics:
        st.error(f"❌ No metrics found with both '{suffix_pre}' and '{suffix_post}' suffixes")
        st.info(f"ℹ️ Looking for columns like: 'metric{suffix_pre}' and 'metric{suffix_post}'")
        return
    
    st.info(f"✅ Found {len(base_metrics)} metrics with CUPED data: {', '.join(base_metrics[:5])}{'...' if len(base_metrics) > 5 else ''}")
    
    # Allow selection
    selected_metrics = st.multiselect(
        "Select Metrics for CUPED Analysis",
        options=base_metrics,
        default=base_metrics[:min(5, len(base_metrics))] if base_metrics else [],
        key="cuped_metrics"
    )
    
    if not selected_metrics:
        st.info("ℹ️ Please select at least one metric")
        return
    
    st.divider()
    
    # View switcher
    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="cuped_analysis_view_mode"
    )
    
    if view_mode == "Summary":
        # Summary tables
        st.subheader("📋 CUPED-Adjusted Treatment Effect Summary")
        
        summary_data = []
        for metric in selected_metrics:
            pre_col = metric + suffix_pre
            post_col = metric + suffix_post
            
            # CUPED adjustment
            adj = _cuped_adjust(df[pre_col], df[post_col])
            if adj is None:
                continue
            
            # Calculate statistics by group
            cuped_data = {}
            cuped_means = {}
            
            for g in groups:
                data = adj[df[group_column] == g].dropna()
                if len(data) >= 2:
                    cuped_data[g] = data
                    cuped_means[g] = data.mean()
            
            if len(cuped_means) < 2:
                continue
            
            # Calculate pairwise statistics
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    if g1 in cuped_data and g2 in cuped_data:
                        y1, y2 = cuped_data[g1], cuped_data[g2]
                        m1, m2 = cuped_means[g1], cuped_means[g2]
                        
                        # Uplift
                        uplift_pct = ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan
                        
                        # SMD
                        smd_val = _smd(y1, y2)
                        
                        # p-value
                        try:
                            _, p = ttest_ind(y1, y2, equal_var=False)
                        except:
                            p = np.nan
                        
                        summary_data.append({
                            'Metric': metric,
                            'Group 1': str(g1),
                            'Group 2': str(g2),
                            'CUPED Mean G1': f"{m1:.3f}",
                            'CUPED Mean G2': f"{m2:.3f}",
                            'Uplift (%)': f"{uplift_pct:+.2f}" if np.isfinite(uplift_pct) else "N/A",
                            'CUPED SMD': f"{smd_val:.3f}" if np.isfinite(smd_val) else "N/A",
                            'CUPED p-value': f"{p:.4f}" if np.isfinite(p) else "N/A",
                            'Significant': "Yes" if np.isfinite(p) and p < 0.05 else "No"
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CUPED Summary as CSV",
                data=csv,
                file_name="cuped_analysis_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No valid data for CUPED summary")
    
    else:
        # Visual report
        st.subheader("📈 CUPED-Adjusted Visualizations")
        
        try:
            fig = create_cuped_analysis_plotly(
                df,
                selected_metrics,
                group_column,
                suffix_pre=suffix_pre,
                suffix_post=suffix_post,
                title="CUPED-Adjusted Treatment Effect Analysis"
            )
            
            render_plot_with_download(
                fig,
                "cuped_analysis_report.html",
                "Download interactive CUPED analysis report"
            )
        except Exception as e:
            st.error(f"❌ Error generating CUPED visualizations: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def render_did_analysis():
    """Render Difference-in-Differences analysis"""
    st.header("📉 Difference-in-Differences Analysis")
    st.markdown("Analyze treatment effects using Difference-in-Differences (DiD) methodology")
    
    # Check if data is available
    if 'results_uploaded_data' not in st.session_state or st.session_state.results_uploaded_data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    df = st.session_state.results_uploaded_data.copy()
    
    if 'results_group_column' not in st.session_state or not st.session_state.results_group_column:
        st.warning("⚠️ Please select a group column in the 'Configuration' tab")
        return
    
    group_column = st.session_state.results_group_column
    groups = st.session_state.results_groups
    
    st.divider()
    
    # Suffix configuration
    st.subheader("⚙️ DiD Configuration")
    col_suffix1, col_suffix2 = st.columns(2)
    
    with col_suffix1:
        suffix_pre = st.text_input(
            "Pre-period suffix",
            value="_pre",
            key="did_suffix_pre",
            help="Suffix for pre-period metric columns (e.g., 'revenue_pre')"
        )
    
    with col_suffix2:
        suffix_post = st.text_input(
            "Post-period suffix",
            value="_post",
            key="did_suffix_post",
            help="Suffix for post-period metric columns (e.g., 'revenue_post')"
        )
    
    # Find base metrics
    all_cols = df.columns.tolist()
    from utils.data_filtering import is_id_column
    
    base_metrics = []
    for col in all_cols:
        if col.endswith(suffix_post):
            base = col[:-len(suffix_post)]
            pre_col = base + suffix_pre
            if pre_col in all_cols:
                # Skip if either pre or post column is an ID column
                if not is_id_column(df, pre_col) and not is_id_column(df, col):
                    base_metrics.append(base)
    
    if not base_metrics:
        st.error(f"❌ No metrics found with both '{suffix_pre}' and '{suffix_post}' suffixes")
        st.info(f"ℹ️ Looking for columns like: 'metric{suffix_pre}' and 'metric{suffix_post}'")
        return
    
    st.info(f"✅ Found {len(base_metrics)} metrics with DiD data: {', '.join(base_metrics[:5])}{'...' if len(base_metrics) > 5 else ''}")
    
    # Allow selection
    selected_metrics = st.multiselect(
        "Select Metrics for DiD Analysis",
        options=base_metrics,
        default=base_metrics[:min(5, len(base_metrics))] if base_metrics else [],
        key="did_metrics"
    )
    
    if not selected_metrics:
        st.info("ℹ️ Please select at least one metric")
        return
    
    st.divider()
    
    # View switcher
    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="did_analysis_view_mode"
    )
    
    if view_mode == "Summary":
        # Summary tables
        st.subheader("📋 DiD Gap Summary")
        
        summary_data = []
        for metric in selected_metrics:
            pre_col = metric + suffix_pre
            post_col = metric + suffix_post
            
            # Calculate DiD gaps
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    g1df = df[df[group_column] == g1]
                    g2df = df[df[group_column] == g2]
                    
                    g1_pre = g1df[pre_col].mean()
                    g1_post = g1df[post_col].mean()
                    g2_pre = g2df[pre_col].mean()
                    g2_post = g2df[post_col].mean()
                    
                    diff_pre = g2_pre - g1_pre
                    diff_post = g2_post - g1_post
                    did_gap = diff_post - diff_pre
                    
                    # Calculate p-value on change scores
                    g1_change = (g1df[post_col] - g1df[pre_col]).dropna()
                    g2_change = (g2df[post_col] - g2df[pre_col]).dropna()
                    
                    try:
                        if len(g1_change) > 1 and len(g2_change) > 1:
                            _, p = ttest_ind(g2_change, g1_change, equal_var=False)
                        else:
                            p = np.nan
                    except:
                        p = np.nan
                    
                    summary_data.append({
                        'Metric': metric,
                        'Group 1': str(g1),
                        'Group 2': str(g2),
                        'G1 Pre': f"{g1_pre:.3f}",
                        'G1 Post': f"{g1_post:.3f}",
                        'G2 Pre': f"{g2_pre:.3f}",
                        'G2 Post': f"{g2_post:.3f}",
                        'Gap Pre (G2-G1)': f"{diff_pre:.3f}",
                        'Gap Post (G2-G1)': f"{diff_post:.3f}",
                        'Δ DiD Gap': f"{did_gap:.3f}",
                        'p-value': f"{p:.4f}" if np.isfinite(p) else "N/A",
                        'Significant': "Yes" if np.isfinite(p) and p < 0.05 else "No"
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download DiD Summary as CSV",
                data=csv,
                file_name="did_analysis_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No valid data for DiD summary")
    
    else:
        # Visual report
        st.subheader("📈 DiD Visualizations")
        
        try:
            fig = create_did_analysis_plotly(
                df,
                selected_metrics,
                group_column,
                suffix_pre=suffix_pre,
                suffix_post=suffix_post,
                title="Difference-in-Differences Analysis"
            )
            
            render_plot_with_download(
                fig,
                "did_analysis_report.html",
                "Download interactive DiD analysis report"
            )
        except Exception as e:
            st.error(f"❌ Error generating DiD visualizations: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
