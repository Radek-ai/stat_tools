"""
Configuration UI for rebalancer page (group selection, filtering, balance reports).
"""

import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import ttest_ind

from utils.balance_plots import create_balance_report_plotly
from utils.data_filtering import is_id_column
from utils.stats import smd as _smd
from utils.streamlit_filters import render_filtering_tabs, render_apply_reset_filters


def render_configuration():
    """Render the configuration section"""
    # Initialize session state for filtered data
    if 'rebalancer_filtered_data' not in st.session_state:
        st.session_state.rebalancer_filtered_data = None
    
    if st.session_state.get('rebalancer_uploaded_data') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    df = st.session_state.rebalancer_uploaded_data.copy()
    
    st.header("⚙️ Configuration")
    
    # Group column selection
    st.subheader("📋 Select Group Column")
    group_column = st.selectbox(
        "Group Column",
        options=[""] + df.columns.tolist(),
        index=0,  # Default to empty
        key="rebalancer_upload_group_column",
        help="Select the column containing group assignments"
    )
    
    if not group_column or group_column == "":
        return
    
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    st.info(f"📊 Found {n_groups} groups")
    
    # Show group sizes
    group_sizes = df[group_column].value_counts()
    st.write("**Group Sizes:**")
    col_size1, col_size2 = st.columns(2)
    with col_size1:
        for group_name in groups[:len(groups)//2 + len(groups)%2]:
            size = group_sizes.get(group_name, 0)
            st.metric(str(group_name), f"{size:,}", f"{size/len(df)*100:.1f}%")
    with col_size2:
        for group_name in groups[len(groups)//2 + len(groups)%2:]:
            size = group_sizes.get(group_name, 0)
            st.metric(str(group_name), f"{size:,}", f"{size/len(df)*100:.1f}%")
    
    st.divider()
    
    # Data Filtering Section
    st.subheader("🔍 Data Filtering")
    st.markdown("Apply filters to remove outliers or filter by specific values")

    filtered_df, filter_config = render_filtering_tabs(
        df,
        key_prefix="rebalancer_",
        exclude_columns=[group_column],
        exclude_id_columns=True,
    )

    render_apply_reset_filters(
        original_df=df,
        filtered_df=filtered_df,
        filtered_state_key="rebalancer_filtered_data",
        key_prefix="rebalancer_",
        artifact=st.session_state.get("rebalancer_artifact"),
        artifact_filtered_df_name="filtered_data",
        artifact_filtered_df_description="Data after applying filters",
        artifact_log_category="filtering",
        artifact_log_id="current_filters",
        filter_config=filter_config,
    )
    
    # Initial balance report (in expander)
    with st.expander("📈 Initial Group Balance Report", expanded=False):
        
        # Get all numeric and categorical columns for initial report (exclude ID columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols or categorical_cols:
            # Filter out group column and ID columns
            numeric_cols_filtered = [c for c in numeric_cols if c != group_column and not is_id_column(df, c)]
            categorical_cols_filtered = [c for c in categorical_cols if c != group_column and not is_id_column(df, c)]
            
            if not numeric_cols_filtered and not categorical_cols_filtered:
                st.info("ℹ️ No valid columns available for balance analysis (excluding group column)")
            else:
                # View switcher
                initial_view_mode = st.radio(
                    "View Mode",
                    options=["Summary", "Visual Report"],
                    index=0,
                    horizontal=True,
                    key="initial_balance_view_mode"
                )
                
                if initial_view_mode == "Summary":
                    # Summary view
                    st.markdown("**Numeric Balance Summary:**")
                    if numeric_cols_filtered:
                        eval_data = []
                        groups = sorted(df[group_column].unique())
                        
                        for col in numeric_cols_filtered[:10]:  # Limit for performance
                            groups_data = {}
                            for group_name in groups:
                                group_df = df[df[group_column] == group_name]
                                groups_data[group_name] = group_df[col].dropna()
                            
                            # Calculate pairwise statistics
                            pairs = []
                            for i, g1 in enumerate(groups):
                                for g2 in groups[i+1:]:
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
                        st.info("No numeric columns available")
                    
                    st.markdown("**Categorical Balance Summary:**")
                    if categorical_cols_filtered:
                        for col in categorical_cols_filtered[:5]:  # Limit for performance
                            try:
                                tmp = df[[group_column, col]].copy()
                                tmp[col] = tmp[col].astype(str).fillna("__MISSING__")
                                tmp[group_column] = tmp[group_column].astype(str)
                                ct = pd.crosstab(tmp[group_column], tmp[col], normalize="index").fillna(0)
                                
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
                            except Exception:
                                st.warning(f"Could not compute imbalance for {col}")
                    else:
                        st.info("No categorical columns available")
                else:
                    # Visual Report
                    balance_fig = create_balance_report_plotly(
                        df,
                        value_columns=numeric_cols_filtered[:5] if len(numeric_cols_filtered) > 5 else numeric_cols_filtered,
                        strat_columns=categorical_cols_filtered[:3] if len(categorical_cols_filtered) > 3 else categorical_cols_filtered,
                        group_column=group_column,
                        title="Initial Group Balance Analysis"
                    )
                    st.plotly_chart(balance_fig, use_container_width=True)
        else:
            st.info("ℹ️ No numeric or categorical columns found for balance analysis")
    
    # Filtered data balance report (in expander)
    if st.session_state.get('rebalancer_filtered_data') is not None:
        filtered_df_for_balance = st.session_state.rebalancer_filtered_data.copy()
        with st.expander("📊 Filtered Data Balance Report", expanded=False):
            # Get all numeric and categorical columns for filtered report (exclude ID columns)
            numeric_cols = filtered_df_for_balance.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = filtered_df_for_balance.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols or categorical_cols:
                # Filter out group column and ID columns
                numeric_cols_filtered = [c for c in numeric_cols if c != group_column and not is_id_column(filtered_df_for_balance, c)]
                categorical_cols_filtered = [c for c in categorical_cols if c != group_column and not is_id_column(filtered_df_for_balance, c)]
                
                if not numeric_cols_filtered and not categorical_cols_filtered:
                    st.info("ℹ️ No valid columns available for balance analysis (excluding group column)")
                else:
                    # View switcher
                    filtered_view_mode = st.radio(
                        "View Mode",
                        options=["Summary", "Visual Report"],
                        index=0,
                        horizontal=True,
                        key="filtered_balance_view_mode"
                    )
                    
                    if filtered_view_mode == "Summary":
                        # Summary view
                        st.markdown("**Numeric Balance Summary:**")
                        if numeric_cols_filtered:
                            eval_data = []
                            groups_list = sorted(filtered_df_for_balance[group_column].unique())
                            
                            for col in numeric_cols_filtered[:10]:  # Limit for performance
                                groups_data = {}
                                for group_name in groups_list:
                                    group_df = filtered_df_for_balance[filtered_df_for_balance[group_column] == group_name]
                                    groups_data[group_name] = group_df[col].dropna()
                                
                                # Calculate pairwise statistics
                                pairs = []
                                for i, g1 in enumerate(groups_list):
                                    for g2 in groups_list[i+1:]:
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
                            st.info("No numeric columns available")
                        
                        st.markdown("**Categorical Balance Summary:**")
                        if categorical_cols_filtered:
                            for col in categorical_cols_filtered[:5]:  # Limit for performance
                                try:
                                    tmp = filtered_df_for_balance[[group_column, col]].copy()
                                    tmp[col] = tmp[col].astype(str).fillna("__MISSING__")
                                    tmp[group_column] = tmp[group_column].astype(str)
                                    ct = pd.crosstab(tmp[group_column], tmp[col], normalize="index").fillna(0)
                                    
                                    # Calculate pairwise imbalance matrix
                                    groups_list = sorted(ct.index.tolist())
                                    n_groups_list = len(groups_list)
                                    
                                    if n_groups_list < 2:
                                        st.info(f"Need at least 2 groups for {col}")
                                        continue
                                    
                                    # Create pairwise imbalance matrix
                                    imbalance_matrix = pd.DataFrame(0.0, index=groups_list, columns=groups_list)
                                    
                                    for i, g1 in enumerate(groups_list):
                                        for j, g2 in enumerate(groups_list):
                                            if i != j:
                                                # Pairwise difference: |g1_distribution - g2_distribution|.sum() * 100
                                                diff = (ct.loc[g1] - ct.loc[g2]).abs().sum() * 100
                                                imbalance_matrix.loc[g1, g2] = diff
                                    
                                    # Set diagonal to NaN for cleaner display
                                    for g in groups_list:
                                        imbalance_matrix.loc[g, g] = np.nan
                                    
                                    st.write(f"**{col}:**")
                                    st.dataframe(imbalance_matrix.round(2), use_container_width=True)
                                except Exception:
                                    st.warning(f"Could not compute imbalance for {col}")
                        else:
                            st.info("No categorical columns available")
                    else:
                        # Visual Report
                        balance_fig = create_balance_report_plotly(
                            filtered_df_for_balance,
                            value_columns=numeric_cols_filtered[:5] if len(numeric_cols_filtered) > 5 else numeric_cols_filtered,
                            strat_columns=categorical_cols_filtered[:3] if len(categorical_cols_filtered) > 3 else categorical_cols_filtered,
                            group_column=group_column,
                            title="Filtered Data Group Balance Analysis"
                        )
                        st.plotly_chart(balance_fig, use_container_width=True)
            else:
                st.info("ℹ️ No numeric or categorical columns found for balance analysis")
