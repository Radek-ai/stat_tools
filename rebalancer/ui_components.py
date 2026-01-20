"""
UI components for rebalancer page
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.balance_plots import create_balance_report_plotly
from utils.data_filtering import is_id_column
from others.multi_rebalancer import MultiGroupRebalancer


def create_streamlit_progress_callback(progress_placeholder, status_placeholder):
    """
    Create a progress callback function for Streamlit UI.
    
    Parameters:
    -----------
    progress_placeholder : streamlit.delta_generator.DeltaGenerator
        Placeholder for progress bar
    status_placeholder : streamlit.delta_generator.DeltaGenerator
        Placeholder for status text
        
    Returns:
    --------
    Callable: Progress callback function
    """
    def callback(stage: str, info: dict):
        if stage == "start":
            progress_placeholder.progress(0.0)
            desc = info.get("description", "Rebalancing")
            initial_loss = info.get("initial_loss", 0.0)
            step_info = info.get("step_info", "")
            step_prefix = f"{step_info} - " if step_info else ""
            status_placeholder.info(f"🔄 {step_prefix}{desc} - Initial loss: {initial_loss:.4f}")
        
        elif stage == "update":
            iteration = info.get("iteration", 0)
            total = info.get("total", 1)
            initial_loss = info.get("initial_loss", 0.0)
            current_loss = info.get("current_loss", 0.0)
            gain = info.get("gain", 0.0)
            progress = info.get("progress", 0.0)
            step_info = info.get("step_info", "")
            step_prefix = f"{step_info} - " if step_info else ""
            
            progress_placeholder.progress(progress)
            status_placeholder.info(
                f"🔄 {step_prefix}Iteration {iteration} / {total} | "
                f"Initial: {initial_loss:.4f} | "
                f"Current: {current_loss:.4f} | "
                f"Gain: {gain:.4f}"
            )
        
        elif stage == "complete":
            progress_placeholder.progress(1.0)
            final_loss = info.get("final_loss", 0.0)
            initial_loss = info.get("initial_loss", 0.0)
            total_gain = info.get("total_gain", 0.0)
            step_info = info.get("step_info", "")
            step_prefix = f"{step_info} - " if step_info else ""
            status_placeholder.success(
                f"✅ {step_prefix}Complete! Final loss: {final_loss:.4f} | "
                f"Total gain: {total_gain:.4f}"
            )
    
    return callback


def render_data_upload():
    """Render the data upload section"""
    # Initialize session state
    if 'rebalancer_uploaded_data' not in st.session_state:
        st.session_state.rebalancer_uploaded_data = None
    
    st.header("📤 Upload Data with Existing Groups")
    st.markdown("Upload a CSV file that already contains group assignments. The rebalancer will trim rows to improve balance.")
    
    # Dummy data loader expander
    with st.expander("🎲 Load Dummy Data", expanded=False):
        st.markdown("Load pre-generated sample data with imbalanced groups for testing")
        
        if st.button("🎲 Load Dummy Data", key="rebalancer_load_dummy", type="primary"):
            import os
            dummy_file = os.path.join("dummy_data", "rebalancer_dummy.csv")
            if os.path.exists(dummy_file):
                df = pd.read_csv(dummy_file)
                st.session_state.rebalancer_uploaded_data = df
                st.session_state.rebalancer_filename = "dummy_rebalancer_data.csv"
                st.success(f"✅ Dummy data loaded! ({len(df)} rows, {len(df.columns)} columns)")
                st.rerun()
            else:
                st.error(f"❌ Dummy data file not found: {dummy_file}")
                st.info("💡 Run 'python dummy_data_builders/generate_all_dummy_data.py' to generate the files")
    
    uploaded_file = st.file_uploader(
        "Or choose a CSV file",
        type=['csv'],
        key="rebalancer_file_upload",
        help="Upload a CSV file with existing group assignments"
    )
    
    # Check if we have data (either from upload or dummy)
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.rebalancer_uploaded_data = df
            st.session_state.rebalancer_filename = uploaded_file.name
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            df = None
    
    # Also check if data was loaded from dummy
    if df is None and st.session_state.rebalancer_uploaded_data is not None:
        df = st.session_state.rebalancer_uploaded_data
        if 'rebalancer_filename' not in st.session_state:
            st.session_state.rebalancer_filename = "dummy_rebalancer_data.csv"
    
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
    
    filtered_df = df.copy()
    
    # Filtering tabs
    tab_outliers, tab_values = st.tabs(["Outlier Filtering", "Value-Based Filtering"])
    
    with tab_outliers:
        st.subheader("📉 Outlier Filtering")
        st.markdown("Remove or clip outliers from numeric columns")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        from utils.data_filtering import is_id_column
        numeric_cols = [c for c in numeric_cols if c != group_column and not is_id_column(df, c)]
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for outlier filtering")
        else:
            # Select column for outlier filtering
            outlier_column = st.selectbox(
                "Select Column for Outlier Filtering",
                options=numeric_cols,
                key="rebalancer_outlier_column"
            )
            
            # Outlier method
            outlier_method = st.selectbox(
                "Outlier Method",
                options=["None", "Percentile", "IQR"],
                index=0,
                key="rebalancer_outlier_method",
                help="None: No filtering | Percentile: Remove rows outside percentile range | IQR: Remove rows outside IQR range"
            )
            
            if outlier_method == "Percentile":
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    p_low = st.number_input(
                        "Lower Percentile",
                        min_value=0.0,
                        max_value=100.0,
                        value=1.0,
                        step=0.1,
                        key="rebalancer_p_low",
                        help="Lower percentile (0-100)"
                    )
                with col_p2:
                    p_high = st.number_input(
                        "Upper Percentile",
                        min_value=0.0,
                        max_value=100.0,
                        value=99.0,
                        step=0.1,
                        key="rebalancer_p_high",
                        help="Upper percentile (0-100)"
                    )
                
                if p_low >= p_high:
                    st.error("❌ Lower percentile must be less than upper percentile")
                else:
                    # Show what will be removed
                    from utils.data_filtering import filter_outliers_percentile
                    lower_bound = df[outlier_column].quantile(p_low / 100.0)
                    upper_bound = df[outlier_column].quantile(p_high / 100.0)
                    rows_before = len(filtered_df)
                    metric_total_before = filtered_df[outlier_column].sum()
                    filtered_df = filter_outliers_percentile(filtered_df, outlier_column, p_low, p_high)
                    rows_after = len(filtered_df)
                    removed = rows_before - rows_after
                    metric_total_after = filtered_df[outlier_column].sum()
                    metric_removed_pct = ((metric_total_before - metric_total_after) / metric_total_before * 100) if metric_total_before > 0 else 0
                    
                    st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%) | {metric_removed_pct:.1f}% of total {outlier_column} outside range [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            elif outlier_method == "IQR":
                from utils.data_filtering import filter_outliers_iqr
                iqr_multiplier = st.number_input(
                    "IQR Multiplier",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    key="rebalancer_iqr_multiplier",
                    help="Multiplier for IQR method (default 1.5)"
                )
                
                # Show what will be removed
                Q1 = df[outlier_column].quantile(0.25)
                Q3 = df[outlier_column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                rows_before = len(filtered_df)
                metric_total_before = filtered_df[outlier_column].sum()
                filtered_df = filter_outliers_iqr(filtered_df, outlier_column, iqr_multiplier)
                rows_after = len(filtered_df)
                removed = rows_before - rows_after
                metric_total_after = filtered_df[outlier_column].sum()
                metric_removed_pct = ((metric_total_before - metric_total_after) / metric_total_before * 100) if metric_total_before > 0 else 0
                
                st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%) | {metric_removed_pct:.1f}% of total {outlier_column} outside range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    with tab_values:
        st.subheader("🔢 Value-Based Filtering")
        st.markdown("Filter rows based on specific column values")
        
        # Numeric value filtering
        st.markdown("**Numeric Column Filtering:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        from utils.data_filtering import is_id_column
        numeric_cols = [c for c in numeric_cols if c != group_column and not is_id_column(df, c)]
        
        if numeric_cols:
            from utils.data_filtering import filter_by_value_range
            filter_numeric_col = st.selectbox(
                "Select Numeric Column",
                options=["None"] + numeric_cols,
                key="rebalancer_filter_numeric_col"
            )
            
            if filter_numeric_col != "None":
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    min_val = st.number_input(
                        "Minimum Value",
                        value=None,
                        key="rebalancer_min_val",
                        help="Minimum value (inclusive), leave empty for no limit"
                    )
                with col_v2:
                    max_val = st.number_input(
                        "Maximum Value",
                        value=None,
                        key="rebalancer_max_val",
                        help="Maximum value (inclusive), leave empty for no limit"
                    )
                
                if min_val is not None or max_val is not None:
                    rows_before = len(filtered_df)
                    filtered_df = filter_by_value_range(filtered_df, filter_numeric_col, min_val, max_val)
                    rows_after = len(filtered_df)
                    removed = rows_before - rows_after
                    
                    if removed > 0:
                        st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%)")
        
        # Categorical value filtering
        st.markdown("**Categorical Column Filtering:**")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c != group_column and not is_id_column(df, c)]
        
        if categorical_cols:
            from utils.data_filtering import filter_by_categorical_values
            filter_cat_col = st.selectbox(
                "Select Categorical Column",
                options=["None"] + categorical_cols,
                key="rebalancer_filter_cat_col"
            )
            
            if filter_cat_col != "None":
                unique_vals = df[filter_cat_col].unique().tolist()
                
                filter_mode = st.radio(
                    "Filter Mode",
                    options=["Keep selected values", "Exclude selected values"],
                    key="rebalancer_filter_mode"
                )
                
                if filter_mode == "Keep selected values":
                    keep_vals = st.multiselect(
                        "Values to Keep",
                        options=unique_vals,
                        key="rebalancer_keep_vals"
                    )
                    if keep_vals:
                        rows_before = len(filtered_df)
                        filtered_df = filter_by_categorical_values(filtered_df, filter_cat_col, keep_values=keep_vals)
                        rows_after = len(filtered_df)
                        removed = rows_before - rows_after
                        if removed > 0:
                            st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%)")
                else:
                    exclude_vals = st.multiselect(
                        "Values to Exclude",
                        options=unique_vals,
                        key="rebalancer_exclude_vals"
                    )
                    if exclude_vals:
                        rows_before = len(filtered_df)
                        filtered_df = filter_by_categorical_values(filtered_df, filter_cat_col, exclude_values=exclude_vals)
                        rows_after = len(filtered_df)
                        removed = rows_before - rows_after
                        if removed > 0:
                            st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%)")
    
    # Apply filters button
    st.divider()
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("✅ Apply Filters", type="primary", use_container_width=True, key="rebalancer_apply_filters"):
            st.session_state.rebalancer_filtered_data = filtered_df
            st.success(f"✅ Filters applied! {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}% retained)")
            st.rerun()
    
    with col_btn2:
        if st.button("🔄 Reset Filters", use_container_width=True, key="rebalancer_reset_filters"):
            st.session_state.rebalancer_filtered_data = None
            st.rerun()
    
    st.divider()
    
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
                    from scipy.stats import ttest_ind
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                    from others.evaluate_balance_multi import _smd
                    
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
                        from scipy.stats import ttest_ind
                        from others.evaluate_balance_multi import _smd
                        
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


def render_rebalancing():
    """Render the rebalancing section"""
    # Initialize session state
    if 'rebalanced_data' not in st.session_state:
        st.session_state.rebalanced_data = None
    if 'rebalancing_config' not in st.session_state:
        st.session_state.rebalancing_config = None
    
    if st.session_state.get('rebalancer_uploaded_data') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    # Use filtered data if available, otherwise use original data
    if st.session_state.get('rebalancer_filtered_data') is not None:
        df = st.session_state.rebalancer_filtered_data.copy()
        st.info(f"📁 Using filtered data: {len(df)} rows")
    else:
        df = st.session_state.rebalancer_uploaded_data.copy()
        st.info(f"📁 Using uploaded data: {len(df)} rows (no filters applied)")
    
    # Check if group column is selected
    if 'rebalancer_upload_group_column' not in st.session_state or not st.session_state.rebalancer_upload_group_column:
        st.warning("⚠️ Please select a group column in the 'Configuration' tab first")
        return
    
    group_column = st.session_state.rebalancer_upload_group_column
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        st.error("❌ Need at least 2 groups for rebalancing")
        return
    
    st.header("⚖️ Rebalancing Configuration")
    
    st.info(f"📊 Rebalancing {n_groups} groups: {', '.join(map(str, groups))}")
    
    # Show current group sizes
    group_sizes = df[group_column].value_counts()
    col_size1, col_size2 = st.columns(2)
    with col_size1:
        st.write("**Current Group Sizes:**")
        for group_name in groups:
            size = group_sizes.get(group_name, 0)
            st.metric(str(group_name), f"{size:,}", f"{size/len(df)*100:.1f}%")
    
    with col_size2:
        st.write("**Group Size Statistics:**")
        sizes_list = [group_sizes.get(g, 0) for g in groups]
        st.metric("Min Size", f"{min(sizes_list):,}")
        st.metric("Max Size", f"{max(sizes_list):,}")
        st.metric("Size Difference", f"{max(sizes_list) - min(sizes_list):,}")
    
    st.divider()
    
    # Rebalancing mode
    st.subheader("⚙️ Rebalancing Mode")
    rebalancing_mode = st.radio(
        "Select Rebalancing Mode",
        options=["Basic", "Advanced"],
        index=1,
        horizontal=True,
        key="rebalancing_mode",
        help="Basic: Even-size seed search. Advanced: Iterative rebalancing with middle/odd group strategy"
    )
    
    # Get columns for balancing (exclude ID columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out group column and ID columns
    numeric_cols = [c for c in numeric_cols if c != group_column and not is_id_column(df, c)]
    categorical_cols = [c for c in categorical_cols if c != group_column and not is_id_column(df, c)]
    
    if not numeric_cols and not categorical_cols:
        st.error("❌ No valid columns available for balancing (excluding group column)")
        return
    
    st.divider()
    
    # Value columns selection
    st.subheader("📊 Select Value Columns")
    value_columns = st.multiselect(
        "Numeric columns to balance",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))] if numeric_cols else [],
        key="rebalancer_value_columns",
        help="Select numeric columns to use for balance calculation"
    )
    
    # Stratification columns selection
    strat_columns = st.multiselect(
        "Categorical columns to balance",
        options=categorical_cols,
        default=categorical_cols[:min(3, len(categorical_cols))] if categorical_cols else [],
        key="rebalancer_strat_columns",
        help="Select categorical columns to use for balance calculation"
    )
    
    if not value_columns and not strat_columns:
        st.warning("⚠️ Please select at least one value or stratification column")
        return
    
    st.divider()
    
    # Balancing Objectives
    st.subheader("🎯 Balancing Objectives")
    st.markdown("Set target metrics for balancing. The rebalancer will optimize towards these targets.")
    
    col_obj1, col_obj2 = st.columns(2)
    
    numeric_p_values = {}
    with col_obj1:
        st.markdown("**Numeric Balance Targets:**")
        if value_columns:
            for col in value_columns:
                p_val = st.number_input(
                    f"Target p-value: {col}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.95,
                    step=0.01,
                    key=f"rebalancer_p_value_{col}",
                    help="Target p-value for t-test between groups. Use negative value to maximize p-value."
                )
                numeric_p_values[col] = p_val
        else:
            st.info("No numeric columns selected")
    
    categorical_imbalance = {}
    with col_obj2:
        st.markdown("**Categorical Balance Targets:**")
        if strat_columns:
            for col in strat_columns:
                imbalance = st.number_input(
                    f"Max imbalance %: {col}",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                    key=f"rebalancer_imbalance_{col}",
                    help="Maximum allowed total imbalance percentage for this categorical column"
                )
                categorical_imbalance[col] = imbalance
        else:
            st.info("No categorical columns selected")
    
    st.divider()
    
    # Rebalancing parameters
    st.subheader("⚙️ Rebalancing Parameters")
    
    if rebalancing_mode == "Basic":
        # Basic mode: Even-size seed search
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            enable_seed_search = st.checkbox(
                "Enable Even Size Seed Search",
                value=True,
                key="rebalancer_enable_seed_search",
                help="Subsample all groups to smallest size, minimizing total loss"
            )
        
        with col_param2:
            if enable_seed_search:
                even_size_trials = st.number_input(
                    "Even Size Seed Trials",
                    min_value=0,
                    max_value=100000,
                    value=1000,
                    step=100,
                    key="rebalancer_even_size_trials",
                    help="Number of random seeds to try for even-size subsampling"
                )
            else:
                even_size_trials = 0
    
    else:  # Advanced mode
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            max_removals = st.number_input(
                "Max Removals",
                min_value=1,
                max_value=10000,
                value=100,
                step=10,
                key="rebalancer_max_removals",
                help="Maximum number of rows to remove per group"
            )
        
        with col_param2:
            top_k_candidates = st.number_input(
                "Top K Candidates",
                min_value=1,
                max_value=1000,
                value=10,
                step=10,
                key="rebalancer_top_k",
                help="Number of top candidates to consider for trimming"
            )
        
        with col_param3:
            k_random_candidates = st.number_input(
                "Random Candidates",
                min_value=0,
                max_value=10000,
                value=200,
                step=10,
                key="rebalancer_random_candidates",
                help="Number of random candidates to consider"
            )
        
        col_param4, col_param5 = st.columns(2)
        
        with col_param4:
            gain_threshold = st.number_input(
                "Gain Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key="rebalancer_gain_threshold",
                help="Minimum gain required to make a move"
            )
        
        with col_param5:
            enable_seed_search = st.checkbox(
                "Enable Even Size Seed Search",
                value=False,
                key="rebalancer_enable_seed_search_advanced",
                help="First subsample all groups to smallest size"
            )
            
            if enable_seed_search:
                even_size_trials = st.number_input(
                    "Even Size Seed Trials",
                    min_value=0,
                    max_value=100000,
                    value=1000,
                    step=100,
                    key="rebalancer_even_size_trials_advanced",
                    help="Number of random seeds to try"
                )
            else:
                even_size_trials = 0
    
    # Continue rebalancing option
    can_continue = (
        st.session_state.rebalanced_data is not None and 
        st.session_state.rebalancing_config is not None
    )
    
    continue_rebalancing = False
    if can_continue:
        continue_rebalancing = st.checkbox(
            "🔄 Continue Rebalancing from Current State",
            value=False,
            key="rebalancer_continue",
            help="Continue rebalancing from the previously rebalanced groups. This will use the current rebalanced state as the starting point."
        )
    
    st.divider()
    
    # Rebalance button
    if st.button("⚖️ Start Rebalancing", type="primary", use_container_width=True):
        try:
            # If continuing, use existing rebalanced data (skip even-size seed search in Basic mode)
            if continue_rebalancing:
                if st.session_state.rebalanced_data is None:
                    st.error("❌ No previous rebalanced data found. Please run rebalancing first.")
                    return
                df_to_use = st.session_state.rebalanced_data.copy()
                st.info("🔄 Continuing from previous rebalanced state")
            else:
                df_to_use = df.copy()
            
            # Initialize rebalancer
            rebalancer = MultiGroupRebalancer(
                group_column=group_column,
                value_columns=value_columns,
                strat_columns=strat_columns
            )
            
            # Set objectives
            objective = {
                'numeric_p_value': numeric_p_values,
                'categorical_total_imbalance': categorical_imbalance
            }
            rebalancer.set_objective(objective)
            
            # If continuing, restore loss history
            if continue_rebalancing and st.session_state.rebalancing_config:
                existing_loss_history = st.session_state.rebalancing_config.get('loss_history', [])
                if existing_loss_history:
                    rebalancer.loss_history = existing_loss_history.copy()
            
            # Progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_callback = create_streamlit_progress_callback(progress_placeholder, status_placeholder)
            
            if rebalancing_mode == "Basic":
                # Basic mode: Even-size seed search only
                # If continuing, skip even-size search (already done)
                if continue_rebalancing:
                    st.info("ℹ️ Skipping even-size seed search in continuation mode. Using current state.")
                    rebalanced_df = df_to_use.copy()
                elif even_size_trials > 0:
                    best_seed_indices = rebalancer.find_best_even_size_seed_multi(
                        df_to_use,
                        trials=even_size_trials,
                        progress_callback=progress_callback
                    )
                    # Create rebalanced dataframe with only the selected indices
                    rebalanced_df = df_to_use.loc[best_seed_indices].copy()
                else:
                    st.warning("⚠️ Even size seed search is disabled. No rebalancing performed.")
                    rebalanced_df = df_to_use.copy()
            else:
                # Advanced mode: Full rebalancing
                rebalanced_df = rebalancer.rebalance_multi_group(
                    df_to_use,
                    max_removals=int(max_removals),
                    top_k_candidates=int(top_k_candidates),
                    k_random_candidates=int(k_random_candidates),
                    gain_threshold=gain_threshold,
                    even_size_seed_trials=even_size_trials if enable_seed_search and not continue_rebalancing else 0,
                    continuation=continue_rebalancing,
                    progress_callback=progress_callback
                )
            
            # Store results
            st.session_state.rebalanced_data = rebalanced_df
            
            # Store config (combine loss histories from multiple runs)
            if continue_rebalancing and st.session_state.rebalancing_config:
                # Get existing loss history runs
                existing_config = st.session_state.rebalancing_config
                existing_loss_runs = existing_config.get('loss_history_runs', [])
                if not existing_loss_runs:
                    # If old format, convert single history to list of runs
                    old_history = existing_config.get('loss_history', [])
                    if old_history:
                        existing_loss_runs = [old_history]
                
                # Add new run
                new_run = rebalancer.loss_history if hasattr(rebalancer, 'loss_history') else []
                if new_run:
                    existing_loss_runs.append(new_run)
                
                # Combine all runs for total history
                combined_history = []
                for run in existing_loss_runs:
                    combined_history.extend(run)
                
                loss_history_runs = existing_loss_runs
                loss_history = combined_history
            else:
                # First run
                new_run = rebalancer.loss_history if hasattr(rebalancer, 'loss_history') else []
                loss_history_runs = [new_run] if new_run else []
                loss_history = new_run
            
            st.session_state.rebalancing_config = {
                'group_column': group_column,
                'value_columns': value_columns,
                'strat_columns': strat_columns,
                'mode': rebalancing_mode,
                'loss_history': loss_history,  # Combined history for backward compatibility
                'loss_history_runs': loss_history_runs,  # Separate runs for annotations
                'middle_group': rebalancer.middle_group if hasattr(rebalancer, 'middle_group') else None,
                'odd_group': rebalancer.odd_group if hasattr(rebalancer, 'odd_group') else None
            }
            
            if continue_rebalancing:
                success_msg = "✅ Additional rebalancing run complete!"
            else:
                success_msg = "✅ Even size search complete!" if rebalancing_mode == "Basic" else "✅ Rebalancing complete!"
            st.success(success_msg)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during rebalancing: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    # Show results if available
    if st.session_state.rebalanced_data is not None:
        st.divider()
        st.header("📊 Rebalancing Results")
        
        rebalanced_df = st.session_state.rebalanced_data
        config = st.session_state.rebalancing_config
        
        # Show group size changes
        st.subheader("📈 Group Size Changes")
        original_sizes = df[group_column].value_counts()
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
                st.warning(f"⚠️ Could not plot loss history: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
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
            from scipy.stats import ttest_ind
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from others.evaluate_balance_multi import _smd
            
            st.markdown("**Numeric Balance Summary:**")
            if value_columns:
                eval_data = []
                groups = sorted(rebalanced_df[group_column].unique())
                
                for col in value_columns[:10]:  # Limit for performance
                    groups_data = {}
                    for group_name in groups:
                        group_df = rebalanced_df[rebalanced_df[group_column] == group_name]
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
                st.info("No categorical columns selected")
        
        else:  # Visual Report
            balance_fig = create_balance_report_plotly(
                rebalanced_df,
                value_columns=value_columns[:5] if len(value_columns) > 5 else value_columns,
                strat_columns=strat_columns[:3] if len(strat_columns) > 3 else strat_columns,
                group_column=group_column,
                title="Rebalanced Group Balance Analysis"
            )
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
