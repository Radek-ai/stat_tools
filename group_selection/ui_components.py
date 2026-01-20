"""
UI components for group selection page
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable

# Import filtering utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_filtering import (
    remove_nans,
    filter_outliers_percentile,
    filter_outliers_iqr,
    filter_by_value_range,
    filter_by_categorical_values,
    is_id_column
)


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
            desc = info.get("description", "Balancing")
            initial_loss = info.get("initial_loss", 0.0)
            status_placeholder.info(f"🔄 {desc} - Initial loss: {initial_loss:.4f}")
        
        elif stage == "update":
            iteration = info.get("iteration", 0)
            total = info.get("total", 1)
            initial_loss = info.get("initial_loss", 0.0)
            current_loss = info.get("current_loss", 0.0)
            gain = info.get("gain", 0.0)
            progress = info.get("progress", 0.0)
            
            progress_placeholder.progress(progress)
            status_placeholder.info(
                f"🔄 Iteration {iteration} / {total} | "
                f"Initial: {initial_loss:.4f} | "
                f"Current: {current_loss:.4f} | "
                f"Gain: {gain:.4f}"
            )
        
        elif stage == "complete":
            progress_placeholder.progress(1.0)
            final_loss = info.get("final_loss", 0.0)
            initial_loss = info.get("initial_loss", 0.0)
            total_gain = info.get("total_gain", 0.0)
            status_placeholder.success(
                f"✅ Complete! Final loss: {final_loss:.4f} | "
                f"Total gain: {total_gain:.4f}"
            )
    
    return callback


def render_data_upload():
    """Render the data upload section"""
    # Initialize session state
    if 'uploaded_data_raw' not in st.session_state:
        st.session_state.uploaded_data_raw = None
    
    st.header("📤 Upload Data")
    st.markdown("Upload CSV data for group balancing")
    
    # Dummy data loader expander
    with st.expander("🎲 Load Dummy Data", expanded=False):
        st.markdown("Load pre-generated sample data for testing")
        
        if st.button("🎲 Load Dummy Data", key="group_load_dummy", type="primary"):
            import os
            dummy_file = os.path.join("dummy_data", "group_selection_dummy.csv")
            if os.path.exists(dummy_file):
                df = pd.read_csv(dummy_file)
                st.session_state.uploaded_data_raw = df
                st.session_state.uploaded_filename = "dummy_data.csv"
                st.success(f"✅ Dummy data loaded! ({len(df)} rows, {len(df.columns)} columns)")
                st.rerun()
            else:
                st.error(f"❌ Dummy data file not found: {dummy_file}")
                st.info("💡 Run 'python dummy_data_builders/generate_all_dummy_data.py' to generate the files")
    
    uploaded_file = st.file_uploader(
        "Or choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your data for group balancing"
    )
    
    # Check if we have data (either from upload or dummy)
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data_raw = df
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            if 'uploaded_data_raw' in st.session_state:
                del st.session_state.uploaded_data_raw
            df = None
    
    # Also check if data was loaded from dummy
    if df is None and st.session_state.uploaded_data_raw is not None:
        df = st.session_state.uploaded_data_raw
    
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
            st.dataframe(df.head(20), use_container_width=True)
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
    """Render the configuration section (filtering and group setup)"""
    # Initialize session state
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    
    if st.session_state.get('uploaded_data_raw') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    st.header("⚙️ Configuration")
    st.markdown("Apply filters and configure group settings")
    
    # Data Filtering Section
    st.subheader("🔍 Data Filtering")
    st.markdown("Apply filters to remove outliers or filter by specific values")
    
    df = st.session_state.uploaded_data_raw.copy()
    filtered_df = df.copy()
    
    # Filtering tabs
    tab_outliers, tab_values = st.tabs(["Outlier Filtering", "Value-Based Filtering"])
    
    with tab_outliers:
        st.subheader("📉 Outlier Filtering")
        st.markdown("Remove or clip outliers from numeric columns")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for outlier filtering")
        else:
                # Select column for outlier filtering
                outlier_column = st.selectbox(
                    "Select Column for Outlier Filtering",
                    options=numeric_cols,
                    key="outlier_column"
                )
                
                # Outlier method
                outlier_method = st.selectbox(
                    "Outlier Method",
                    options=["None", "Percentile", "IQR"],
                    index=0,
                    key="outlier_method",
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
                            key="p_low",
                            help="Lower percentile (0-100)"
                        )
                    with col_p2:
                        p_high = st.number_input(
                            "Upper Percentile",
                            min_value=0.0,
                            max_value=100.0,
                            value=99.0,
                            step=0.1,
                            key="p_high",
                            help="Upper percentile (0-100)"
                        )
                    
                    if p_low >= p_high:
                        st.error("❌ Lower percentile must be less than upper percentile")
                    else:
                        # Show what will be removed
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
                    iqr_multiplier = st.number_input(
                        "IQR Multiplier",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.5,
                        step=0.1,
                        key="iqr_multiplier",
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
        
        if numeric_cols:
            filter_numeric_col = st.selectbox(
                "Select Numeric Column",
                options=["None"] + numeric_cols,
                key="filter_numeric_col"
            )
            
            if filter_numeric_col != "None":
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    min_val = st.number_input(
                        "Minimum Value",
                        value=None,
                        key="min_val",
                        help="Minimum value (inclusive), leave empty for no limit"
                    )
                with col_v2:
                    max_val = st.number_input(
                        "Maximum Value",
                        value=None,
                        key="max_val",
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
        
        if categorical_cols:
            filter_cat_col = st.selectbox(
                "Select Categorical Column",
                options=["None"] + categorical_cols,
                key="filter_cat_col"
            )
            
            if filter_cat_col != "None":
                unique_vals = df[filter_cat_col].unique().tolist()
                
                filter_mode = st.radio(
                    "Filter Mode",
                    options=["Keep selected values", "Exclude selected values"],
                    key="filter_mode"
                )
                
                if filter_mode == "Keep selected values":
                    keep_vals = st.multiselect(
                        "Values to Keep",
                        options=unique_vals,
                        key="keep_vals"
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
                        key="exclude_vals"
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
        if st.button("✅ Apply Filters", type="primary", use_container_width=True):
            st.session_state.filtered_data = filtered_df
            st.success(f"✅ Filters applied! {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}% retained)")
            st.rerun()
    
    with col_btn2:
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state.filtered_data = None
            st.rerun()
    
    # Show filtered data summary
    if st.session_state.filtered_data is not None:
        st.subheader("📊 Filtered Data Summary")
        filtered_df = st.session_state.filtered_data
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Original Rows", f"{len(df):,}")
        with col_sum2:
            st.metric("Filtered Rows", f"{len(filtered_df):,}")
        with col_sum3:
            removed = len(df) - len(filtered_df)
            st.metric("Removed", f"{removed:,}", f"-{removed/len(df)*100:.1f}%")
        
        with st.expander("📋 Filtered Data Preview"):
            st.dataframe(filtered_df.head(20), use_container_width=True)
    
    st.divider()
    
    # # Group Configuration Section
    # st.subheader("⚙️ Group Configuration")
    # st.markdown("Configure group names, proportions, and columns for balancing")
    
    if st.session_state.filtered_data is not None:
        df_for_balancing = st.session_state.filtered_data
    elif st.session_state.uploaded_data_raw is not None:
        df_for_balancing = st.session_state.uploaded_data_raw
    else:
        st.warning("⚠️ Please upload data and apply filters (if needed) first")
        return
    
    # Group configuration will be handled in render_group_balancing
    # This section is just a placeholder to show it's part of configuration
    # st.info("💡 Group configuration (names, proportions, columns) is available in the 'Group Balancing' tab after applying filters")


def render_group_balancing():
    """Render the group balancing section"""
    st.header("⚖️ Group Balancing")
    
    # Check if we have data
    if 'filtered_data' in st.session_state and st.session_state.filtered_data is not None:
        df = st.session_state.filtered_data.copy()
        st.info(f"📁 Using filtered data: {len(df)} rows")
    elif 'uploaded_data_raw' in st.session_state and st.session_state.uploaded_data_raw is not None:
        df = st.session_state.uploaded_data_raw.copy()
        st.info(f"📁 Using uploaded data: {len(df)} rows (no filters applied)")
    else:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    # Initialize session state for balancing
    if 'balanced_data' not in st.session_state:
        st.session_state.balanced_data = None
    if 'balancing_config' not in st.session_state:
        st.session_state.balancing_config = None
    
    st.divider()
    
    # Configuration Section
    st.subheader("⚙️ Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.markdown("**Group Settings**")
        
        # Number of groups
        n_groups = st.number_input(
            "Number of Groups",
            min_value=2,
            max_value=10,
            value=2,
            key="balancing_n_groups",
            help="Number of groups to create (e.g., 2 for control/treatment)"
        )
        
        # Group column name
        group_column = st.text_input(
            "Group Column Name",
            value="group",
            key="balancing_group_column",
            help="Name of the column that will store group assignments"
        )
        
        # Group names and proportions
        group_names = []
        group_proportions = []
        
        # Always use vertical layout
        st.markdown("**Group Names and Proportions:**")
        
        for i in range(n_groups):
            col_name, col_prop = st.columns([3, 1])
            with col_name:
                # Default names
                if n_groups == 2 and i == 0:
                    default_name = "control"
                elif n_groups == 2 and i == 1:
                    default_name = "treatment"
                else:
                    default_name = f"group_{i+1}"
                
                # Use key that includes n_groups so widgets reset when n_groups changes
                name = st.text_input(f"Group {i+1} Name", value=default_name, key=f"balancing_group_name_{n_groups}_{i}")
                group_names.append(name)
            with col_prop:
                # Default proportion (equal split)
                # Use key that includes n_groups so it resets when n_groups changes
                default_prop = 1.0 / n_groups
                prop = st.number_input(f"Prop {i+1}", value=default_prop, min_value=0.0, max_value=1.0, key=f"balancing_group_prop_{n_groups}_{i}")
                group_proportions.append(prop)
        
        # Normalize proportions
        total_prop = sum(group_proportions)
        if total_prop > 0:
            group_proportions = [p / total_prop for p in group_proportions]
            if abs(total_prop - 1.0) > 0.01:
                st.info(f"💡 Proportions normalized to sum to 1.0")
    
    with col_config2:
        st.markdown("**Column Selection**")
        
        # Numeric columns (exclude ID columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not is_id_column(df, col)]
        if numeric_cols:
            value_columns = st.multiselect(
                "Numeric Columns (for balancing)",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else [],
                key="balancing_value_columns",
                help="Select numeric columns to balance between groups"
            )
        else:
            st.warning("⚠️ No numeric columns found in data")
            value_columns = []
        
        # Categorical columns (exclude ID columns)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if not is_id_column(df, col)]
        if categorical_cols:
            strat_columns = st.multiselect(
                "Categorical Columns (for stratification)",
                options=categorical_cols,
                default=categorical_cols[:min(2, len(categorical_cols))] if categorical_cols else [],
                key="balancing_strat_columns",
                help="Select categorical columns for stratified balancing"
            )
        else:
            st.info("ℹ️ No categorical columns found")
            strat_columns = []
    
    # Validation
    if not value_columns and not strat_columns:
        st.error("❌ Please select at least one numeric or categorical column")
        return
    
    if len(group_names) != len(set(group_names)):
        st.error("❌ Group names must be unique")
        return
    
    st.divider()
    
    # Check if we can continue from existing balanced data
    can_continue = (
        st.session_state.balanced_data is not None and 
        st.session_state.balancing_config is not None and
        st.session_state.balancing_config.get('mode') == "Advanced"
    )
    
    # Continue balancing checkbox
    continue_balancing = False
    if can_continue:
        continue_balancing = st.checkbox(
            "🔄 Continue Balancing from Current State",
            value=False,
            key="continue_balancing",
            help="Continue balancing from the previously balanced groups. This will use the current balanced state as the starting point."
        )
    
    # Mode Selection
    st.subheader("🎛️ Group Selection Mode")
    
    # If continuing, force Advanced mode
    if continue_balancing:
        selection_mode = "Advanced"
        st.info("ℹ️ **Continue Balancing Mode**: Using current balanced groups as starting point. Advanced mode is required.")
    else:
        selection_mode = st.radio(
            "Selection Mode",
            options=["Basic", "Advanced"],
            index=1,
            key="selection_mode",
            help="Basic: Stratified initial group assignment only. Advanced: Initial assignment + iterative balancing."
        )
    
    if selection_mode == "Basic":
        st.info("ℹ️ **Basic Mode**: Creates stratified initial groups based on numeric and categorical columns. No iterative balancing will be performed.")
        
        # Basic mode settings
        col_basic1, col_basic2 = st.columns(2)
        with col_basic1:
            n_bins = st.number_input(
                "Number of Bins for Numeric Columns",
                value=4,
                min_value=2,
                max_value=10,
                key="basic_n_bins",
                help="Number of bins to create for numeric column stratification"
            )
        
        with col_basic2:
            random_seed = st.number_input(
                "Random Seed",
                value=42,
                min_value=None,
                max_value=None,
                key="basic_random_seed",
                help="Random seed for reproducible results (None for random)"
            )
        
        # Initialize empty objectives for basic mode (not used)
        numeric_p_values = {}
        categorical_imbalance = {}
    else:
        st.info("ℹ️ **Advanced Mode**: Creates initial stratified groups, then iteratively balances them using optimization algorithms.")
        
        # Balancing Objectives (only in advanced mode)
        st.subheader("🎯 Balancing Objectives")
        
        col_obj1, col_obj2 = st.columns(2)
        
        numeric_p_values = {}
        with col_obj1:
            st.markdown("**Numeric Balance Targets:**")
            if value_columns:
                for col in value_columns:
                    p_val = st.number_input(
                        f"Target p-value for {col}",
                        value=0.95,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        key=f"balancing_p_value_{col}",
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
                        f"Max imbalance (%) for {col}",
                        value=5.0,
                        min_value=0.0,
                        max_value=100.0,
                        step=0.5,
                        key=f"balancing_imbalance_{col}",
                        help="Maximum acceptable total imbalance percentage"
                    )
                    categorical_imbalance[col] = imbalance
            else:
                st.info("No categorical columns selected")
        
        st.divider()
        
        # Algorithm Settings
        st.subheader("🔧 Algorithm Settings")
        
        algorithm = st.selectbox(
            "Balancing Algorithm",
            options=["Sequential Moves", "Swaps"],
            index=0,
            key="balancing_algorithm",
            help="Sequential: Move rows between groups. Swaps: Swap rows between groups."
        )
        
        # Batch mode toggle
        use_batch_mode = st.checkbox(
            "Use Batch Mode (Less Overfitting)",
            value=False,
            key="balancing_batch_mode",
            help="Move/swap groups of rows at once instead of single rows. Reduces overfitting by making larger, more robust changes."
        )
        
        if use_batch_mode:
            # Batch mode settings
            st.info("ℹ️ **Batch Mode**: Moves/swaps groups of rows at once. This reduces overfitting by making larger, more robust changes.")
            
            col_batch1, col_batch2, col_batch3 = st.columns(3)
            
            with col_batch1:
                max_iterations = st.number_input(
                    "Max Iterations",
                    value=50,
                    min_value=1,
                    max_value=500,
                    key="balancing_max_iterations",
                    help="Maximum number of iterations to run"
                )
            
            with col_batch2:
                subset_size = st.number_input(
                    "Batch Size (Subset Size)",
                    value=5,
                    min_value=1,
                    max_value=50,
                    key="balancing_subset_size",
                    help="Number of rows to move/swap in each batch"
                )
            
            with col_batch3:
                n_samples = st.number_input(
                    "Random Samples",
                    value=10,
                    min_value=1,
                    max_value=100,
                    key="balancing_n_samples",
                    help="Number of random batch samples to try"
                )
            
            col_batch4, col_batch5 = st.columns(2)
            
            with col_batch4:
                gain_threshold = st.number_input(
                    "Gain Threshold",
                    value=0.001,
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                    key="balancing_gain_threshold",
                    help="Minimum gain required to continue balancing"
                )
            
            with col_batch5:
                early_break = st.checkbox(
                    "Early Break",
                    value=True,
                    key="balancing_early_break",
                    help="Stop searching candidates once a good move is found"
                )
            
            # Set defaults for single-row mode (not used but needed for consistency)
            top_k_candidates = 20
            k_random_candidates = 20
        else:
            # Single-row mode settings
            # Adjust defaults based on algorithm
            default_top_k = 10 if algorithm == "Swaps" else 20
            default_k_random = 10 if algorithm == "Swaps" else 20
            
            col_alg1, col_alg2, col_alg3 = st.columns(3)
            
            with col_alg1:
                max_iterations = st.number_input(
                    "Max Iterations",
                    value=50,
                    min_value=1,
                    max_value=500,
                    key="balancing_max_iterations",
                    help="Maximum number of iterations to run"
                )
                
                top_k_candidates = st.number_input(
                    "Top K Candidates",
                    value=default_top_k,
                    min_value=1,
                    max_value=100,
                    key="balancing_top_k",
                    help="Number of outlier candidates to preselect based on z-scores"
                )
            
            with col_alg2:
                k_random_candidates = st.number_input(
                    "Random Candidates",
                    value=default_k_random,
                    min_value=1,
                    max_value=100,
                    key="balancing_k_random",
                    help="Number of random candidates to consider"
                )
                
                gain_threshold = st.number_input(
                    "Gain Threshold",
                    value=0.001,
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                    key="balancing_gain_threshold",
                    help="Minimum gain required to continue balancing"
                )
            
            with col_alg3:
                early_break = st.checkbox(
                    "Early Break",
                    value=True,
                    key="balancing_early_break",
                    help="Stop searching candidates once a good move is found"
                )
            
            # Set defaults for batch mode (not used but needed for consistency)
            subset_size = 5
            n_samples = 10
    
    # Build objective (only used in advanced mode)
    objective = {
        'numeric_p_value': numeric_p_values,
        'categorical_total_imbalance': categorical_imbalance
    }
    
    # Run balancing button
    st.divider()
    if continue_balancing:
        button_label = "🔄 Continue Balancing"
    else:
        button_label = "🚀 Create Groups" if selection_mode == "Basic" else "🚀 Run Balancing"
    if st.button(button_label, type="primary", use_container_width=True):
        # Import balancer
        from others.balancer import MultiGroupBalancer
        
        spinner_text = "📊 Creating stratified groups..." if selection_mode == "Basic" else "⚖️ Balancing groups... This may take a while."
        with st.spinner(spinner_text):
            try:
                # Initialize balancer (needed for both new and continuing)
                balancer = MultiGroupBalancer(
                    group_column=group_column,
                    value_columns=value_columns,
                    strat_columns=strat_columns
                )
                
                # If continuing, use existing balanced data (skip initial group creation)
                if continue_balancing:
                    balanced_df = st.session_state.balanced_data.copy()
                    # Verify that the group column still exists and has the right groups
                    if group_column not in balanced_df.columns:
                        raise ValueError(f"Group column '{group_column}' not found in existing balanced data")
                else:
                    # Create initial groups
                    if selection_mode == "Basic":
                        random_seed_value = random_seed
                        n_bins_value = n_bins
                    else:
                        random_seed_value = 42
                        n_bins_value = 4
                    
                    balanced_df = balancer.create_initial_groups(
                        df,
                        group_names=group_names,
                        group_proportions=group_proportions,
                        n_bins=n_bins_value,
                        random_state=random_seed_value if random_seed_value is not None else None
                    )
                
                # Run balancing algorithm (only in advanced mode)
                if selection_mode == "Advanced":
                    balancer.set_objective(objective)
                    
                    # Create progress placeholders
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    progress_callback = create_streamlit_progress_callback(progress_placeholder, status_placeholder)
                    
                    # Check if batch mode is enabled
                    use_batch = st.session_state.get("balancing_batch_mode", False)
                    
                    if use_batch:
                        # Get batch parameters from session state
                        subset_size = st.session_state.get("balancing_subset_size", 5)
                        n_samples = st.session_state.get("balancing_n_samples", 10)
                        
                        # Use batch methods
                        if algorithm == "Sequential Moves":
                            balanced_df = balancer.balance_sequential_batch(
                                balanced_df,
                                max_iterations=int(max_iterations),
                                subset_size=int(subset_size),
                                n_samples=int(n_samples),
                                gain_threshold=gain_threshold,
                                early_break=early_break,
                                verbose=False,
                                progress_callback=progress_callback
                            )
                        else:  # Swaps
                            balanced_df = balancer.balance_swap_batch(
                                balanced_df,
                                max_iterations=int(max_iterations),
                                subset_size=int(subset_size),
                                n_samples=int(n_samples),
                                gain_threshold=gain_threshold,
                                early_break=early_break,
                                verbose=False,
                                progress_callback=progress_callback
                            )
                    else:
                        # Use single-row methods
                        if algorithm == "Sequential Moves":
                            balanced_df = balancer.balance_sequential(
                                balanced_df,
                                max_iterations=int(max_iterations),
                                top_k_candidates=int(top_k_candidates),
                                k_random_candidates=int(k_random_candidates),
                                gain_threshold=gain_threshold,
                                early_break=early_break,
                                verbose=False,
                                progress_callback=progress_callback
                            )
                        else:  # Swaps
                            balanced_df = balancer.balance_swap(
                                balanced_df,
                                max_iterations=int(max_iterations),
                                top_k_candidates=int(top_k_candidates),
                                k_random_candidates=int(k_random_candidates),
                                gain_threshold=gain_threshold,
                                early_break=early_break,
                                verbose=False,
                                progress_callback=progress_callback
                            )
                
                # Store results
                st.session_state.balanced_data = balanced_df
                batch_mode_value = st.session_state.get("balancing_batch_mode", False) if selection_mode == "Advanced" else False
                
                # Handle loss history for multiple runs
                if continue_balancing and selection_mode == "Advanced":
                    # Get existing loss history runs
                    existing_config = st.session_state.balancing_config
                    existing_loss_runs = existing_config.get('loss_history_runs', [])
                    if not existing_loss_runs:
                        # If old format, convert single history to list of runs
                        old_history = existing_config.get('loss_history')
                        if old_history:
                            existing_loss_runs = [old_history]
                    
                    # Add new run
                    new_run = balancer.loss_history if hasattr(balancer, 'loss_history') else []
                    if new_run:
                        existing_loss_runs.append(new_run)
                    
                    # Combine all runs for total history
                    combined_history = []
                    for run in existing_loss_runs:
                        combined_history.extend(run)
                    
                    loss_history_runs = existing_loss_runs
                    loss_history = combined_history
                elif selection_mode == "Advanced":
                    # First run
                    new_run = balancer.loss_history if hasattr(balancer, 'loss_history') else []
                    loss_history_runs = [new_run] if new_run else []
                    loss_history = new_run
                else:
                    loss_history_runs = None
                    loss_history = None
                
                st.session_state.balancing_config = {
                    'group_column': group_column,
                    'group_names': group_names,
                    'value_columns': value_columns,
                    'strat_columns': strat_columns,
                    'mode': selection_mode,
                    'algorithm': algorithm if selection_mode == "Advanced" else None,
                    'batch_mode': batch_mode_value,
                    'loss_history': loss_history,  # Combined history for backward compatibility
                    'loss_history_runs': loss_history_runs  # Separate runs for annotations
                }
                
                if continue_balancing:
                    success_msg = "✅ Additional balancing run complete!"
                else:
                    success_msg = "✅ Groups created successfully!" if selection_mode == "Basic" else "✅ Balancing complete!"
                st.success(success_msg)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during balancing: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Show results if available
    if st.session_state.balanced_data is not None:
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
            original_size = len(df)
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
                st.warning(f"⚠️ Could not plot loss history: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
        # Evaluation with View Switcher
        st.divider()
        st.subheader("📈 Balance Evaluation")
        
        # Generate plot immediately (for download button)
        try:
            from utils.balance_plots import create_balance_report_plotly
            balance_fig = create_balance_report_plotly(
                balanced_df,
                value_columns=config['value_columns'],
                strat_columns=config['strat_columns'],
                group_column=config['group_column'],
                title="Group Balance Analysis Report"
            )
            st.session_state.balance_report_fig = balance_fig
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
            from scipy.stats import ttest_ind
            
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
                    # Import SMD helper
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                    from others.evaluate_balance_multi import _smd
                    
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
                        from scipy.stats import ttest_ind
                        import sys
                        import os
                        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                        from others.evaluate_balance_multi import _smd
                        
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
                from power_analysis.components import render_download_button
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
