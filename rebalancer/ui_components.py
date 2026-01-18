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


def render_data_upload():
    """Render the data upload section"""
    # Initialize session state
    if 'rebalancer_uploaded_data' not in st.session_state:
        st.session_state.rebalancer_uploaded_data = None
    
    st.header("📤 Upload Data with Existing Groups")
    st.markdown("Upload a CSV file that already contains group assignments. The rebalancer will trim rows to improve balance.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key="rebalancer_file_upload",
        help="Upload a CSV file with existing group assignments"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.rebalancer_uploaded_data = df
            st.session_state.rebalancer_filename = uploaded_file.name
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Group column selection (don't show values to avoid lag)
            st.subheader("📋 Select Group Column")
            group_column = st.selectbox(
                "Group Column",
                options=[""] + df.columns.tolist(),
                index=0,  # Default to empty
                key="rebalancer_upload_group_column",
                help="Select the column containing group assignments"
            )
            
            if group_column and group_column != "":
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
            
            # Display preview (without group column values to avoid lag)
            with st.expander("📋 Data Preview", expanded=True):
                preview_df = df.head(20).copy()
                if group_column and group_column != "":
                    # Show group column but don't render all values
                    preview_df[group_column] = preview_df[group_column].astype(str)
                st.dataframe(preview_df, use_container_width=True)
                st.caption(f"Showing first 20 rows of {len(df)} total rows")
            
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
            
            # Initial balance report
            if group_column and group_column != "":
                st.divider()
                st.subheader("📈 Initial Group Balance Report")
                
                # Get all numeric and categorical columns for initial report
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if numeric_cols or categorical_cols:
                    # Filter out group column from numeric/categorical if it's there
                    numeric_cols_filtered = [c for c in numeric_cols if c != group_column]
                    categorical_cols_filtered = [c for c in categorical_cols if c != group_column]
                    
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
                                        overall = ct.mean(axis=0)
                                        imbalance = (ct.sub(overall, axis=1).abs().sum(axis=1)) * 100
                                        
                                        st.write(f"**{col}:**")
                                        imbalance_df = pd.DataFrame({
                                            'Group': imbalance.index,
                                            'Total Imbalance (%)': imbalance.values
                                        })
                                        st.dataframe(imbalance_df, use_container_width=True, hide_index=True)
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
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            if 'rebalancer_uploaded_data' in st.session_state:
                del st.session_state.rebalancer_uploaded_data


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
    
    df = st.session_state.rebalancer_uploaded_data.copy()
    
    st.header("⚖️ Rebalancing Configuration")
    
    # Group column selection (use from upload if set, otherwise select)
    if 'rebalancer_upload_group_column' in st.session_state:
        group_column = st.session_state.rebalancer_upload_group_column
        if group_column and group_column != "":
            st.subheader("📋 Group Configuration")
            st.info(f"📊 Using group column: **{group_column}** (change in Data Upload tab)")
        else:
            st.subheader("📋 Group Configuration")
            st.warning("⚠️ Please select a group column in the 'Data Upload' tab first")
            group_column = None
    else:
        st.subheader("📋 Group Configuration")
        group_column = st.selectbox(
            "Group Column",
            options=df.columns.tolist(),
            key="rebalancer_group_column",
            help="Select the column containing group assignments"
        )
    
    if group_column and group_column != "":
        groups = sorted(df[group_column].unique())
        n_groups = len(groups)
        
        if n_groups < 2:
            st.error("❌ Need at least 2 groups for rebalancing")
            return
        
        st.info(f"📊 Found {n_groups} groups: {', '.join(map(str, groups))}")
        
        # Show group sizes
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
        
        # Current balance report expander
        with st.expander("📊 Current Group Balance Report", expanded=False):
            # Get all numeric and categorical columns for balance report
            numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols_all = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols_all or categorical_cols_all:
                report_view = st.radio(
                    "Report View",
                    options=["Summary", "Visual Report"],
                    index=0,
                    horizontal=True,
                    key="current_balance_view"
                )
                
                if report_view == "Visual Report":
                    current_balance_fig = create_balance_report_plotly(
                        df,
                        value_columns=numeric_cols_all[:5] if len(numeric_cols_all) > 5 else numeric_cols_all,
                        strat_columns=categorical_cols_all[:3] if len(categorical_cols_all) > 3 else categorical_cols_all,
                        group_column=group_column,
                        title="Current Group Balance Analysis"
                    )
                    st.plotly_chart(current_balance_fig, use_container_width=True)
                else:
                    # Summary view - show basic stats
                    st.write("**Numeric Columns Summary:**")
                    if numeric_cols_all:
                        summary_data = []
                        for col in numeric_cols_all[:10]:  # Limit for performance
                            for g in groups:
                                group_data = df[df[group_column] == g][col].dropna()
                                if len(group_data) > 0:
                                    summary_data.append({
                                        'Column': col,
                                        'Group': str(g),
                                        'Mean': group_data.mean(),
                                        'Std': group_data.std(),
                                        'Count': len(group_data)
                                    })
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                    else:
                        st.info("No numeric columns found")
                    
                    st.write("**Categorical Columns Summary:**")
                    if categorical_cols_all:
                        for col in categorical_cols_all[:5]:  # Limit for performance
                            st.write(f"**{col}:**")
                            ct = pd.crosstab(df[group_column], df[col], normalize='index') * 100
                            st.dataframe(ct.round(2), use_container_width=True)
                    else:
                        st.info("No categorical columns found")
            else:
                st.info("ℹ️ No numeric or categorical columns found for balance analysis")
    
    st.divider()
    
    # Column selection
    st.subheader("📊 Column Selection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        value_columns = st.multiselect(
            "Numeric Columns (for balancing)",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else [],
            key="rebalancer_value_columns",
            help="Select numeric columns to balance between groups"
        )
    else:
        st.warning("⚠️ No numeric columns found in data")
        value_columns = []
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        strat_columns = st.multiselect(
            "Categorical Columns (for stratification)",
            options=categorical_cols,
            default=categorical_cols[:min(2, len(categorical_cols))] if categorical_cols else [],
            key="rebalancer_strat_columns",
            help="Select categorical columns for stratified balancing"
        )
    else:
        st.info("ℹ️ No categorical columns found")
        strat_columns = []
    
    if not value_columns and not strat_columns:
        st.error("❌ Please select at least one numeric or categorical column")
        return
    
    st.divider()
    
    # Check if we can continue from existing rebalanced data
    can_continue = (
        st.session_state.get('rebalanced_data') is not None and 
        st.session_state.get('rebalancing_config') is not None and
        st.session_state.rebalancing_config.get('mode') == "Advanced"
    )
    
    # Continue rebalancing checkbox
    continue_rebalancing = False
    if can_continue:
        continue_rebalancing = st.checkbox(
            "🔄 Continue Rebalancing from Current State",
            value=False,
            key="continue_rebalancing",
            help="Continue rebalancing from the previously rebalanced groups. This will use the current rebalanced state as the starting point."
        )
    
    # Mode Selection
    st.subheader("🎛️ Rebalancing Mode")
    
    # If continuing, force Advanced mode
    if continue_rebalancing:
        rebalancing_mode = "Advanced"
        st.info("ℹ️ **Continue Rebalancing Mode**: Using current rebalanced groups as starting point. Advanced mode is required.")
    else:
        rebalancing_mode = st.radio(
            "Rebalancing Mode",
            options=["Basic", "Advanced"],
            index=0,
            key="rebalancing_mode",
            help="Basic: Even size seed search only. Advanced: Full iterative rebalancing."
        )
    
    if rebalancing_mode == "Basic":
        st.info("ℹ️ **Basic Mode**: Subsamples all groups to the smallest group size using seed search to minimize loss.")
        
        col_basic1, col_basic2 = st.columns(2)
        with col_basic1:
            use_even_size_basic = st.checkbox(
                "Enable Even Size Seed Search",
                value=True,
                key="rebalancer_use_even_size_basic",
                help="Subsample all groups to smallest size"
            )
            
            if use_even_size_basic:
                even_size_trials = st.number_input(
                    "Seed Search Trials",
                    value=1000,
                    min_value=1,
                    max_value=100000,
                    step=100,
                    key="rebalancer_even_size_trials_basic",
                    help="Number of random seeds to try for even size subsampling"
                )
            else:
                even_size_trials = 0
        
        # Initialize empty objectives for basic mode
        numeric_p_values = {}
        categorical_imbalance = {}
        group_size_diff = None
        # even_size_trials is already defined above in Basic mode
    else:
        st.info("ℹ️ **Advanced Mode**: Full iterative rebalancing with configurable objectives.")
        
        # Balancing Objectives
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
                        f"Max imbalance (%) for {col}",
                        value=5.0,
                        min_value=0.0,
                        max_value=100.0,
                        step=0.5,
                        key=f"rebalancer_imbalance_{col}",
                        help="Maximum acceptable total imbalance percentage"
                    )
                    categorical_imbalance[col] = imbalance
            else:
                st.info("No categorical columns selected")
        
        st.divider()
        
        # Algorithm Settings
        st.subheader("🔧 Algorithm Settings")
        
        col_alg1, col_alg2, col_alg3 = st.columns(3)
        
        with col_alg1:
            max_removals = st.number_input(
                "Max Removals per Pair",
                value=100,
                min_value=1,
                max_value=50000,
                step=100,
                key="rebalancer_max_removals",
                help="Maximum number of rows to remove per group pair"
            )
            
            top_k_candidates = st.number_input(
                "Top K Candidates",
                value=20,
                min_value=1,
                max_value=1000,
                step=10,
                key="rebalancer_top_k",
                help="Number of outlier candidates to preselect based on z-scores"
            )
        
        with col_alg2:
            k_random_candidates = st.number_input(
                "Random Candidates",
                value=20,
                min_value=1,
                max_value=1000,
                step=10,
                key="rebalancer_k_random",
                help="Number of random candidates to consider"
            )
            
            gain_threshold = st.number_input(
                "Gain Threshold",
                value=0.000,
                min_value=0.0,
                max_value=10.0,
                format="%.4f",
                step=0.0001,
                key="rebalancer_gain_threshold",
                help="Minimum gain required to continue rebalancing"
            )
        
        with col_alg3:
            early_break = st.checkbox(
                "Early Break",
                value=True,
                key="rebalancer_early_break",
                help="Stop searching candidates once a good move is found"
            )
            
            use_even_size = st.checkbox(
                "Enable Even Size Seed Search",
                value=True,
                key="rebalancer_use_even_size",
                help="Subsample all groups to smallest size before iterative rebalancing"
            )
            
            if use_even_size:
                even_size_trials = st.number_input(
                    "Even Size Seed Trials",
                    value=1000,
                    min_value=1,
                    max_value=100000,
                    step=100,
                    key="rebalancer_even_size_trials",
                    help="Number of seed trials for initial even size subsampling"
                )
            else:
                even_size_trials = 0
        
        group_size_diff = None
    
    # Build objective
    objective = {
        'numeric_p_value': numeric_p_values,
        'categorical_total_imbalance': categorical_imbalance
    }
    if group_size_diff is not None:
        objective['group_size_diff'] = group_size_diff
    
    # Run rebalancing button
    st.divider()
    if continue_rebalancing:
        button_label = "🔄 Continue Rebalancing"
    else:
        button_label = "🚀 Run Even Size Search" if rebalancing_mode == "Basic" else "🚀 Run Rebalancing"
    if st.button(button_label, type="primary", use_container_width=True):
        from others.multi_rebalancer import MultiGroupRebalancer
        
        spinner_text = "🔍 Searching for best even size subsample..." if rebalancing_mode == "Basic" else "⚖️ Rebalancing groups... This may take a while."
        with st.spinner(spinner_text):
            try:
                # Initialize rebalancer
                rebalancer = MultiGroupRebalancer(
                    group_column=group_column,
                    value_columns=value_columns,
                    strat_columns=strat_columns
                )
                
                rebalancer.set_objective(objective)
                
                # If continuing, use existing rebalanced data
                if continue_rebalancing:
                    df = st.session_state.rebalanced_data.copy()
                
                # Run rebalancing
                if rebalancing_mode == "Basic":
                    # Only even size seed search
                    drop_indices = rebalancer.find_best_even_size_seed_multi(df, even_size_trials)
                    rebalanced_df = df.drop(index=drop_indices)
                    # No loss history for basic mode
                    loss_history = None
                else:
                    # Full rebalancing
                    rebalanced_df = rebalancer.rebalance_multi_group(
                        df,
                        max_removals=int(max_removals),
                        top_k_candidates=int(top_k_candidates),
                        k_random_candidates=int(k_random_candidates),
                        verbose=False,
                        early_break_regularization=early_break,
                        gain_threshold=gain_threshold,
                        even_size_seed_trials=int(even_size_trials)
                    )
                    new_loss_history = rebalancer.loss_history if hasattr(rebalancer, 'loss_history') else None
                    
                    # Handle loss history for multiple runs
                    if continue_rebalancing:
                        existing_config = st.session_state.rebalancing_config
                        existing_loss_runs = existing_config.get('loss_history_runs', [])
                        if not existing_loss_runs:
                            # If old format, convert single history to list of runs
                            old_history = existing_config.get('loss_history')
                            if old_history:
                                existing_loss_runs = [old_history]
                        
                        # Add new run
                        if new_loss_history:
                            existing_loss_runs.append(new_loss_history)
                        
                        # Combine all runs for total history
                        combined_history = []
                        for run in existing_loss_runs:
                            combined_history.extend(run)
                        
                        loss_history_runs = existing_loss_runs
                        loss_history = combined_history
                    else:
                        # First run
                        loss_history_runs = [new_loss_history] if new_loss_history else []
                        loss_history = new_loss_history
                
                # Store results
                st.session_state.rebalanced_data = rebalanced_df
                # Get middle and odd groups if available (only in Advanced mode)
                middle_group = None
                odd_group = None
                if rebalancing_mode == "Advanced" and hasattr(rebalancer, 'middle_group'):
                    middle_group = rebalancer.middle_group
                    odd_group = rebalancer.odd_group
                
                st.session_state.rebalancing_config = {
                    'group_column': group_column,
                    'value_columns': value_columns,
                    'strat_columns': strat_columns,
                    'mode': rebalancing_mode,
                    'loss_history': loss_history,  # Combined history for backward compatibility
                    'loss_history_runs': loss_history_runs if rebalancing_mode == "Advanced" else None,  # Separate runs for annotations
                    'middle_group': middle_group,  # Middle group chosen for rebalancing
                    'odd_group': odd_group  # Odd group chosen for rebalancing
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
    if st.session_state.get('rebalanced_data') is not None:
        st.divider()
        st.subheader("📊 Rebalancing Results")
        
        rebalanced_df = st.session_state.rebalanced_data
        config = st.session_state.rebalancing_config
        
        # Get original dataframe for comparison
        original_df = st.session_state.rebalancer_uploaded_data.copy()
        
        # Show middle/odd group information if available
        if config.get('middle_group') and config.get('odd_group'):
            st.info(f"🎯 **Rebalancing Strategy**: Middle group = **{config['middle_group']}** (most balanced), Odd group = **{config['odd_group']}** (most imbalanced)")
        
        # Group sizes comparison
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write("**Before Rebalancing:**")
            original_sizes = original_df[config['group_column']].value_counts()
            for group_name in sorted(original_df[config['group_column']].unique()):
                size = original_sizes.get(group_name, 0)
                st.metric(str(group_name), f"{size:,}", f"{size/len(original_df)*100:.1f}%")
        
        with col_res2:
            st.write("**After Rebalancing:**")
            rebalanced_sizes = rebalanced_df[config['group_column']].value_counts()
            for group_name in sorted(rebalanced_df[config['group_column']].unique()):
                size = rebalanced_sizes.get(group_name, 0)
                original_size = original_sizes.get(group_name, 0)
                change = size - original_size
                st.metric(str(group_name), f"{size:,}", f"{change:+,}")
        
        # Summary metrics
        st.write("**Summary:**")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Original Rows", f"{len(original_df):,}")
        with col_sum2:
            st.metric("Rebalanced Rows", f"{len(rebalanced_df):,}")
        with col_sum3:
            rows_removed = len(original_df) - len(rebalanced_df)
            st.metric("Rows Removed", f"{rows_removed:,}", f"{rows_removed/len(original_df)*100:.1f}%")
        
        if config.get('loss_history') and config.get('mode') == "Advanced" and len(config['loss_history']) > 1:
            st.divider()
            st.subheader("📉 Loss History")
            
            try:
                import plotly.graph_objects as go
                
                loss_history = config['loss_history']
                loss_history_runs = config.get('loss_history_runs', [])
                
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
                                showarrow=False,
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
                            showarrow=False,
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
                            showarrow=False,
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
                    
                    if len(loss_history) > 0:
                        initial_loss = loss_history[0]
                        final_loss = loss_history[-1]
                        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                        
                        fig_loss.add_annotation(
                            x=0,
                            y=initial_loss,
                            text=f"Initial: {initial_loss:.4f}",
                            showarrow=False,
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='green'
                        )
                        
                        if len(loss_history) > 1:
                            fig_loss.add_annotation(
                                x=len(loss_history) - 1,
                                y=final_loss,
                                text=f"Final: {final_loss:.4f}<br>Improvement: {improvement:.1f}%",
                                showarrow=False,
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
        st.subheader("📈 Balance Evaluation")
        
        # View switcher
        view_mode = st.radio(
            "View Mode",
            options=["Summary", "Visual Report"],
            index=0,
            horizontal=True,
            key="rebalancer_view_mode"
        )
        
        if view_mode == "Summary":
            # Generate balance report plot for download (but don't show it)
            balance_fig = create_balance_report_plotly(
                rebalanced_df,
                value_columns=config['value_columns'],
                strat_columns=config['strat_columns'],
                group_column=config['group_column'],
                title="Rebalanced Group Balance Analysis"
            )
            st.session_state.rebalancer_balance_fig = balance_fig
            
            # Show summary tables (similar to group selection)
            from scipy.stats import ttest_ind
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from others.evaluate_balance_multi import _smd
            
            st.markdown("**Numeric Balance Summary:**")
            if config['value_columns']:
                eval_data = []
                groups = sorted(rebalanced_df[config['group_column']].unique())
                
                for col in config['value_columns']:
                    groups_data = {}
                    for group_name in groups:
                        group_df = rebalanced_df[rebalanced_df[config['group_column']] == group_name]
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
            if config['strat_columns']:
                for col in config['strat_columns']:
                    try:
                        tmp = rebalanced_df[[config['group_column'], col]].copy()
                        tmp[col] = tmp[col].astype(str).fillna("__MISSING__")
                        tmp[config['group_column']] = tmp[config['group_column']].astype(str)
                        ct = pd.crosstab(tmp[config['group_column']], tmp[col], normalize="index").fillna(0)
                        overall = ct.mean(axis=0)
                        imbalance = (ct.sub(overall, axis=1).abs().sum(axis=1)) * 100
                        
                        st.write(f"**{col}:**")
                        imbalance_df = pd.DataFrame({
                            'Group': imbalance.index,
                            'Total Imbalance (%)': imbalance.values
                        })
                        st.dataframe(imbalance_df, use_container_width=True, hide_index=True)
                    except Exception:
                        st.warning(f"Could not compute imbalance for {col}")
            else:
                st.info("No categorical columns selected")
            
        else:  # Visual Report
            if st.session_state.get('rebalancer_balance_fig') is None:
                balance_fig = create_balance_report_plotly(
                    rebalanced_df,
                    value_columns=config['value_columns'],
                    strat_columns=config['strat_columns'],
                    group_column=config['group_column'],
                    title="Rebalanced Group Balance Analysis Report"
                )
                st.session_state.rebalancer_balance_fig = balance_fig
            else:
                balance_fig = st.session_state.rebalancer_balance_fig
            
            from power_analysis.components import render_plot_with_download
            render_plot_with_download(
                balance_fig,
                filename="rebalanced_balance_report.html"
            )
        
        # Download section
        st.divider()
        st.subheader("💾 Download Results")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            # Download CSV
            csv_data = rebalanced_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Data (CSV)",
                data=csv_data,
                file_name=f"rebalanced_data_{st.session_state.get('rebalancer_filename', 'data')}",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            # Download HTML plot
            if st.session_state.get('rebalancer_balance_fig') is not None:
                html_str = st.session_state.rebalancer_balance_fig.to_html()
                st.download_button(
                    "Download Plots (HTML)",
                    data=html_str,
                    file_name="rebalanced_balance_report.html",
                    mime="text/html",
                    use_container_width=True
                )
            else:
                st.download_button(
                    "Download Plots (HTML)",
                    data="",
                    file_name="",
                    disabled=True,
                    use_container_width=True
                )
        
        with col_dl3:
            # Placeholder for Excel if needed
            st.download_button(
                "Download Summary (Excel)",
                data="",
                file_name="",
                disabled=True,
                use_container_width=True
            )
