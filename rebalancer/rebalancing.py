"""
Rebalancing logic for rebalancer page.
"""
import numpy as np
import pandas as pd
import streamlit as st
from others.multi_rebalancer import MultiGroupRebalancer
from rebalancer.tooltips import PARAMETER_TOOLTIPS
from utils.artifact_builder import ArtifactBuilder
from utils.data_filtering import is_id_column
from utils.streamlit_errors import handle_error
from utils.streamlit_progress import create_streamlit_progress_callback


def render_rebalancing():
    """Render the rebalancing section"""
    # Initialize session state
    if 'rebalanced_data' not in st.session_state:
        st.session_state.rebalanced_data = None
    if 'rebalancing_config' not in st.session_state:
        st.session_state.rebalancing_config = None
    
    # Get artifact builder
    artifact = st.session_state.get('rebalancer_artifact')
    if artifact is None:
        artifact = ArtifactBuilder(page_name='rebalancer')
        st.session_state.rebalancer_artifact = artifact
    
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
        help=PARAMETER_TOOLTIPS.get("rebalancing_mode", "")
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
        help=PARAMETER_TOOLTIPS.get("numeric_columns", "")
    )
    
    # Stratification columns selection
    strat_columns = st.multiselect(
        "Categorical columns to balance",
        options=categorical_cols,
        default=categorical_cols[:min(3, len(categorical_cols))] if categorical_cols else [],
        key="rebalancer_strat_columns",
        help=PARAMETER_TOOLTIPS.get("categorical_columns", "")
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
                    help=PARAMETER_TOOLTIPS.get("target_p_value", "")
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
                    help=PARAMETER_TOOLTIPS.get("max_imbalance_percent", "")
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
                value=False,
                key="rebalancer_enable_seed_search",
                help=PARAMETER_TOOLTIPS.get("enable_seed_search", "")
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
                    help=PARAMETER_TOOLTIPS.get("even_size_trials", "")
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
                help=PARAMETER_TOOLTIPS.get("max_removals", "")
            )
        
        with col_param2:
            top_k_candidates = st.number_input(
                "Top K Candidates",
                min_value=1,
                max_value=1000,
                value=10,
                step=10,
                key="rebalancer_top_k",
                help=PARAMETER_TOOLTIPS.get("top_k_candidates", "")
            )
        
        with col_param3:
            k_random_candidates = st.number_input(
                "Random Candidates",
                min_value=0,
                max_value=10000,
                value=200,
                step=10,
                key="rebalancer_random_candidates",
                help=PARAMETER_TOOLTIPS.get("k_random_candidates", "")
            )
        
        col_param4, col_param5 = st.columns(2)
        
        with col_param4:
            gain_threshold = st.number_input(
                "Gain Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0001,
                step=0.0001,
                format="%.4f",
                key="rebalancer_gain_threshold",
                help=PARAMETER_TOOLTIPS.get("gain_threshold", "")
            )
        
        with col_param5:
            early_break = st.checkbox(
                "Early Break",
                value=False,
                key="rebalancer_early_break",
                help=PARAMETER_TOOLTIPS.get("early_break", "")
            )
            
            enable_seed_search = st.checkbox(
                "Enable Even Size Seed Search",
                value=False,
                key="rebalancer_enable_seed_search_advanced",
                help=PARAMETER_TOOLTIPS.get("enable_seed_search_advanced", "")
            )
            
            if enable_seed_search:
                even_size_trials = st.number_input(
                    "Even Size Seed Trials",
                    min_value=0,
                    max_value=100000,
                    value=1000,
                    step=100,
                    key="rebalancer_even_size_trials_advanced",
                    help=PARAMETER_TOOLTIPS.get("even_size_trials_advanced", "")
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
            help=PARAMETER_TOOLTIPS.get("continue_rebalancing", "")
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
            progress_callback = create_streamlit_progress_callback(
                progress_placeholder,
                status_placeholder,
                default_description="Rebalancing",
                show_step_info=True,
            )
            
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
                    verbose=False,
                    early_break_regularization=early_break,
                    gain_threshold=gain_threshold,
                    even_size_seed_trials=int(even_size_trials),
                    continuation=continue_rebalancing,
                    progress_callback=progress_callback
                )
            
            # Store results
            st.session_state.rebalanced_data = rebalanced_df
            
            # Add to artifact
            artifact = st.session_state.get('rebalancer_artifact')
            if artifact:
                artifact.add_df('rebalanced_data', rebalanced_df, 'Final rebalanced groups')
                
                # Determine run number for log message
                existing_runs = artifact.config.get('rebalancing_runs', 0)
                run_number = existing_runs + 1 if continue_rebalancing else 1
                run_label = f" (Run {run_number})" if continue_rebalancing else ""
                
                artifact.add_log(
                    category='rebalancing',
                    message=f'Rebalancing complete{run_label}: {len(rebalanced_df)} rows in {len(rebalanced_df[group_column].unique())} groups',
                    details={
                        'mode': rebalancing_mode,
                        'n_groups': len(rebalanced_df[group_column].unique()),
                        'group_names': sorted(rebalanced_df[group_column].unique().tolist()),
                        'is_continuation': continue_rebalancing,
                        'run_number': run_number
                    }
                )
            
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
            
            # Add rebalancing config to artifact
            artifact = st.session_state.get('rebalancer_artifact')
            if artifact:
                # Track continuation runs
                existing_runs = artifact.config.get('rebalancing_runs', 0)
                if continue_rebalancing:
                    rebalancing_runs = existing_runs + 1
                else:
                    rebalancing_runs = 1
                
                # Prepare loss history summary
                loss_summary = None
                if loss_history:
                    loss_summary = {
                        'total_iterations': len(loss_history),
                        'initial_loss': loss_history[0] if loss_history else None,
                        'final_loss': loss_history[-1] if loss_history else None,
                        'all_values': loss_history,  # Store complete loss history
                        'n_runs': len(loss_history_runs) if loss_history_runs else 1,
                        'loss_history_runs': loss_history_runs if loss_history_runs else None  # Store separate runs if available
                    }
                
                artifact.set_config({
                    'rebalancing_mode': rebalancing_mode,
                    'group_column': group_column,
                    'value_columns': value_columns,
                    'strat_columns': strat_columns,
                    'objectives': {
                        'numeric_p_values': numeric_p_values,
                        'categorical_imbalance': categorical_imbalance
                    },
                    'middle_group': rebalancer.middle_group if hasattr(rebalancer, 'middle_group') else None,
                    'odd_group': rebalancer.odd_group if hasattr(rebalancer, 'odd_group') else None,
                    'rebalancing_runs': rebalancing_runs,
                    'loss_history_summary': loss_summary
                })
            
            if continue_rebalancing:
                success_msg = "✅ Additional rebalancing run complete!"
            else:
                success_msg = "✅ Even size search complete!" if rebalancing_mode == "Basic" else "✅ Rebalancing complete!"
            st.success(success_msg)
            st.rerun()
            
        except Exception as e:
            handle_error(e, "Error during rebalancing")
    
    # Show results if available
    if st.session_state.rebalanced_data is not None:
        from rebalancer.results import render_rebalancing_results
        render_rebalancing_results(df, group_column, groups)
