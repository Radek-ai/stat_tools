"""
Group balancing logic for group selection page.
"""

import numpy as np
import pandas as pd
import streamlit as st

from others.balancer import MultiGroupBalancer
from group_selection.tooltips import PARAMETER_TOOLTIPS
from utils.artifact_builder import ArtifactBuilder
from utils.data_filtering import is_id_column
from utils.streamlit_errors import handle_error
from utils.streamlit_progress import create_streamlit_progress_callback


def render_group_balancing():
    """Render the group balancing section"""
    # Get artifact builder
    artifact = st.session_state.get('group_selection_artifact')
    if artifact is None:
        artifact = ArtifactBuilder(page_name='group_selection')
        st.session_state.group_selection_artifact = artifact
    
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
            help=PARAMETER_TOOLTIPS.get("n_groups", "")
        )
        
        # Group column name
        group_column = st.text_input(
            "Group Column Name",
            value="group",
            key="balancing_group_column",
            help=PARAMETER_TOOLTIPS.get("group_column_name", "")
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
                name = st.text_input(
                    f"Group {i+1} Name", 
                    value=default_name, 
                    key=f"balancing_group_name_{n_groups}_{i}",
                    help=PARAMETER_TOOLTIPS.get("group_name", "")
                )
                group_names.append(name)
            with col_prop:
                # Default proportion (equal split)
                # Use key that includes n_groups so it resets when n_groups changes
                default_prop = 1.0 / n_groups
                prop = st.number_input(
                    f"Prop {i+1}", 
                    value=default_prop, 
                    min_value=0.0, 
                    max_value=1.0, 
                    key=f"balancing_group_prop_{n_groups}_{i}",
                    help=PARAMETER_TOOLTIPS.get("group_proportion", "")
                )
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
                help=PARAMETER_TOOLTIPS.get("numeric_columns", "")
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
                help=PARAMETER_TOOLTIPS.get("categorical_columns", "")
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
    
    # Mode Selection
    st.subheader("🎛️ Group Selection Mode")
    
    # Check if continuing (read from session state, default to False)
    # This allows the checkbox (placed later) to affect mode selection
    continue_balancing = st.session_state.get("continue_balancing", False) if can_continue else False
    
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
            help=PARAMETER_TOOLTIPS.get("selection_mode", "")
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
                help=PARAMETER_TOOLTIPS.get("n_bins", "")
            )
            
        with col_basic2:
            random_seed = st.number_input(
                "Random Seed",
                value=42,
                min_value=None,
                max_value=None,
                key="basic_random_seed",
                help=PARAMETER_TOOLTIPS.get("random_seed", "")
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
                        f"Max imbalance (%) for {col}",
                        value=5.0,
                        min_value=0.0,
                        max_value=100.0,
                        step=0.5,
                        key=f"balancing_imbalance_{col}",
                        help=PARAMETER_TOOLTIPS.get("max_imbalance_percent", "")
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
            help=PARAMETER_TOOLTIPS.get("algorithm", "")
        )
        
        # Batch mode toggle
        use_batch_mode = st.checkbox(
            "Use Batch Mode (Less Overfitting)",
            value=False,
            key="balancing_batch_mode",
            help=PARAMETER_TOOLTIPS.get("batch_mode", "")
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
                    help=PARAMETER_TOOLTIPS.get("max_iterations", "")
                )
            
            with col_batch2:
                subset_size = st.number_input(
                    "Batch Size (Subset Size)",
                    value=5,
                    min_value=1,
                    max_value=50,
                    key="balancing_subset_size",
                    help=PARAMETER_TOOLTIPS.get("batch_size", "")
                )
            
            with col_batch3:
                n_samples = st.number_input(
                    "Random Samples",
                    value=10,
                    min_value=1,
                    max_value=100,
                    key="balancing_n_samples",
                    help=PARAMETER_TOOLTIPS.get("random_samples", "")
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
                    help=PARAMETER_TOOLTIPS.get("gain_threshold", "")
                )
            
            with col_batch5:
                early_break = st.checkbox(
                    "Early Break",
                    value=True,
                    key="balancing_early_break",
                    help=PARAMETER_TOOLTIPS.get("early_break", "")
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
                    help=PARAMETER_TOOLTIPS.get("max_iterations", "")
                )
                
                top_k_candidates = st.number_input(
                    "Top K Candidates",
                    value=default_top_k,
                    min_value=1,
                    max_value=100,
                    key="balancing_top_k",
                    help=PARAMETER_TOOLTIPS.get("top_k_candidates", "")
                )
            
            with col_alg2:
                k_random_candidates = st.number_input(
                    "Random Candidates",
                    value=default_k_random,
                    min_value=1,
                    max_value=100,
                    key="balancing_k_random",
                    help=PARAMETER_TOOLTIPS.get("k_random_candidates", "")
                )
                
                gain_threshold = st.number_input(
                    "Gain Threshold",
                    value=0.001,
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                    key="balancing_gain_threshold",
                    help=PARAMETER_TOOLTIPS.get("gain_threshold", "")
                )
            
            with col_alg3:
                early_break = st.checkbox(
                    "Early Break",
                    value=True,
                    key="balancing_early_break",
                    help=PARAMETER_TOOLTIPS.get("early_break", "")
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
    
    # Continue balancing checkbox (placed right before the button, similar to rebalancer)
    # Note: The checkbox value is read earlier to determine mode, but displayed here for better UX
    if can_continue:
        continue_balancing = st.checkbox(
            "🔄 Continue Balancing from Current State",
            value=st.session_state.get("continue_balancing", False),
            key="continue_balancing",
            help=PARAMETER_TOOLTIPS.get("continue_balancing", "")
        )
        # If checkbox is checked, force Advanced mode (will take effect on next rerun)
        if continue_balancing:
            selection_mode = "Advanced"
    else:
        continue_balancing = False
    
    if continue_balancing:
        button_label = "🔄 Continue Balancing"
    else:
        button_label = "🚀 Create Groups" if selection_mode == "Basic" else "🚀 Run Balancing"
    if st.button(button_label, type="primary", use_container_width=True):
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
                    progress_callback = create_streamlit_progress_callback(
                        progress_placeholder,
                        status_placeholder,
                        default_description="Balancing",
                        show_step_info=False,
                    )
                    
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
                
                # Add to artifact
                artifact = st.session_state.get('group_selection_artifact')
                if artifact:
                    artifact.add_df('balanced_data', balanced_df, 'Final balanced groups')
                    
                    # Determine run number for log message
                    existing_runs = artifact.config.get('balancing_runs', 0)
                    run_number = existing_runs + 1 if continue_balancing else 1
                    run_label = f" (Run {run_number})" if continue_balancing else ""
                    
                    artifact.add_log(
                        category='balancing',
                        message=f'Group balancing complete{run_label}: {len(balanced_df)} rows assigned to {len(balanced_df[group_column].unique())} groups',
                        details={
                            'mode': selection_mode,
                            'n_groups': len(balanced_df[group_column].unique()),
                            'group_names': sorted(balanced_df[group_column].unique().tolist()),
                            'batch_mode': batch_mode_value,
                            'is_continuation': continue_balancing,
                            'run_number': run_number
                        }
                    )
                
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
                
                # Add balancing config to artifact
                artifact = st.session_state.get('group_selection_artifact')
                if artifact:
                    # Track continuation runs
                    existing_runs = artifact.config.get('balancing_runs', 0)
                    if continue_balancing:
                        balancing_runs = existing_runs + 1
                    else:
                        balancing_runs = 1
                    
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
                        'balancing_mode': selection_mode,
                        'group_column': group_column,
                        'n_groups': len(group_names),
                        'group_names': group_names,
                        'group_proportions': group_proportions,
                        'value_columns': value_columns,
                        'strat_columns': strat_columns,
                        'objectives': {
                            'numeric_p_values': numeric_p_values,
                            'categorical_imbalance': categorical_imbalance
                        } if selection_mode == "Advanced" else {},
                        'balancing_runs': balancing_runs,
                        'loss_history_summary': loss_summary
                    })
                
                if continue_balancing:
                    success_msg = "✅ Additional balancing run complete!"
                else:
                    success_msg = "✅ Groups created successfully!" if selection_mode == "Basic" else "✅ Balancing complete!"
                st.success(success_msg)
                st.rerun()
                
            except Exception as e:
                handle_error(e, "Error during balancing")
    
    # Show results if available
    if st.session_state.balanced_data is not None:
        # Import and call results display
        from group_selection.results import render_balancing_results
        render_balancing_results(df)
