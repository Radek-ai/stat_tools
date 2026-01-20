
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
import random
import itertools
from typing import List, Dict, Union, Tuple, Optional


class MultiGroupRebalancer:
    def __init__(self, group_column: str, value_columns: List[str], strat_columns: List[str]):
        """
        Initialize the MultiGroupRebalancer class for balancing multiple groups by trimming.

        Parameters:
        group_column (str): The name of the column containing group labels.
        value_columns (List[str]): List of numeric columns to balance.
        strat_columns (List[str]): List of categorical columns to balance.
        """
        self.group_column = group_column
        self.value_columns = value_columns
        self.strat_columns = strat_columns
        self.target_metrics = {}

    def set_objective(self, objective: Dict[str, Union[float, Dict[str, float]]]):
        """
        Set the target metrics for balancing.

        Parameters:
        objective (Dict[str, Union[float, Dict[str, float]]]): The target metrics for balancing.
            - group_size_diff: Target group size difference (optional)
            - numeric_p_value: Dict of {column_name: target_p_value}
            - categorical_total_imbalance: Dict of {column_name: max_imbalance_percent}
        """
        self.target_metrics = objective

    def _compute_pairwise_loss(self, df: pd.DataFrame, g1_name: str, g2_name: str) -> float:
        """
        Compute loss between two specific groups.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        g1_name (str): Name of first group.
        g2_name (str): Name of second group.

        Returns:
        float: The computed pairwise loss.
        """
        g1 = df[df[self.group_column] == g1_name]
        g2 = df[df[self.group_column] == g2_name]
        
        if len(g1) == 0 or len(g2) == 0:
            return float('inf')
        
        loss = 0.0
        
        # Group size loss
        n1, n2 = len(g1), len(g2)
        if n1 + n2 > 0:
            group_diff = abs(n1 - n2) / (n1 + n2)
            if self.target_metrics.get("group_size_diff"):
                loss += max(0, (group_diff - self.target_metrics["group_size_diff"]) / 
                          self.target_metrics["group_size_diff"])

        # Numeric column loss
        for col in self.value_columns:
            x1, x2 = g1[col].dropna(), g2[col].dropna()
            if len(x1) > 1 and len(x2) > 1:
                _, p = ttest_ind(x1, x2, equal_var=False)
                p_goal = self.target_metrics.get("numeric_p_value", {}).get(col)
                if p_goal is not None:
                    if p_goal < 0:
                        loss += max(0, (p - abs(p_goal)) / abs(p_goal))
                    else:
                        loss += max(0, (p_goal - p) / p_goal)

        # Categorical imbalance
        for col in self.strat_columns:
            ct = pd.crosstab(df[self.group_column], df[col], normalize='index')
            if g1_name in ct.index and g2_name in ct.index:
                diff = (ct.loc[g1_name] - ct.loc[g2_name]).abs().sum() * 100
                c_goal = self.target_metrics.get("categorical_total_imbalance", {}).get(col, 5.0)
                loss += max(0, (diff - c_goal) / c_goal)

        return loss

    def compute_total_loss(self, df: pd.DataFrame) -> float:
        """
        Compute total loss across all group pairs.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

        Returns:
        float: The total loss across all pairs.
        """
        groups = df[self.group_column].unique()
        total_loss = 0.0
        
        for g1, g2 in itertools.combinations(groups, 2):
            total_loss += self._compute_pairwise_loss(df, g1, g2)
        
        return total_loss

    def find_middle_and_odd_groups(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Find the middle (most balanced) and odd (most imbalanced) groups.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

        Returns:
        Tuple[str, str]: (middle_group, odd_group)
        """
        groups = df[self.group_column].unique()
        
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups to find middle and odd groups")
        
        # Compute all pairwise losses
        pairwise_losses = {}
        for g1, g2 in itertools.combinations(groups, 2):
            loss = self._compute_pairwise_loss(df, g1, g2)
            pairwise_losses[(g1, g2)] = loss
            pairwise_losses[(g2, g1)] = loss  # Symmetric
        
        # Find middle group (minimum total pairwise loss)
        group_total_losses = {g: 0.0 for g in groups}
        for (g1, g2), loss in pairwise_losses.items():
            if g1 != g2:  # Avoid double counting
                group_total_losses[g1] += loss
        
        middle_group = min(group_total_losses, key=group_total_losses.get)
        
        # Find odd group (maximum loss with middle)
        odd_group = None
        max_loss = -np.inf
        for g in groups:
            if g != middle_group:
                loss = pairwise_losses.get((middle_group, g), 0)
                if loss > max_loss:
                    max_loss = loss
                    odd_group = g
        
        return middle_group, odd_group

    def preselect_candidates(self, df: pd.DataFrame, group_to_trim: str, top_k: int) -> pd.Index:
        """
        Preselect candidates for removal based on z-scores.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        group_to_trim (str): The group to trim.
        top_k (int): The number of top candidates to select.

        Returns:
        pd.Index: The indices of the top candidates.
        """
        group_df = df[df[self.group_column] == group_to_trim].copy()

        if group_df.empty or top_k <= 0:
            return pd.Index([])

        scores = pd.Series(0.0, index=group_df.index)

        for col in self.value_columns:
            col_values = group_df[col]
            mean = col_values.mean()
            std = col_values.std()
            if std > 0:
                z = ((col_values - mean).abs()) / std
                scores += z

        return scores.nlargest(min(top_k, len(scores))).index

    def compute_pair_loss(self, df: pd.DataFrame, g1_name: str, g2_name: str) -> float:
        """
        Compute loss for a specific pair of groups (used during trimming).

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        g1_name (str): Name of first group.
        g2_name (str): Name of second group.

        Returns:
        float: The computed loss for this pair.
        """
        return self._compute_pairwise_loss(df, g1_name, g2_name)

    def estimate_row_impact(self, df: pd.DataFrame, row_idx: int, g1_name: str, g2_name: str, 
                           current_pair_loss: float) -> float:
        """
        Estimate the impact of removing a row on the pair loss.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        row_idx (int): The index of the row to estimate the impact for.
        g1_name (str): Name of first group in the pair.
        g2_name (str): Name of second group in the pair.
        current_pair_loss (float): The current loss for this pair.

        Returns:
        float: The estimated impact on the loss (positive = loss reduction).
        """
        new_df = df.drop(index=row_idx)
        new_pair_loss = self.compute_pair_loss(new_df, g1_name, g2_name)
        return current_pair_loss - new_pair_loss

    def choose_group_to_trim(self, df: pd.DataFrame, g1_name: str, g2_name: str, 
                            trim_from: Optional[str] = None) -> Optional[str]:
        """
        Choose which group to trim from a pair.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        g1_name (str): Name of first group.
        g2_name (str): Name of second group.
        trim_from (Optional[str]): If specified, always trim from this group.

        Returns:
        Optional[str]: The group to trim or None if no valid group.
        """
        if trim_from:
            return trim_from if trim_from in [g1_name, g2_name] else None
        
        sizes = df[self.group_column].value_counts()
        n1, n2 = sizes.get(g1_name, 0), sizes.get(g2_name, 0)

        if n1 == 0 or n2 == 0:
            return None

        if n1 == n2:
            return random.choice([g1_name, g2_name])

        # Prefer trimming from larger group
        p = 0.8 if n1 > n2 else 0.2
        if random.random() < p:
            return g1_name if n1 > n2 else g2_name
        else:
            return g2_name if n1 > n2 else g1_name

    def find_best_even_size_seed_multi(self, df: pd.DataFrame, trials: int, progress_callback=None) -> pd.Index:
        """
        Find the best random seed for subsampling all groups to the size of the smallest group,
        minimizing total loss.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        trials (int): Number of random seeds to try.

        Returns:
        pd.Index: The indices to drop for the best seed.
        """
        sizes = df[self.group_column].value_counts()
        groups = sizes.index.tolist()
        min_size = sizes.min()
        
        # If all groups are already the same size, nothing to do
        if sizes.nunique() == 1:
            return pd.Index([])
        
        best_loss = float('inf')
        best_indices = pd.Index([])
        initial_loss = self.compute_total_loss(df)
        best_gain = 0.0
        
        if progress_callback:
            progress_callback("start", {
                "total": trials,
                "initial_loss": initial_loss,
                "description": "Multi-group seed search"
            })
        else:
            progress_bar = tqdm(range(trials), desc=f"Multi-group seed search, initial loss: {initial_loss:.4f}", unit="trial")
        
        for seed in range(trials):
            drop_indices = pd.Index([])
            
            # For each group, subsample to min_size if it's larger
            for group_name in groups:
                group_size = sizes[group_name]
                if group_size > min_size:
                    group_df = df[df[self.group_column] == group_name]
                    n_to_remove = group_size - min_size
                    # Use seed + group index to get different samples per group
                    group_seed = seed + hash(group_name) % 10000
                    sampled_indices = group_df.sample(n=n_to_remove, random_state=group_seed).index
                    drop_indices = drop_indices.union(sampled_indices)
            
            if len(drop_indices) > 0:
                temp_df = df.drop(index=drop_indices)
                loss = self.compute_total_loss(temp_df)
                gain = initial_loss - loss
                
                if loss < best_loss:
                    best_loss = loss
                    best_indices = drop_indices
                    best_gain = gain
            
            if progress_callback:
                progress_callback("update", {
                    "iteration": seed + 1,
                    "total": trials,
                    "initial_loss": initial_loss,
                    "current_loss": best_loss,
                    "gain": best_gain,
                    "progress": (seed + 1) / trials
                })
            else:
                progress_bar.set_postfix(best_gain=best_gain)
        
        if progress_callback:
            progress_callback("complete", {
                "final_loss": best_loss,
                "initial_loss": initial_loss,
                "total_iterations": trials,
                "total_gain": best_gain
            })
        
        return best_indices

    def trim_pair(self, df: pd.DataFrame, g1_name: str, g2_name: str, 
                  max_removals: int = 100, top_k_candidates: int = 20,
                  k_random_candidates: int = 20, verbose: bool = False,
                  early_break_regularization: bool = True, gain_threshold: float = 0.000,
                  trim_from: Optional[str] = None, progress_callback=None,
                  step_info: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """
        Trim the DataFrame to balance a specific pair of groups.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        g1_name (str): Name of first group.
        g2_name (str): Name of second group.
        max_removals (int): The maximum number of rows to remove.
        top_k_candidates (int): The number of top candidates to preselect.
        k_random_candidates (int): The number of random candidates to consider.
        verbose (bool): Whether to print verbose output.
        early_break_regularization (bool): Whether to break early if a positive gain is found.
        gain_threshold (float): The threshold for early break based on gain.
        trim_from (Optional[str]): If specified, always trim from this group.

        Returns:
        pd.DataFrame: The balanced DataFrame.
        """
        df = df.copy()
        removals = 0
        total_gain = 0.0
        initial_pair_loss = self.compute_pair_loss(df, g1_name, g2_name)
        
        if progress_callback:
            step_text = ""
            if step_info:
                step_text = f"Step {step_info.get('current', 1)}/{step_info.get('total', 1)}: "
            progress_callback("start", {
                "total": max_removals,
                "initial_loss": initial_pair_loss,
                "description": f"{step_text}Trimming pair ({g1_name}, {g2_name})",
                "step_info": step_info
            })
        
        for removal_idx in range(max_removals):
            current_pair_loss = self.compute_pair_loss(df, g1_name, g2_name)
            
            group_to_trim = self.choose_group_to_trim(df, g1_name, g2_name, trim_from=trim_from)
            if group_to_trim is None:
                if verbose:
                    print(f"No group to trim for pair ({g1_name}, {g2_name}).")
                break

            candidates = df[df[self.group_column] == group_to_trim]
            if candidates.empty:
                if verbose:
                    print(f"No candidates left in group {group_to_trim}.")
                break

            # Get random candidates
            candidate_idxs = candidates.sample(n=min(len(candidates), k_random_candidates), 
                                             replace=False).index
            
            # Add preselected candidates
            if top_k_candidates > 0:
                promising_candidate_idxs = self.preselect_candidates(df, group_to_trim, top_k=top_k_candidates)
                candidate_idxs = promising_candidate_idxs.union(candidate_idxs)

            best_delta = -np.inf
            best_idx = None
            for idx in candidate_idxs:
                delta = self.estimate_row_impact(df, idx, g1_name, g2_name, current_pair_loss)
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
                    if early_break_regularization and best_delta > 0:
                        break

            if best_delta <= gain_threshold:
                if verbose:
                    print(f"No improvement possible for pair ({g1_name}, {g2_name}).")
                break

            df = df.drop(index=best_idx)
            removals += 1
            total_gain += best_delta
            current_pair_loss = self.compute_pair_loss(df, g1_name, g2_name)
            
            if verbose:
                print(f"Removed index {best_idx} from {group_to_trim}, Δloss = {best_delta:.4f}")
            
            if progress_callback:
                progress_callback("update", {
                    "iteration": removal_idx + 1,
                    "total": max_removals,
                    "initial_loss": initial_pair_loss,
                    "current_loss": current_pair_loss,
                    "gain": best_delta,
                    "progress": (removal_idx + 1) / max_removals,
                    "step_info": step_info
                })
        
        if progress_callback:
            final_pair_loss = self.compute_pair_loss(df, g1_name, g2_name)
            progress_callback("complete", {
                "final_loss": final_pair_loss,
                "initial_loss": initial_pair_loss,
                "total_iterations": removals,
                "total_gain": total_gain
            })
        
        return df

    def rebalance_multi_group(self, df: pd.DataFrame, max_removals: int = 100,
                             top_k_candidates: int = 20, k_random_candidates: int = 20,
                             verbose: bool = False, early_break_regularization: bool = True,
                             gain_threshold: float = 0.000, even_size_seed_trials: int = 0,
                             progress_callback=None, continuation: bool = False) -> pd.DataFrame:
        """
        Rebalance multiple groups by finding middle/odd groups and balancing sequentially.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        max_removals (int): The maximum number of rows to remove per pair.
        top_k_candidates (int): The number of top candidates to preselect.
        k_random_candidates (int): The number of random candidates to consider.
        verbose (bool): Whether to print verbose output.
        early_break_regularization (bool): Whether to break early if a positive gain is found.
        gain_threshold (float): The threshold for early break based on gain.
        even_size_seed_trials (int): If >0, perform this many random seed trials to subsample
            all groups to the smallest group size, minimizing total loss.
        progress_callback (Callable, optional): Callback function for progress updates.
        continuation (bool): If True, always trim from odd group in first step to preserve
            middle group as anchor. If False (first run), allow trimming from either group.

        Returns:
        pd.DataFrame: The rebalanced DataFrame.
        """
        df = df.copy()
        self.loss_history = []
        current_step = 0
        
        # Step 1: Even size seed search (if requested)
        if even_size_seed_trials > 0:
            if verbose:
                print("Performing multi-group even size seed search...")
            # Even size search has its own callback, so we don't need step info here
            drop_indices = self.find_best_even_size_seed_multi(df, even_size_seed_trials, progress_callback=progress_callback)
            if len(drop_indices) > 0:
                df = df.drop(index=drop_indices)
                if verbose:
                    print(f"Dropped {len(drop_indices)} rows to equalize group sizes.")
        
        initial_loss = self.compute_total_loss(df)
        self.loss_history.append(initial_loss)
        
        # Step 2: Find middle and odd groups
        if verbose:
            print("Finding middle and odd groups...")
        middle_group, odd_group = self.find_middle_and_odd_groups(df)
        # Store for access after rebalancing
        self.middle_group = middle_group
        self.odd_group = odd_group
        if verbose:
            print(f"Middle group: {middle_group}, Odd group: {odd_group}")
        
        # Calculate remaining groups and total steps
        remaining_groups = [g for g in df[self.group_column].unique() 
                           if g not in [middle_group, odd_group]]
        total_steps = 1 + len(remaining_groups)  # 1 for (middle, odd) + N for (middle, X)
        
        # Step 3: Balance (middle, odd)
        # If continuation=True, always trim from odd to preserve middle as anchor
        # If continuation=False (first run), allow trimming from either group
        current_step += 1
        if verbose:
            print(f"Balancing pair ({middle_group}, {odd_group})...")
        df = self.trim_pair(df, middle_group, odd_group,
                           max_removals=max_removals,
                           top_k_candidates=top_k_candidates,
                           k_random_candidates=k_random_candidates,
                           verbose=verbose,
                           early_break_regularization=early_break_regularization,
                           gain_threshold=gain_threshold,
                           trim_from=odd_group if continuation else None,  # Trim from odd if continuation, otherwise allow either
                           progress_callback=progress_callback,
                           step_info={"current": current_step, "total": total_steps})
        self.loss_history.append(self.compute_total_loss(df))
        
        # Step 4: Balance (middle, X) for each remaining group
        for group_x in remaining_groups:
            current_step += 1
            if verbose:
                print(f"Balancing pair ({middle_group}, {group_x})...")
            # Always trim from X, not from middle (middle is anchor)
            df = self.trim_pair(df, middle_group, group_x,
                               max_removals=max_removals,
                               top_k_candidates=top_k_candidates,
                               k_random_candidates=k_random_candidates,
                               verbose=verbose,
                               early_break_regularization=early_break_regularization,
                               gain_threshold=gain_threshold,
                               trim_from=group_x,
                               progress_callback=progress_callback,
                               step_info={"current": current_step, "total": total_steps})
            self.loss_history.append(self.compute_total_loss(df))
        
        if progress_callback:
            final_loss = self.compute_total_loss(df)
            progress_callback("complete", {
                "final_loss": final_loss,
                "initial_loss": initial_loss,
                "total_iterations": len(self.loss_history) - 1,
                "total_gain": initial_loss - final_loss
            })
        
        return df
