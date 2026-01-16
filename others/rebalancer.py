
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
import random
from typing import List, Dict, Union

class GroupBalancer:
    def __init__(self, group_column: str, group1_name: str, group2_name: str, 
                 value_columns: List[str], strat_columns: List[str]):
        """
        Initialize the GroupBalancer class.

        Parameters:
        group_column (str): The name of the column containing group labels.
        group1_name (str): The name of the first group.
        group2_name (str): The name of the second group.
        value_columns (List[str]): List of numeric columns to balance.
        strat_columns (List[str]): List of categorical columns to balance.
        """
        self.group_column = group_column
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.value_columns = value_columns
        self.strat_columns = strat_columns

    def set_objective(self, objective: Dict[str, Union[float, Dict[str, float]]]):
        """
        Set the target metrics for balancing.

        Parameters:
        objective (Dict[str, Union[float, Dict[str, float]]]): The target metrics for balancing.
        """
        self.target_metrics = objective

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

        if group_df.empty:
            return pd.Index([])

        scores = pd.Series(0.0, index=group_df.index)

        for col in self.value_columns:
            col_values = group_df[col]
            mean = col_values.mean()
            std = col_values.std()
            if std > 0:
                z = ((col_values - mean).abs()) / std
                scores += z

        return scores.nlargest(top_k).index

    def compute_loss(self, df: pd.DataFrame) -> float:
        """
        Compute the loss based on the current state of the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

        Returns:
        float: The computed loss.
        """
        g1 = df[df[self.group_column] == self.group1_name]
        g2 = df[df[self.group_column] == self.group2_name]
        loss = 0
        # Group size loss
        n1, n2 = len(g1), len(g2)
        group_diff = abs(n1 - n2) / (n1 + n2)
        if self.target_metrics.get("group_size_diff"):
            loss = max(0, (group_diff - self.target_metrics["group_size_diff"]) / self.target_metrics["group_size_diff"])

        # Numeric column loss
        for col in self.value_columns:
            x1, x2 = g1[col].dropna(), g2[col].dropna()
            if len(x1) > 1 and len(x2) > 1:
                _, p = ttest_ind(x1, x2, equal_var=False)
                p_goal = self.target_metrics["numeric_p_value"][col]
                if p_goal < 0:
                    loss += max(0, (p-abs(p_goal)) / abs(p_goal))
                else:   
                    loss += max(0, (p_goal - p) / p_goal)

        # Categorical imbalance
        for col in self.strat_columns:
            ct = pd.crosstab(df[self.group_column], df[col], normalize='index')
            diff = (ct.loc[self.group1_name] - ct.loc[self.group2_name]).abs().sum() * 100
            c_goal = self.target_metrics.get("categorical_total_imbalance", {}).get(col, 5.0)
            loss += max(0, (diff - c_goal) / c_goal)

        return loss

    def estimate_row_impact(self, df: pd.DataFrame, row_idx: int, current_loss: float) -> float:
        """
        Estimate the impact of removing a row on the loss.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        row_idx (int): The index of the row to estimate the impact for.
        current_loss (float): The current loss of the DataFrame.

        Returns:
        float: The estimated impact on the loss.
        """
        new_df = df.drop(index=row_idx)
        new_loss = self.compute_loss(new_df)
        return current_loss - new_loss  # Positive = loss reduction

    def choose_group_to_trim(self, df: pd.DataFrame) -> Union[str, None]:
        """
        Choose which group to trim based on the current group sizes.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

        Returns:
        Union[str, None]: The group to trim or None if no valid group.
        """
        sizes = df[self.group_column].value_counts()
        n1, n2 = sizes.get(self.group1_name, 0), sizes.get(self.group2_name, 0)

        if n1 == 0 or n2 == 0:
            return None  # No valid group

        if n1 == n2:
            return random.choice([self.group1_name, self.group2_name])

        p = 0.9 if n1 > n2 else 0.1
        if random.random() < p:
            return self.group1_name if n1 > n2 else self.group2_name
        else:
            return self.group2_name if n1 > n2 else self.group1_name

    def find_best_even_size_seed(self, df: pd.DataFrame, trials: int) -> pd.Index:
        """
        Find the best random seed for subsampling the larger group to match the smaller group size, minimizing the loss.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        trials (int): Number of random seeds to try.

        Returns:
        pd.Index: The indices to drop for the best seed.
        """
        sizes = df[self.group_column].value_counts()
        n1, n2 = sizes.get(self.group1_name, 0), sizes.get(self.group2_name, 0)
        if n1 == n2:
            return pd.Index([])
        group_to_subsample = self.group1_name if n1 > n2 else self.group2_name
        n_subsample = abs(n1 - n2)
        best_loss = float('inf')
        best_indices = None
        initial_loss = self.compute_loss(df)
        best_gain = 0.0
        progress_bar = tqdm(range(trials), desc=f"Seed search, initial loss: {initial_loss:.4f}", unit="trial")
        for seed in progress_bar:
            sampled_indices = df[df[self.group_column] == group_to_subsample].sample(n=n_subsample, random_state=seed).index
            temp_df = df.drop(index=sampled_indices)
            loss = self.compute_loss(temp_df)
            gain = initial_loss - loss
            if loss < best_loss:
                best_loss = loss
                best_indices = sampled_indices
                best_gain = gain
            progress_bar.set_postfix(best_gain=best_gain)
        return best_indices

    def trim_to_balance(self, df: pd.DataFrame, max_removals: int = 100, top_k_candidates: int = 20, 
                        k_random_candidates: int = 20, verbose: bool = False, 
                        early_break_regularization: bool = True, even_size_seed_trials: int = 0, gain_threshold: float = 0.000) -> pd.DataFrame:
        """
        Trim the DataFrame to balance the groups.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        max_removals (int): The maximum number of rows to remove.
        top_k_candidates (int): The number of top candidates to preselect based on zscore - poor regularization, but trims fewer rows.
        k_random_candidates (int): The number of random candidates to consider.
        verbose (bool): Whether to print verbose output.
        early_break_regularization (bool): Whether to break early if a positive gain is found.
        even_size_seed_trials (int): If >0, perform this many random seed trials to subsample the larger group to match the smaller group, minimizing loss. If 0, do not subsample for even sizes.
        gain_threshold (float): The threshold for early break based on gain.

        Returns:
        pd.DataFrame: The balanced DataFrame.
        """
        df = df.copy()
        removals = 0
        total_gain = 0
        self.loses = []
        if even_size_seed_trials > 0:
            drop_indices = self.find_best_even_size_seed(df, even_size_seed_trials)
            if len(drop_indices) > 0:
                df = df.drop(index=drop_indices)
        inintial_loss = self.compute_loss(df)
        progress_bar = tqdm(range(max_removals), desc="Balancing, initial loss: {0:.2f}".format(inintial_loss), unit="removal")
        for _ in progress_bar:
            current_loss = self.compute_loss(df)
            self.loses.append(current_loss)
            group_to_trim = self.choose_group_to_trim(df)
            if group_to_trim is None:
                if verbose: print("No group to trim.")
                break

            candidates = df[df[self.group_column] == group_to_trim]
            if candidates.empty:
                if verbose: print("No candidates left in group.")
                break

            candidate_idxs = candidates.sample(n=min(len(candidates), k_random_candidates)).index
            if top_k_candidates > 0:
                promising_candidate_idxs = self.preselect_candidates(df, group_to_trim, top_k=top_k_candidates)
                candidate_idxs = promising_candidate_idxs.union(candidate_idxs)

            best_delta = -np.inf
            best_idx = None
            for idx in candidate_idxs:
                delta = self.estimate_row_impact(df, idx, current_loss)
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
                    if early_break_regularization and best_delta > 0:
                        break

            if best_delta <= gain_threshold:
                print("No improvement possible.")
                break

            df = df.drop(index=best_idx)
            removals += 1
            total_gain += best_delta
            progress_bar.set_postfix(gain=total_gain)
            if verbose:
                print(f"Removed index {best_idx}, Δloss = {best_delta:.4f}")

        return df