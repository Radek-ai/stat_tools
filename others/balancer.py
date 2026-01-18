
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
import itertools
import random
from typing import List, Dict, Union, Tuple

class MultiGroupBalancer:
    def __init__(
        self,
        group_column: str,
        value_columns: List[str],
        strat_columns: List[str],
    ):
        self.group_column = group_column
        self.value_columns = value_columns
        self.strat_columns = strat_columns
        # Default objective
        self.target_metrics = {}
        # Cache for pairwise losses: {(g1, g2): loss_value}
        self.pairwise_loss_cache = {}

    def set_objective(self, objective: Dict[str, Union[float, Dict[str, float]]]):
        self.target_metrics = objective

    def _get_pair_key(self, g1: str, g2: str) -> Tuple[str, str]:
        """Ensure consistent sorting for pair keys."""
        return tuple(sorted((str(g1), str(g2))))

    # ------------------------------------------------------------------
    # Optimized Pairwise Loss (Index-based)
    # ------------------------------------------------------------------
    def _compute_single_pair_loss(
        self, df: pd.DataFrame, idx1: pd.Index, idx2: pd.Index, g1_name: str, g2_name: str
    ) -> float:
        """
        Compute loss for a single pair of groups using their row indices.
        Avoids full df filtering and global crosstabs.
        """
        if len(idx1) <= 1 or len(idx2) <= 1:
            return 0.0

        loss = 0.0
        
        # --- Numeric columns (Vectorized t-test) ---
        # Note: Extracting values via loc is faster than repeated boolean indexing
        for col in self.value_columns:
            p_goal = self.target_metrics.get("numeric_p_value", {}).get(col)
            if p_goal is None:
                continue
            x1 = df.loc[idx1, col].dropna().values
            x2 = df.loc[idx2, col].dropna().values
            
            if len(x1) > 1 and len(x2) > 1:
                # Welch's t-test
                _, p = ttest_ind(x1, x2, equal_var=False)
                
            
                if p_goal < 0:
                    loss += max(0, (p - abs(p_goal)) / abs(p_goal))
                else:
                    loss += max(0, (p_goal - p) / p_goal)

        # --- Categorical columns (Localized Value Counts) ---
        for col in self.strat_columns:
            c_goal = self.target_metrics.get("categorical_total_imbalance", {}).get(col)
            if c_goal is None:
                continue
            # Normalize=True gives proportions directly
            vc1 = df.loc[idx1, col].value_counts(normalize=True)
            vc2 = df.loc[idx2, col].value_counts(normalize=True)
            
            # Align and calculate difference (handles missing categories via fill_value=0)
            diff = (vc1.subtract(vc2, fill_value=0)).abs().sum() * 100
            
            
            loss += max(0, (diff - c_goal) / c_goal)

        return loss

    def initialize_loss_cache(self, df: pd.DataFrame, group_indices: Dict[str, pd.Index]) -> float:
        """
        Compute all pairwise losses from scratch and populate the cache.
        Returns total loss.
        """
        self.pairwise_loss_cache = {}
        groups = list(group_indices.keys())
        total_loss = 0.0

        for g1, g2 in itertools.combinations(groups, 2):
            key = self._get_pair_key(g1, g2)
            loss = self._compute_single_pair_loss(df, group_indices[g1], group_indices[g2], g1, g2)
            self.pairwise_loss_cache[key] = loss
            total_loss += loss
            
        return total_loss

    def compute_total_loss(self, df: pd.DataFrame) -> float:
        """
        Legacy/Convenience wrapper. 
        If you have group_indices managed, use initialize_loss_cache instead for speed.
        """
        groups = df[self.group_column].unique()
        # Build indices temporarily
        group_indices = {g: df.index[df[self.group_column] == g] for g in groups}
        return self.initialize_loss_cache(df, group_indices)

    # ------------------------------------------------------------------
    # Candidate Selection
    # ------------------------------------------------------------------
    def preselect_candidates(
        self, df: pd.DataFrame, source_indices: pd.Index, top_k: int
    ) -> pd.Index:
        """Find rows that are outliers in their current group."""
        if len(source_indices) == 0 or top_k <= 0:
            return pd.Index([])

        scores = pd.Series(0.0, index=source_indices)
        for col in self.value_columns:
            vals = df.loc[source_indices, col]
            std = vals.std()
            if std > 0:
                scores += ((vals - vals.mean()).abs()) / std
        
        return scores.nlargest(min(top_k, len(scores))).index

    # ------------------------------------------------------------------
    # Fast Move Estimation
    # ------------------------------------------------------------------
    def _estimate_move_gain_fast(
        self,
        df: pd.DataFrame,
        row_idx: int,
        source_group: str,
        target_group: str,
        group_indices: Dict[str, pd.Index],
        current_total_loss: float,
        groups_list: List[str]
    ) -> float:
        """
        Estimate gain of moving row_idx from Source -> Target using the cache.
        Does NOT copy the dataframe.
        """
        # 1. Simulate new indices
        # dropping/appending on Index is relatively cheap compared to DF copy
        new_idx_source = group_indices[source_group].drop(row_idx)
        new_idx_target = group_indices[target_group].append(pd.Index([row_idx]))

        # 2. Identify affected pairs and calculate delta
        # We only care about pairs involving Source or Target.
        # gain = (Old Loss of affected pairs) - (New Loss of affected pairs)
        
        old_affected_loss = 0.0
        new_affected_loss = 0.0

        # A. Source-Target Pair
        st_key = self._get_pair_key(source_group, target_group)
        old_affected_loss += self.pairwise_loss_cache.get(st_key, 0.0)
        new_affected_loss += self._compute_single_pair_loss(
            df, new_idx_source, new_idx_target, source_group, target_group
        )

        # B. Pairs involving Source (excluding Target)
        for other_g in groups_list:
            if other_g == source_group or other_g == target_group:
                continue
            
            # Pair (Source, Other)
            key = self._get_pair_key(source_group, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            
            # Recalculate with new source indices
            new_loss = self._compute_single_pair_loss(
                df, new_idx_source, group_indices[other_g], source_group, other_g
            )
            new_affected_loss += new_loss

        # C. Pairs involving Target (excluding Source)
        for other_g in groups_list:
            if other_g == source_group or other_g == target_group:
                continue
            
            # Pair (Target, Other)
            key = self._get_pair_key(target_group, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            
            # Recalculate with new target indices
            new_loss = self._compute_single_pair_loss(
                df, new_idx_target, group_indices[other_g], target_group, other_g
            )
            new_affected_loss += new_loss

        # Gain = Reduction in loss
        return old_affected_loss - new_affected_loss

    def _apply_move_and_update_cache(
        self,
        df: pd.DataFrame,
        best_idx: int,
        source_group: str,
        target_group: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Actually move the row in the DF, update indices, and update the cache.
        Returns the new total loss.
        """
        # 1. Update Dataframe
        df.loc[best_idx, self.group_column] = target_group
        
        # 2. Update Indices Map
        group_indices[source_group] = group_indices[source_group].drop(best_idx)
        group_indices[target_group] = group_indices[target_group].append(pd.Index([best_idx]))
        
        # 3. Recompute and Update Cache for Affected Pairs
        # (Same logic as estimate, but actually writing to cache)
        
        # Source-Target
        st_key = self._get_pair_key(source_group, target_group)
        self.pairwise_loss_cache[st_key] = self._compute_single_pair_loss(
            df, group_indices[source_group], group_indices[target_group], source_group, target_group
        )
        
        # Others
        for other_g in groups_list:
            if other_g == source_group or other_g == target_group:
                continue
                
            # Update (Source, Other)
            key_s = self._get_pair_key(source_group, other_g)
            self.pairwise_loss_cache[key_s] = self._compute_single_pair_loss(
                df, group_indices[source_group], group_indices[other_g], source_group, other_g
            )
            
            # Update (Target, Other)
            key_t = self._get_pair_key(target_group, other_g)
            self.pairwise_loss_cache[key_t] = self._compute_single_pair_loss(
                df, group_indices[target_group], group_indices[other_g], target_group, other_g
            )
            
        # 4. Return new total sum
        return sum(self.pairwise_loss_cache.values())

    # ------------------------------------------------------------------
    # Optimized Sequential
    # ------------------------------------------------------------------
    def balance_sequential(
        self,
        df: pd.DataFrame,
        max_iterations: int = 50,
        top_k_candidates: int = 20,
        k_random_candidates: int = 20,
        gain_threshold: float = 0.001, # slightly higher default to stop micro-moves
        early_break: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        history = []

        groups = df[self.group_column].unique().tolist()
        # Create map of Group -> Indices for fast access
        group_indices = {g: df.index[df[self.group_column] == g] for g in groups}
        
        # Initial Loss Calculation & Caching
        current_loss = self.initialize_loss_cache(df, group_indices)
        initial_loss = current_loss
        history.append(current_loss)

        ordered_pairs = [(g1, g2) for g1 in groups for g2 in groups if g1 != g2]
        
        progress = tqdm(range(max_iterations), desc="Sequential Multi-Pair", unit="iter")

        for it in progress:
            iteration_start_loss = current_loss
            
            # Shuffle pairs to avoid bias
            random.shuffle(ordered_pairs)

            for source_group, target_group in ordered_pairs:
                if len(group_indices[source_group]) == 0:
                    continue

                # --- Candidate Selection ---
                # Use indices map directly
                z_candidates = self.preselect_candidates(df, group_indices[source_group], top_k_candidates)
                
                # Sample random candidates efficiently
                idx_pool = group_indices[source_group]
                if len(idx_pool) > k_random_candidates:
                    rand_candidates = pd.Index(np.random.choice(idx_pool, k_random_candidates, replace=False))
                else:
                    rand_candidates = idx_pool
                
                candidate_idxs = z_candidates.union(rand_candidates)

                best_gain = -np.inf
                best_idx = None

                # --- Fast Estimation Loop ---
                for idx in candidate_idxs:
                    gain = self._estimate_move_gain_fast(
                        df, idx, source_group, target_group, 
                        group_indices, current_loss, groups
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if early_break and best_gain > gain_threshold:
                            break

                # --- Apply Move if good enough ---
                if best_idx is not None and best_gain > gain_threshold:
                    current_loss = self._apply_move_and_update_cache(
                        df, best_idx, source_group, target_group, group_indices, groups
                    )
                    if verbose:
                        print(f"  Move {best_idx}: {source_group}->{target_group}, gain={best_gain:.4f}")

            # End of iteration logging
            history.append(current_loss)
            iteration_realized_gain = iteration_start_loss - current_loss

            if verbose:
                print(f"Iter {it}: Loss {iteration_start_loss:.5f} -> {current_loss:.5f}, Gain {iteration_realized_gain:.5f}")

            if iteration_realized_gain <= gain_threshold:
                if verbose:
                    print("Converged.")
                break

            progress.set_postfix(loss=f"{current_loss:.4f}", gain=f"{iteration_realized_gain:.4f}")

        self.loss_history = history
        return df

    # ------------------------------------------------------------------
    # Initial Groups (Kept from previous verified logic)
    # ------------------------------------------------------------------
    def create_initial_groups(
        self,
        df: pd.DataFrame,
        group_names: List[str],
        group_proportions: List[float],
        value_columns_overrides: List[str] | None = None,
        n_bins: int = 4,
        random_state: int = None,
    ) -> pd.DataFrame:
        """
        Create initial group assignments using stratified splitting on
        categorical + bucketed numeric features. Fewer bins to avoid tiny strata.
        """
        if value_columns_overrides is None:
            value_columns_overrides = self.value_columns

        df = df.copy()
        rng = np.random.default_rng(random_state)

        # --- Bucket numeric columns (use fewer bins!) ---
        bucket_cols = []
        for col in value_columns_overrides:
            binned_col = f"{col}_bin"
            df[binned_col] = pd.qcut(
                df[col],
                n_bins,
                labels=False,
                duplicates="drop",
            )
            bucket_cols.append(binned_col)

        # --- Combine stratification columns ---
        strat_cols = self.strat_columns + bucket_cols
        df["_strata"] = df[strat_cols].astype(str).agg("_".join, axis=1)

        df["_group"] = None

        for stratum, group_rows in df.groupby("_strata"):
            n_rows = len(group_rows)
            n_groups = len(group_names)

            if n_rows < n_groups:
                # Too few rows: assign randomly
                for idx in group_rows.index:
                    df.loc[idx, "_group"] = rng.choice(group_names)
                continue

            # Target counts per group
            target_counts = [int(np.round(p * n_rows)) for p in group_proportions]

            # Adjust rounding
            diff = n_rows - sum(target_counts)
            if diff > 0:
                # add extras
                for i in range(diff):
                    target_counts[i % n_groups] += 1
            elif diff < 0:
                # remove extras (subtract from largest)
                for i in range(-diff):
                    max_idx = np.argmax(target_counts)
                    target_counts[max_idx] -= 1
            # for i in range(diff):
            #     target_counts[i % n_groups] += 1

            rows = group_rows.index.to_list()
            rng.shuffle(rows)
            pairs = list(zip(group_names, target_counts))
            rng.shuffle(pairs)
            start = 0
            for group_name, count in pairs:
                end = start + count
                df.loc[rows[start:end], "_group"] = group_name
                start = end

        df[self.group_column] = df["_group"]
        df.drop(columns=["_strata", "_group"] + bucket_cols, inplace=True)

        return df

    def _estimate_swap_gain_fast(
        self,
        df: pd.DataFrame,
        idx1: int,
        idx2: int,
        g1: str,
        g2: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Estimate swap gain: idx1(g1) <-> idx2(g2) using cache deltas.
        """
        # Simulate new indices
        new_idx_g1 = group_indices[g1].drop(idx1).append(pd.Index([idx2]))
        new_idx_g2 = group_indices[g2].drop(idx2).append(pd.Index([idx1]))
        
        old_affected_loss = 0.0
        new_affected_loss = 0.0
        
        # G1-G2 pair
        key = self._get_pair_key(g1, g2)
        old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
        new_affected_loss += self._compute_single_pair_loss(df, new_idx_g1, new_idx_g2, g1, g2)
        
        # G1-other pairs
        for other_g in groups_list:
            if other_g == g1 or other_g == g2: continue
            key = self._get_pair_key(g1, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            new_affected_loss += self._compute_single_pair_loss(df, new_idx_g1, group_indices[other_g], g1, other_g)
        
        # G2-other pairs  
        for other_g in groups_list:
            if other_g == g1 or other_g == g2: continue
            key = self._get_pair_key(g2, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            new_affected_loss += self._compute_single_pair_loss(df, new_idx_g2, group_indices[other_g], g2, other_g)
        
        return old_affected_loss - new_affected_loss

    def _apply_swap_and_update_cache(
        self,
        df: pd.DataFrame,
        idx1: int,
        idx2: int,
        g1: str,
        g2: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Execute swap, update indices/cache.
        """
        # Swap groups in DF
        df.loc[idx1, self.group_column], df.loc[idx2, self.group_column] = g2, g1
        
        # Update indices
        group_indices[g1] = group_indices[g1].drop(idx1).append(pd.Index([idx2]))
        group_indices[g2] = group_indices[g2].drop(idx2).append(pd.Index([idx1]))
        
        # Update cache (same pairs as estimate)
        key = self._get_pair_key(g1, g2)
        self.pairwise_loss_cache[key] = self._compute_single_pair_loss(df, group_indices[g1], group_indices[g2], g1, g2)
        
        for other_g in groups_list:
            if other_g == g1 or other_g == g2: continue
            key_g1 = self._get_pair_key(g1, other_g)
            self.pairwise_loss_cache[key_g1] = self._compute_single_pair_loss(df, group_indices[g1], group_indices[other_g], g1, other_g)
            key_g2 = self._get_pair_key(g2, other_g)
            self.pairwise_loss_cache[key_g2] = self._compute_single_pair_loss(df, group_indices[g2], group_indices[other_g], g2, other_g)
        
        return sum(self.pairwise_loss_cache.values())

    def balance_swap(
        self,
        df: pd.DataFrame,
        max_iterations: int = 50,
        top_k_candidates: int = 10,  # Smaller default due to O(k^2)
        k_random_candidates: int = 10,
        gain_threshold: float = 0.001,
        early_break: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Swap-optimized version using the same cache infrastructure.
        """
        import random  # For shuffling
        
        df = df.copy()
        history = []
        
        groups = df[self.group_column].unique().tolist()
        group_indices = {g: df.index[df[self.group_column] == g] for g in groups}
        
        current_loss = self.initialize_loss_cache(df, group_indices)
        history.append(current_loss)
        
        ordered_pairs = [(g1, g2) for g1 in groups for g2 in groups if g1 != g2]
        
        progress = tqdm(range(max_iterations), desc="Swap Multi-Pair", unit="iter")
        
        for it in progress:
            iteration_start_loss = current_loss
            
            # Precompute candidates per group (shared across pairs)
            group_candidates = {}
            for g in groups:
                if len(group_indices[g]) == 0:
                    group_candidates[g] = pd.Index([])
                    continue
                z_cands = self.preselect_candidates(df, group_indices[g], top_k_candidates)
                rand_cands = pd.Index(np.random.choice(
                    group_indices[g], min(k_random_candidates, len(group_indices[g])), replace=False
                ))
                group_candidates[g] = z_cands.union(rand_cands)
            
            random.shuffle(ordered_pairs)
            
            for g1, g2 in ordered_pairs:
                if len(group_candidates[g1]) == 0 or len(group_candidates[g2]) == 0:
                    continue
                    
                best_gain = -np.inf
                best_pair = None
                
                # Nested candidate loop (now much faster due to cache)
                for idx1 in list(group_candidates[g1])[:20]:  # Cap for speed
                    for idx2 in list(group_candidates[g2])[:20]:
                        gain = self._estimate_swap_gain_fast(df, idx1, idx2, g1, g2, group_indices, groups)
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (idx1, idx2)
                            if early_break and best_gain > gain_threshold:
                                break
                    if early_break and best_gain > gain_threshold:
                        break
                
                if best_pair is not None and best_gain > gain_threshold:
                    idx1, idx2 = best_pair
                    current_loss = self._apply_swap_and_update_cache(df, idx1, idx2, g1, g2, group_indices, groups)
                    if verbose:
                        print(f"Swap {idx1}<->{idx2} ({g1}<->{g2}), gain={best_gain:.4f}")
            
            history.append(current_loss)
            iteration_gain = iteration_start_loss - current_loss
            
            if verbose:
                print(f"Iter {it}: {iteration_start_loss:.5f} -> {current_loss:.5f}, gain={iteration_gain:.5f}")
            
            if iteration_gain <= gain_threshold:
                break
                
            progress.set_postfix(loss=f"{current_loss:.4f}", gain=f"{iteration_gain:.4f}")
        
        self.loss_history = history
        return df

    # ------------------------------------------------------------------
    # Batch Move Operations (Less Overfitting)
    # ------------------------------------------------------------------
    def _estimate_batch_move_gain_fast(
        self,
        df: pd.DataFrame,
        batch_indices: pd.Index,
        source_group: str,
        target_group: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Estimate gain of moving a batch of rows from Source -> Target using the cache.
        """
        if len(batch_indices) == 0:
            return 0.0
        
        # Simulate new indices after moving the batch
        new_idx_source = group_indices[source_group].difference(batch_indices)
        new_idx_target = group_indices[target_group].union(batch_indices)
        
        old_affected_loss = 0.0
        new_affected_loss = 0.0
        
        # A. Source-Target Pair
        st_key = self._get_pair_key(source_group, target_group)
        old_affected_loss += self.pairwise_loss_cache.get(st_key, 0.0)
        new_affected_loss += self._compute_single_pair_loss(
            df, new_idx_source, new_idx_target, source_group, target_group
        )
        
        # B. Pairs involving Source (excluding Target)
        for other_g in groups_list:
            if other_g == source_group or other_g == target_group:
                continue
            
            key = self._get_pair_key(source_group, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            new_loss = self._compute_single_pair_loss(
                df, new_idx_source, group_indices[other_g], source_group, other_g
            )
            new_affected_loss += new_loss
        
        # C. Pairs involving Target (excluding Source)
        for other_g in groups_list:
            if other_g == source_group or other_g == target_group:
                continue
            
            key = self._get_pair_key(target_group, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            new_loss = self._compute_single_pair_loss(
                df, new_idx_target, group_indices[other_g], target_group, other_g
            )
            new_affected_loss += new_loss
        
        return old_affected_loss - new_affected_loss

    def _apply_batch_move_and_update_cache(
        self,
        df: pd.DataFrame,
        batch_indices: pd.Index,
        source_group: str,
        target_group: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Actually move the batch of rows in the DF, update indices, and update the cache.
        Returns the new total loss.
        """
        # 1. Update Dataframe
        df.loc[batch_indices, self.group_column] = target_group
        
        # 2. Update Indices Map
        group_indices[source_group] = group_indices[source_group].difference(batch_indices)
        group_indices[target_group] = group_indices[target_group].union(batch_indices)
        
        # 3. Recompute and Update Cache for Affected Pairs
        st_key = self._get_pair_key(source_group, target_group)
        self.pairwise_loss_cache[st_key] = self._compute_single_pair_loss(
            df, group_indices[source_group], group_indices[target_group], source_group, target_group
        )
        
        for other_g in groups_list:
            if other_g == source_group or other_g == target_group:
                continue
            
            key_s = self._get_pair_key(source_group, other_g)
            self.pairwise_loss_cache[key_s] = self._compute_single_pair_loss(
                df, group_indices[source_group], group_indices[other_g], source_group, other_g
            )
            
            key_t = self._get_pair_key(target_group, other_g)
            self.pairwise_loss_cache[key_t] = self._compute_single_pair_loss(
                df, group_indices[target_group], group_indices[other_g], target_group, other_g
            )
        
        return sum(self.pairwise_loss_cache.values())

    def balance_sequential_batch(
        self,
        df: pd.DataFrame,
        max_iterations: int = 50,
        subset_size: int = 5,
        n_samples: int = 10,
        gain_threshold: float = 0.001,
        early_break: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Balance using batch moves (groups of rows at a time) to reduce overfitting.
        
        Parameters:
        -----------
        subset_size : int
            Number of rows to move in each batch
        n_samples : int
            Number of random samples to try for each batch
        """
        df = df.copy()
        history = []
        
        groups = df[self.group_column].unique().tolist()
        group_indices = {g: df.index[df[self.group_column] == g] for g in groups}
        
        current_loss = self.initialize_loss_cache(df, group_indices)
        history.append(current_loss)
        
        ordered_pairs = [(g1, g2) for g1 in groups for g2 in groups if g1 != g2]
        
        progress = tqdm(range(max_iterations), desc="Batch Sequential", unit="iter")
        
        for it in progress:
            iteration_start_loss = current_loss
            
            random.shuffle(ordered_pairs)
            
            for source_group, target_group in ordered_pairs:
                source_size = len(group_indices[source_group])
                if source_size < subset_size:
                    continue
                
                # Try multiple random samples of subset_size rows
                best_gain = -np.inf
                best_batch = None
                
                # Limit n_samples to avoid too many combinations
                actual_n_samples = min(n_samples, 
                                     max(1, source_size // subset_size))
                
                for _ in range(actual_n_samples):
                    # Randomly sample subset_size rows from source group
                    if source_size > subset_size:
                        batch_indices = pd.Index(
                            np.random.choice(
                                group_indices[source_group], 
                                subset_size, 
                                replace=False
                            )
                        )
                    else:
                        batch_indices = group_indices[source_group]
                    
                    gain = self._estimate_batch_move_gain_fast(
                        df, batch_indices, source_group, target_group,
                        group_indices, groups
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_batch = batch_indices
                        if early_break and best_gain > gain_threshold:
                            break
                
                # Apply batch move if good enough
                if best_batch is not None and best_gain > gain_threshold:
                    current_loss = self._apply_batch_move_and_update_cache(
                        df, best_batch, source_group, target_group, group_indices, groups
                    )
                    if verbose:
                        print(f"  Batch move {len(best_batch)} rows: {source_group}->{target_group}, gain={best_gain:.4f}")
            
            history.append(current_loss)
            iteration_realized_gain = iteration_start_loss - current_loss
            
            if verbose:
                print(f"Iter {it}: Loss {iteration_start_loss:.5f} -> {current_loss:.5f}, Gain {iteration_realized_gain:.5f}")
            
            if iteration_realized_gain <= gain_threshold:
                if verbose:
                    print("Converged.")
                break
            
            progress.set_postfix(loss=f"{current_loss:.4f}", gain=f"{iteration_realized_gain:.4f}")
        
        self.loss_history = history
        return df

    # ------------------------------------------------------------------
    # Batch Swap Operations (Less Overfitting)
    # ------------------------------------------------------------------
    def _estimate_batch_swap_gain_fast(
        self,
        df: pd.DataFrame,
        batch1_indices: pd.Index,
        batch2_indices: pd.Index,
        g1: str,
        g2: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Estimate swap gain: batch1(g1) <-> batch2(g2) using cache deltas.
        """
        if len(batch1_indices) == 0 or len(batch2_indices) == 0:
            return 0.0
        
        # Simulate new indices after swapping batches
        new_idx_g1 = group_indices[g1].difference(batch1_indices).union(batch2_indices)
        new_idx_g2 = group_indices[g2].difference(batch2_indices).union(batch1_indices)
        
        old_affected_loss = 0.0
        new_affected_loss = 0.0
        
        # G1-G2 pair
        key = self._get_pair_key(g1, g2)
        old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
        new_affected_loss += self._compute_single_pair_loss(df, new_idx_g1, new_idx_g2, g1, g2)
        
        # G1-other pairs
        for other_g in groups_list:
            if other_g == g1 or other_g == g2:
                continue
            key = self._get_pair_key(g1, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            new_affected_loss += self._compute_single_pair_loss(df, new_idx_g1, group_indices[other_g], g1, other_g)
        
        # G2-other pairs
        for other_g in groups_list:
            if other_g == g1 or other_g == g2:
                continue
            key = self._get_pair_key(g2, other_g)
            old_affected_loss += self.pairwise_loss_cache.get(key, 0.0)
            new_affected_loss += self._compute_single_pair_loss(df, new_idx_g2, group_indices[other_g], g2, other_g)
        
        return old_affected_loss - new_affected_loss

    def _apply_batch_swap_and_update_cache(
        self,
        df: pd.DataFrame,
        batch1_indices: pd.Index,
        batch2_indices: pd.Index,
        g1: str,
        g2: str,
        group_indices: Dict[str, pd.Index],
        groups_list: List[str]
    ) -> float:
        """
        Execute batch swap, update indices/cache.
        """
        # Swap groups in DF
        df.loc[batch1_indices, self.group_column] = g2
        df.loc[batch2_indices, self.group_column] = g1
        
        # Update indices
        group_indices[g1] = group_indices[g1].difference(batch1_indices).union(batch2_indices)
        group_indices[g2] = group_indices[g2].difference(batch2_indices).union(batch1_indices)
        
        # Update cache
        key = self._get_pair_key(g1, g2)
        self.pairwise_loss_cache[key] = self._compute_single_pair_loss(df, group_indices[g1], group_indices[g2], g1, g2)
        
        for other_g in groups_list:
            if other_g == g1 or other_g == g2:
                continue
            key_g1 = self._get_pair_key(g1, other_g)
            self.pairwise_loss_cache[key_g1] = self._compute_single_pair_loss(df, group_indices[g1], group_indices[other_g], g1, other_g)
            key_g2 = self._get_pair_key(g2, other_g)
            self.pairwise_loss_cache[key_g2] = self._compute_single_pair_loss(df, group_indices[g2], group_indices[other_g], g2, other_g)
        
        return sum(self.pairwise_loss_cache.values())

    def balance_swap_batch(
        self,
        df: pd.DataFrame,
        max_iterations: int = 50,
        subset_size: int = 5,
        n_samples: int = 10,
        gain_threshold: float = 0.001,
        early_break: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Balance using batch swaps (groups of rows at a time) to reduce overfitting.
        
        Parameters:
        -----------
        subset_size : int
            Number of rows to swap in each batch (same size for both groups)
        n_samples : int
            Number of random samples to try for each batch pair
        """
        df = df.copy()
        history = []
        
        groups = df[self.group_column].unique().tolist()
        group_indices = {g: df.index[df[self.group_column] == g] for g in groups}
        
        current_loss = self.initialize_loss_cache(df, group_indices)
        history.append(current_loss)
        
        ordered_pairs = [(g1, g2) for g1 in groups for g2 in groups if g1 != g2]
        
        progress = tqdm(range(max_iterations), desc="Batch Swap", unit="iter")
        
        for it in progress:
            iteration_start_loss = current_loss
            
            random.shuffle(ordered_pairs)
            
            for g1, g2 in ordered_pairs:
                size1 = len(group_indices[g1])
                size2 = len(group_indices[g2])
                
                if size1 < subset_size or size2 < subset_size:
                    continue
                
                # Try multiple random samples
                best_gain = -np.inf
                best_batch_pair = None
                
                # Limit n_samples
                actual_n_samples = min(n_samples, 
                                     max(1, min(size1, size2) // subset_size))
                
                for _ in range(actual_n_samples):
                    # Randomly sample batches from both groups
                    if size1 > subset_size:
                        batch1 = pd.Index(
                            np.random.choice(group_indices[g1], subset_size, replace=False)
                        )
                    else:
                        batch1 = group_indices[g1]
                    
                    if size2 > subset_size:
                        batch2 = pd.Index(
                            np.random.choice(group_indices[g2], subset_size, replace=False)
                        )
                    else:
                        batch2 = group_indices[g2]
                    
                    gain = self._estimate_batch_swap_gain_fast(
                        df, batch1, batch2, g1, g2, group_indices, groups
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_batch_pair = (batch1, batch2)
                        if early_break and best_gain > gain_threshold:
                            break
                
                if best_batch_pair is not None and best_gain > gain_threshold:
                    batch1, batch2 = best_batch_pair
                    current_loss = self._apply_batch_swap_and_update_cache(
                        df, batch1, batch2, g1, g2, group_indices, groups
                    )
                    if verbose:
                        print(f"Batch swap {len(batch1)}<->{len(batch2)} rows ({g1}<->{g2}), gain={best_gain:.4f}")
            
            history.append(current_loss)
            iteration_gain = iteration_start_loss - current_loss
            
            if verbose:
                print(f"Iter {it}: {iteration_start_loss:.5f} -> {current_loss:.5f}, gain={iteration_gain:.5f}")
            
            if iteration_gain <= gain_threshold:
                break
            
            progress.set_postfix(loss=f"{current_loss:.4f}", gain=f"{iteration_gain:.4f}")
        
        self.loss_history = history
        return df