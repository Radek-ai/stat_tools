
"""
Utility functions for power analysis calculations
"""
import numpy as np
from statsmodels.stats.power import TTestIndPower

class PowerAnalysis:
    """Class to handle power analysis calculations"""
    
    def __init__(self, alternative="two-sided"):
        """
        Initialize PowerAnalysis
        
        Args:
            alternative: Type of alternative hypothesis
                - "two-sided": H1: μ1 ≠ μ2 (default)
                - "larger": H1: μ1 > μ2 (one-sided, expecting increase)
                - "smaller": H1: μ1 < μ2 (one-sided, expecting decrease)
        """
        self.analysis = TTestIndPower()
        self.alternative = alternative
    
    def calculate_sample_size(self, uplift, alpha, power, mean, std):
        """
        Calculate required sample size for a single group
        
        Args:
            uplift: Expected relative change
            alpha: Significance level
            power: Statistical power
            mean: Mean value
            std: Standard deviation
            
        Returns:
            Sample size
        """
        
        try:
            es = (mean * uplift) / std
            
            n = float(self.analysis.solve_power(
                effect_size=es, 
                alpha=alpha, 
                power=power, 
                alternative=self.alternative
            ))
            
            return n
        except Exception as e:
            return np.nan
    
    def calculate_sample_sizes(self, uplift, alpha, power, mean_pre, mean_exp, std_pre, std_exp):
        """
        Calculate required sample sizes for both groups (legacy method)
        
        Args:
            uplift: Expected relative change
            alpha: Significance level
            power: Statistical power
            mean_pre: Mean in first group
            mean_exp: Mean in second group
            std_pre: Standard deviation in first group
            std_exp: Standard deviation in second group
            
        Returns:
            Tuple of (n_pre, n_exp) sample sizes
        """
        n_pre = self.calculate_sample_size(uplift, alpha, power, mean_pre, std_pre)
        n_exp = self.calculate_sample_size(uplift, alpha, power, mean_exp, std_exp)
        return n_pre, n_exp
    
    def compute_all_sample_sizes(self, uplifts, alphas, powers, groups, progress_callback=None):
        """
        Compute sample sizes for all parameter combinations and all groups
        
        Args:
            uplifts: Array of uplift values
            alphas: Array of alpha values
            powers: Array of power values
            groups: Dictionary of groups with 'mean' and 'std' for each
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary mapping group names to lists of 2D arrays
        """
        Z_groups = {}
        
        for group_name, group_stats in groups.items():
            Z_group = []
            mean = group_stats['mean']
            std = group_stats['std']
            
            for idx, p in enumerate(powers):
                z = np.zeros((len(alphas), len(uplifts)))
                
                for i, a in enumerate(alphas):
                    for j, u in enumerate(uplifts):
                        n = self.calculate_sample_size(u, a, p, mean, std)
                        z[i, j] = n
                
                Z_group.append(z)
                
                if progress_callback:
                    total_groups = len(groups)
                    group_idx = list(groups.keys()).index(group_name)
                    progress = (group_idx * len(powers) + idx + 1) / (total_groups * len(powers))
                    progress_callback(progress)
            
            Z_groups[group_name] = Z_group
        
        return Z_groups
    
    def get_global_color_ranges(self, Z_pre, Z_exp):
        """
        Calculate global min/max for color scales
        
        Args:
            Z_pre: List of 2D arrays for first group
            Z_exp: List of 2D arrays for second group
            
        Returns:
            Dict with min/max values for each
        """
        return {
            'z_pre_min': min([np.nanmin(z) for z in Z_pre]),
            'z_pre_max': max([np.nanmax(z) for z in Z_pre]),
            'z_exp_min': min([np.nanmin(z) for z in Z_exp]),
            'z_exp_max': max([np.nanmax(z) for z in Z_exp])
        }
    
    def get_slice_color_ranges(self, Z_pre, Z_exp, slice_type='uplift'):
        """
        Calculate color ranges for sliced views
        
        Args:
            Z_pre: List of 2D arrays for first group
            Z_exp: List of 2D arrays for second group
            slice_type: 'uplift' or 'alpha'
            
        Returns:
            Dict with min/max values for slices
        """
        if slice_type == 'uplift':
            # For alpha-power view (varying uplift)
            num_slices = Z_pre[0].shape[1]  # number of uplifts
            z_pre_slices_min = min([
                np.nanmin(np.array([Z_pre[k][:, j] for k in range(len(Z_pre))]))
                for j in range(num_slices)
            ])
            z_pre_slices_max = max([
                np.nanmax(np.array([Z_pre[k][:, j] for k in range(len(Z_pre))]))
                for j in range(num_slices)
            ])
            z_exp_slices_min = min([
                np.nanmin(np.array([Z_exp[k][:, j] for k in range(len(Z_exp))]))
                for j in range(num_slices)
            ])
            z_exp_slices_max = max([
                np.nanmax(np.array([Z_exp[k][:, j] for k in range(len(Z_exp))]))
                for j in range(num_slices)
            ])
        else:  # alpha
            # For uplift-power view (varying alpha)
            num_slices = Z_pre[0].shape[0]  # number of alphas
            z_pre_slices_min = min([
                np.nanmin(np.array([Z_pre[k][i, :] for k in range(len(Z_pre))]))
                for i in range(num_slices)
            ])
            z_pre_slices_max = max([
                np.nanmax(np.array([Z_pre[k][i, :] for k in range(len(Z_pre))]))
                for i in range(num_slices)
            ])
            z_exp_slices_min = min([
                np.nanmin(np.array([Z_exp[k][i, :] for k in range(len(Z_exp))]))
                for i in range(num_slices)
            ])
            z_exp_slices_max = max([
                np.nanmax(np.array([Z_exp[k][i, :] for k in range(len(Z_exp))]))
                for i in range(num_slices)
            ])
        
        return {
            'z_pre_min': z_pre_slices_min,
            'z_pre_max': z_pre_slices_max,
            'z_exp_min': z_exp_slices_min,
            'z_exp_max': z_exp_slices_max
        }