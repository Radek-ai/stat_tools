"""
Plotly visualization functions for power analysis

This module provides functions for creating interactive plots for statistical power analysis.
The module is organized into:
- utils: Common utilities and constants
- single_plots: Line plots for single parameter analysis
- contour_plots: Contour maps with indicators
"""

# Import all public functions for backward compatibility
from power_analysis.plots.single_plots import (
    create_single_plot_uplift,
    create_single_plot_alpha,
    create_single_plot_power
)

from power_analysis.plots.contour_plots import (
    create_contour_plot_power,
    create_contour_plot_uplift,
    create_contour_plot_alpha
)

__all__ = [
    'create_single_plot_uplift',
    'create_single_plot_alpha',
    'create_single_plot_power',
    'create_contour_plot_power',
    'create_contour_plot_uplift',
    'create_contour_plot_alpha',
]

