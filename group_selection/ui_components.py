"""
UI components for group selection page.
This module re-exports functions from the split modules for backward compatibility.
"""
from group_selection.balancing import render_group_balancing
from group_selection.configuration import render_configuration
from group_selection.upload import render_data_upload

__all__ = [
    'render_data_upload',
    'render_configuration',
    'render_group_balancing',
]
