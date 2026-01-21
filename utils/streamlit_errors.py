"""
Streamlit error handling utilities.

Common error handling patterns to reduce duplication across pages.
"""

from __future__ import annotations

import traceback
from typing import Optional

import streamlit as st


def handle_error(
    error: Exception,
    user_message: str,
    show_traceback: bool = True,
    traceback_title: str = "Error Details",
) -> None:
    """
    Display an error message with optional traceback in an expander.
    
    Args:
        error: The exception that occurred
        user_message: User-friendly error message to display
        show_traceback: Whether to show full traceback in expander
        traceback_title: Title for the traceback expander
    """
    st.error(f"❌ {user_message}: {str(error)}")
    if show_traceback:
        with st.expander(traceback_title):
            st.code(traceback.format_exc())


def handle_plot_error(
    plot_name: str,
    error: Exception,
    show_traceback: bool = True,
) -> None:
    """
    Display a warning for plot generation errors.
    
    Args:
        plot_name: Name/description of the plot that failed
        error: The exception that occurred
        show_traceback: Whether to show full traceback
    """
    st.warning(f"Could not generate {plot_name}: {str(error)}")
    if show_traceback:
        st.error(f"Traceback: {traceback.format_exc()}")
