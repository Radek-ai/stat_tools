"""
Streamlit validation utilities.

Common validation patterns to reduce duplication across pages.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st


def validate_uploaded_data(
    data_state_key: str,
    warning_message: Optional[str] = None,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Validate that data has been uploaded.
    
    Args:
        data_state_key: Session state key for the uploaded dataframe
        warning_message: Custom warning message (defaults to standard message)
    
    Returns:
        Tuple of (is_valid, dataframe or None)
    """
    if data_state_key not in st.session_state or st.session_state.get(data_state_key) is None:
        msg = warning_message or "⚠️ Please upload data first in the 'Data Upload' tab"
        st.warning(msg)
        return False, None
    
    return True, st.session_state.get(data_state_key).copy()


def validate_group_column(
    group_column_key: str,
    warning_message: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a group column has been selected.
    
    Args:
        group_column_key: Session state key for the selected group column
        warning_message: Custom warning message (defaults to standard message)
    
    Returns:
        Tuple of (is_valid, group_column or None)
    """
    if group_column_key not in st.session_state or not st.session_state.get(group_column_key):
        msg = warning_message or "⚠️ Please select a group column in the 'Configuration' tab"
        st.warning(msg)
        return False, None
    
    return True, st.session_state.get(group_column_key)


def validate_data_and_group(
    data_state_key: str,
    group_column_key: str,
    data_warning: Optional[str] = None,
    group_warning: Optional[str] = None,
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Validate both uploaded data and group column selection.
    
    Args:
        data_state_key: Session state key for the uploaded dataframe
        group_column_key: Session state key for the selected group column
        data_warning: Custom warning message for missing data
        group_warning: Custom warning message for missing group column
    
    Returns:
        Tuple of (is_valid, dataframe or None, group_column or None)
    """
    is_valid, df = validate_uploaded_data(data_state_key, data_warning)
    if not is_valid:
        return False, None, None
    
    is_valid, group_column = validate_group_column(group_column_key, group_warning)
    if not is_valid:
        return False, df, None
    
    return True, df, group_column
