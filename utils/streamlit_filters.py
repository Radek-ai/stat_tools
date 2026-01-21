"""
Shared Streamlit filtering UI.

This centralizes the repeated filtering patterns used across pages:
- outlier filtering (Percentile / IQR)
- numeric range filtering
- categorical include/exclude filtering

The functions are intentionally UI-only and depend on `utils.data_filtering` for
pure filtering logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_filtering import (
    filter_by_categorical_values,
    filter_by_value_range,
    filter_outliers_iqr,
    filter_outliers_percentile,
    is_id_column,
)
from utils.artifact_builder import ArtifactBuilder


def _k(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name


def render_filtering_tabs(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    exclude_columns: Optional[Sequence[str]] = None,
    exclude_id_columns: bool = False,
    outlier_methods: Sequence[str] = ("None", "Percentile", "IQR"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render outlier + value-based filtering tabs.

    Returns:
        (filtered_df, filter_config)
    """
    exclude_columns_set = set(exclude_columns or [])

    filtered_df = df.copy()
    filter_config: Dict[str, Any] = {}

    tab_outliers, tab_values = st.tabs(["Outlier Filtering", "Value-Based Filtering"])

    # -----------------------------
    # Outlier filtering
    # -----------------------------
    with tab_outliers:
        st.subheader("📉 Outlier Filtering")
        st.markdown("Remove outliers from numeric columns")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_columns_set]
        if exclude_id_columns:
            numeric_cols = [c for c in numeric_cols if not is_id_column(df, c)]

        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for outlier filtering")
        else:
            outlier_column = st.selectbox(
                "Select Column for Outlier Filtering",
                options=numeric_cols,
                key=_k(key_prefix, "outlier_column"),
            )

            outlier_method = st.selectbox(
                "Outlier Method",
                options=list(outlier_methods),
                index=0,
                key=_k(key_prefix, "outlier_method"),
                help="None: No filtering | Percentile: Remove rows outside percentile range | IQR: Remove rows outside IQR range",
            )

            if outlier_method == "Percentile":
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    p_low = st.number_input(
                        "Lower Percentile",
                        min_value=0.0,
                        max_value=100.0,
                        value=1.0,
                        step=0.1,
                        key=_k(key_prefix, "p_low"),
                    )
                with col_p2:
                    p_high = st.number_input(
                        "Upper Percentile",
                        min_value=0.0,
                        max_value=100.0,
                        value=99.0,
                        step=0.1,
                        key=_k(key_prefix, "p_high"),
                    )

                if p_low >= p_high:
                    st.error("❌ Lower percentile must be less than upper percentile")
                else:
                    lower_bound = df[outlier_column].quantile(p_low / 100.0)
                    upper_bound = df[outlier_column].quantile(p_high / 100.0)
                    rows_before = len(filtered_df)
                    metric_total_before = filtered_df[outlier_column].sum()
                    filtered_df = filter_outliers_percentile(filtered_df, outlier_column, p_low, p_high)
                    rows_after = len(filtered_df)
                    removed = rows_before - rows_after
                    metric_total_after = filtered_df[outlier_column].sum()
                    metric_removed_pct = (
                        ((metric_total_before - metric_total_after) / metric_total_before * 100)
                        if metric_total_before > 0
                        else 0
                    )
                    st.info(
                        f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%) | "
                        f"{metric_removed_pct:.1f}% of total {outlier_column} outside range "
                        f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                    )

                    filter_config["outlier_filtering"] = {
                        "method": "Percentile",
                        "column": outlier_column,
                        "p_low": p_low,
                        "p_high": p_high,
                    }

            elif outlier_method == "IQR":
                iqr_multiplier = st.number_input(
                    "IQR Multiplier",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    key=_k(key_prefix, "iqr_multiplier"),
                )

                Q1 = df[outlier_column].quantile(0.25)
                Q3 = df[outlier_column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR

                rows_before = len(filtered_df)
                metric_total_before = filtered_df[outlier_column].sum()
                filtered_df = filter_outliers_iqr(filtered_df, outlier_column, iqr_multiplier)
                rows_after = len(filtered_df)
                removed = rows_before - rows_after
                metric_total_after = filtered_df[outlier_column].sum()
                metric_removed_pct = (
                    ((metric_total_before - metric_total_after) / metric_total_before * 100)
                    if metric_total_before > 0
                    else 0
                )

                st.info(
                    f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%) | "
                    f"{metric_removed_pct:.1f}% of total {outlier_column} outside range "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                )

                filter_config["outlier_filtering"] = {
                    "method": "IQR",
                    "column": outlier_column,
                    "iqr_multiplier": iqr_multiplier,
                }

    # -----------------------------
    # Value-based filtering
    # -----------------------------
    with tab_values:
        st.subheader("🔢 Value-Based Filtering")
        st.markdown("Filter rows based on specific column values")

        st.markdown("**Numeric Column Filtering:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_columns_set]
        if exclude_id_columns:
            numeric_cols = [c for c in numeric_cols if not is_id_column(df, c)]

        if numeric_cols:
            filter_numeric_col = st.selectbox(
                "Select Numeric Column",
                options=["None"] + numeric_cols,
                key=_k(key_prefix, "filter_numeric_col"),
            )

            if filter_numeric_col != "None":
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    min_val = st.number_input(
                        "Minimum Value",
                        value=None,
                        key=_k(key_prefix, "min_val"),
                        help="Minimum value (inclusive), leave empty for no limit",
                    )
                with col_v2:
                    max_val = st.number_input(
                        "Maximum Value",
                        value=None,
                        key=_k(key_prefix, "max_val"),
                        help="Maximum value (inclusive), leave empty for no limit",
                    )

                if min_val is not None or max_val is not None:
                    rows_before = len(filtered_df)
                    filtered_df = filter_by_value_range(filtered_df, filter_numeric_col, min_val, max_val)
                    rows_after = len(filtered_df)
                    removed = rows_before - rows_after
                    if removed > 0:
                        st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%)")

                    filter_config["numeric_value_filtering"] = {
                        "column": filter_numeric_col,
                        "min_value": min_val,
                        "max_value": max_val,
                    }

        st.markdown("**Categorical Column Filtering:**")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in exclude_columns_set]
        if exclude_id_columns:
            categorical_cols = [c for c in categorical_cols if not is_id_column(df, c)]

        if categorical_cols:
            filter_cat_col = st.selectbox(
                "Select Categorical Column",
                options=["None"] + categorical_cols,
                key=_k(key_prefix, "filter_cat_col"),
            )

            if filter_cat_col != "None":
                unique_vals = df[filter_cat_col].unique().tolist()
                filter_mode = st.radio(
                    "Filter Mode",
                    options=["Keep selected values", "Exclude selected values"],
                    key=_k(key_prefix, "filter_mode"),
                )

                if filter_mode == "Keep selected values":
                    keep_vals = st.multiselect(
                        "Values to Keep",
                        options=unique_vals,
                        key=_k(key_prefix, "keep_vals"),
                    )
                    if keep_vals:
                        rows_before = len(filtered_df)
                        filtered_df = filter_by_categorical_values(
                            filtered_df, filter_cat_col, keep_values=keep_vals
                        )
                        rows_after = len(filtered_df)
                        removed = rows_before - rows_after
                        if removed > 0:
                            st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%)")

                        filter_config["categorical_filtering"] = {
                            "column": filter_cat_col,
                            "mode": filter_mode,
                            "keep_values": keep_vals,
                            "exclude_values": [],
                        }
                else:
                    exclude_vals = st.multiselect(
                        "Values to Exclude",
                        options=unique_vals,
                        key=_k(key_prefix, "exclude_vals"),
                    )
                    if exclude_vals:
                        rows_before = len(filtered_df)
                        filtered_df = filter_by_categorical_values(
                            filtered_df, filter_cat_col, exclude_values=exclude_vals
                        )
                        rows_after = len(filtered_df)
                        removed = rows_before - rows_after
                        if removed > 0:
                            st.info(f"📊 Will remove {removed} rows ({removed/rows_before*100:.1f}%)")

                        filter_config["categorical_filtering"] = {
                            "column": filter_cat_col,
                            "mode": filter_mode,
                            "keep_values": [],
                            "exclude_values": exclude_vals,
                        }

    return filtered_df, filter_config


def render_apply_reset_filters(
    *,
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    filtered_state_key: str,
    key_prefix: str,
    artifact: Optional[ArtifactBuilder],
    artifact_filtered_df_name: str = "filtered_data",
    artifact_filtered_df_description: str = "Data after applying filters",
    artifact_log_category: str = "filtering",
    artifact_log_id: str = "current_filters",
    filter_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render Apply/Reset buttons, store filtered df in session state, and write artifact logs.
    """
    st.divider()

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("✅ Apply Filters", type="primary", use_container_width=True, key=_k(key_prefix, "apply_filters")):
            st.session_state[filtered_state_key] = filtered_df

            if artifact is not None:
                artifact.add_df(artifact_filtered_df_name, filtered_df, artifact_filtered_df_description)
                artifact.add_log(
                    category=artifact_log_category,
                    message=f"Filters applied: {len(original_df)} → {len(filtered_df)} rows ({len(filtered_df)/len(original_df)*100:.1f}% retained)",
                    details=filter_config or {},
                    log_id=artifact_log_id,
                )

            st.success(
                f"✅ Filters applied! {len(original_df)} → {len(filtered_df)} rows ({len(filtered_df)/len(original_df)*100:.1f}% retained)"
            )
            st.rerun()

    with col_btn2:
        if st.button("🔄 Reset Filters", use_container_width=True, key=_k(key_prefix, "reset_filters")):
            st.session_state[filtered_state_key] = None

            if artifact is not None:
                artifact.remove_log(category=artifact_log_category)
                if artifact_filtered_df_name in artifact.dataframes:
                    del artifact.dataframes[artifact_filtered_df_name]

            st.rerun()

