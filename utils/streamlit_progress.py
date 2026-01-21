"""
Streamlit progress/status callback utilities.

Used by long-running algorithms to report progress in a consistent way.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st


def create_streamlit_progress_callback(
    progress_placeholder: "st.delta_generator.DeltaGenerator",
    status_placeholder: "st.delta_generator.DeltaGenerator",
    *,
    default_description: str = "Working",
    show_step_info: bool = True,
) -> Callable[[str, Dict[str, Any]], None]:
    """
    Create a callback(stage, info) suitable for algorithm progress reporting.

    Supported stages: "start", "update", "complete".
    """

    def callback(stage: str, info: Dict[str, Any]) -> None:
        if stage == "start":
            progress_placeholder.progress(0.0)
            desc = info.get("description", default_description)
            initial_loss = info.get("initial_loss", 0.0)
            step_info = info.get("step_info", "") if show_step_info else ""
            step_prefix = f"{step_info} - " if step_info else ""
            status_placeholder.info(f"🔄 {step_prefix}{desc} - Initial loss: {initial_loss:.4f}")

        elif stage == "update":
            iteration = info.get("iteration", 0)
            total = info.get("total", 1)
            initial_loss = info.get("initial_loss", 0.0)
            current_loss = info.get("current_loss", 0.0)
            gain = info.get("gain", 0.0)
            progress = info.get("progress", 0.0)

            step_info = info.get("step_info", "") if show_step_info else ""
            step_prefix = f"{step_info} - " if step_info else ""

            progress_placeholder.progress(progress)
            status_placeholder.info(
                f"🔄 {step_prefix}Iteration {iteration} / {total} | "
                f"Initial: {initial_loss:.4f} | "
                f"Current: {current_loss:.4f} | "
                f"Gain: {gain:.4f}"
            )

        elif stage == "complete":
            progress_placeholder.progress(1.0)
            final_loss = info.get("final_loss", 0.0)
            total_gain = info.get("total_gain", 0.0)
            step_info = info.get("step_info", "") if show_step_info else ""
            step_prefix = f"{step_info} - " if step_info else ""
            status_placeholder.success(
                f"✅ {step_prefix}Complete! Final loss: {final_loss:.4f} | "
                f"Total gain: {total_gain:.4f}"
            )

    return callback

