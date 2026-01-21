"""
Streamlit helpers for downloading rendered artifacts (e.g., Plotly figures).

Kept in `utils/` so other pages don't need cross-domain imports.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st


def render_download_button(fig: go.Figure, filename: str, help_text: Optional[str] = None) -> None:
    """Render a download button for a Plotly figure as a standalone interactive HTML."""
    html_buffer = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        label="💾 Download as Interactive HTML",
        data=html_buffer,
        file_name=filename,
        mime="text/html",
        help=help_text or "Download this plot as a standalone HTML file with full interactivity",
    )


def render_plot_with_download(fig: go.Figure, filename: str, help_text: Optional[str] = None) -> None:
    """Render a Plotly figure along with an HTML download button."""
    st.plotly_chart(fig, use_container_width=True)
    render_download_button(fig, filename, help_text)

