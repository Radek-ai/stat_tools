"""
Streamlit helpers for working with ArtifactBuilder.

Goal: keep artifact creation and download-button behavior consistent across pages,
without duplicating the same logic in each `*/streamlit_page.py`.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from utils.artifact_builder import ArtifactBuilder


def get_or_create_artifact(page_name: str, state_key: Optional[str] = None) -> ArtifactBuilder:
    """
    Get or initialize an ArtifactBuilder in session state.

    Args:
        page_name: Artifact page name (used for internal metadata + default filenames)
        state_key: Session-state key to store the artifact. Defaults to `{page_name}_artifact`.
    """
    key = state_key or f"{page_name}_artifact"
    if key not in st.session_state:
        st.session_state[key] = ArtifactBuilder(page_name=page_name)
    return st.session_state[key]


def artifact_has_content(
    artifact: ArtifactBuilder,
    *,
    include_config: bool = True,
    include_logs: bool = True,
) -> bool:
    """Return True if artifact contains anything worth downloading."""
    if len(artifact.dataframes) > 0:
        return True
    if len(artifact.plots) > 0:
        return True
    if include_config and len(artifact.config) > 0:
        return True
    if include_logs and len(artifact.logs) > 0:
        return True
    return False


def render_download_artifact_button(
    *,
    page_name: str,
    state_key: Optional[str] = None,
    label: str = "💾 Download Artifact",
    file_name: Optional[str] = None,
    mime: str = "application/zip",
    use_container_width: bool = True,
    include_config: bool = True,
    include_logs: bool = True,
    help_enabled: str = "Download complete artifact with data, plots, and transformation log",
    help_disabled: str = "Upload data and perform operations to create an artifact",
) -> None:
    """
    Render a Streamlit download button for the current page's artifact.

    This function standardizes the enable/disable conditions and button behavior
    across all pages.
    """
    artifact = get_or_create_artifact(page_name=page_name, state_key=state_key)
    if file_name is None:
        file_name = f"artifact_{page_name}.zip"

    has_data = artifact_has_content(
        artifact,
        include_config=include_config,
        include_logs=include_logs,
    )

    if has_data:
        try:
            zip_bytes = artifact.create_zip()
            st.download_button(
                label=label,
                data=zip_bytes,
                file_name=file_name,
                mime=mime,
                use_container_width=use_container_width,
                help=help_enabled,
            )
        except Exception as e:
            st.error(f"Error creating artifact: {str(e)}")
    else:
        st.download_button(
            label=label,
            data=b"",
            file_name="",
            disabled=True,
            use_container_width=use_container_width,
            help=help_disabled,
        )

