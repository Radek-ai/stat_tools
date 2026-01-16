"""
Shared UI components used across multiple pages
"""
import streamlit as st
import plotly.graph_objects as go


def render_download_button(fig: go.Figure, filename: str, help_text: str = None):
    """
    Render a download button for a Plotly figure.
    
    Args:
        fig: Plotly figure object
        filename: Name of the file to download
        help_text: Optional help text for the button
    """
    html_buffer = fig.to_html(include_plotlyjs='cdn')
    st.download_button(
        label="💾 Download as Interactive HTML",
        data=html_buffer,
        file_name=filename,
        mime="text/html",
        help=help_text or f"Download this plot as a standalone HTML file with full interactivity"
    )


def render_plot_with_download(fig: go.Figure, filename: str, help_text: str = None):
    """
    Render a Plotly figure with download button.
    
    Args:
        fig: Plotly figure object
        filename: Name of the file to download
        help_text: Optional help text for the download button
    """
    st.plotly_chart(fig, use_container_width=True)
    render_download_button(fig, filename, help_text)


def render_scenario_selector(scenarios: dict, key: str = "scenario_selector"):
    """
    Render a scenario selector dropdown with retention info.
    
    Args:
        scenarios: Dictionary of scenarios
        key: Unique key for the selectbox
        
    Returns:
        Tuple of (selected_scenario_name, selected_scenario_metrics) or (None, None)
    """
    scenario_names = list(scenarios.keys()) if scenarios else []
    
    if not scenario_names:
        return None, None
    
    selected_scenario = st.selectbox(
        "Select Scenario:",
        scenario_names,
        key=key
    )
    
    if selected_scenario in scenarios:
        metrics = scenarios[selected_scenario]
        retention_parts = []
        for metric_name, metric_data in metrics.items():
            rows_pct = metric_data.get('rows_retained', 1.0) * 100
            metric_pct = metric_data.get('metric_retained', 1.0) * 100
            retention_parts.append(f"**{metric_name}**: {rows_pct:.0f}% rows, {metric_pct:.0f}% metric")
        
        if retention_parts:
            st.info(" | ".join(retention_parts))
        
        return selected_scenario, metrics
    
    return None, None


def render_validation_message(stats_valid: bool, computed_data_available: bool = None):
    """
    Render validation messages for common states.
    
    Args:
        stats_valid: Whether statistics are valid
        computed_data_available: Whether computed data is available (optional)
    """
    if not stats_valid:
        st.warning("⚠️ Please enter valid statistics above first")
        return False
    
    if computed_data_available is not None and not computed_data_available:
        st.info("👆 Click 'Compute All Plots' button above to generate plots")
        return False
    
    return True

