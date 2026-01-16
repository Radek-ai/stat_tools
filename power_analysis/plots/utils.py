"""
Common utilities and constants for plot functions
"""
import plotly.graph_objects as go

# Constants
PLOT_HEIGHT = 1000
SLIDER_Y_FIRST = -0.15
SLIDER_Y_SECOND = -0.28


def get_color(index):
    """Get a color from a predefined palette for plot traces"""
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf'   # cyan
    ]
    return colors[index % len(colors)]


def create_slider(active, label, steps, y_position, suffix=None):
    """Create a Plotly slider configuration
    
    Args:
        active: Initial active step index
        label: Slider label text
        steps: List of step dictionaries
        y_position: Y position of the slider (0-1, paper coordinates)
        suffix: Optional suffix to add to the label (e.g., "%")
    
    Returns:
        Dictionary with slider configuration
    """
    label_text = label
    if suffix:
        label_text = f"{label}({suffix})"
    
    return dict(
        active=active,
        currentvalue={"prefix": label_text, "visible": True},
        pad={"t": 50},
        steps=steps,
        y=y_position,
        yanchor="bottom"
    )


def create_scenario_dropdown_buttons(scenario_names, scenarios, num_scenarios, 
                                     traces_per_scenario, first_scenario, 
                                     num_metrics, indicators_per_scenario):
    """Create dropdown buttons for scenario selection
    
    Args:
        scenario_names: List of scenario names in order
        scenarios: Dictionary of scenario data
        num_scenarios: Number of scenarios
        traces_per_scenario: Number of plot traces per scenario (typically num_metrics)
        first_scenario: Name of the first scenario (initially visible)
        num_metrics: Number of metrics
        indicators_per_scenario: Number of indicator traces per scenario
    
    Returns:
        List of button dictionaries for the dropdown menu
    """
    buttons = []
    
    for scenario_idx, scenario_name in enumerate(scenario_names):
        # Calculate trace indices for this scenario
        # Plot traces are organized: column-first, then scenario
        # So for 2 metrics and 2 scenarios: metric0_scen0, metric0_scen1, metric1_scen0, metric1_scen1
        # Trace index = col_idx * num_scenarios + scenario_idx
        # For a given scenario_idx, traces are at: [scenario_idx, num_scenarios + scenario_idx, 2*num_scenarios + scenario_idx, ...]
        
        # Build visibility list: True for this scenario's traces, False for others
        visible = []
        
        # Set visibility for plot traces (organized column-first, then scenario)
        total_plot_traces = num_scenarios * traces_per_scenario
        for i in range(total_plot_traces):
            # Calculate which scenario this trace belongs to
            # Trace index = col_idx * num_scenarios + scenario_idx
            # So scenario_idx = trace_index % num_scenarios
            trace_scenario_idx = i % num_scenarios
            visible.append(trace_scenario_idx == scenario_idx)
        
        # Set visibility for indicator traces (organized scenario-first)
        # Indicator traces come after all plot traces
        total_indicators = num_scenarios * indicators_per_scenario
        for i in range(total_indicators):
            # Indicators are organized scenario-first, so scenario_idx = i // indicators_per_scenario
            indicator_scenario_idx = i // indicators_per_scenario
            visible.append(indicator_scenario_idx == scenario_idx)
        
        buttons.append(
            dict(
                label=scenario_name,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Scenario: {scenario_name}"}
                ]
            )
        )
    
    return buttons
