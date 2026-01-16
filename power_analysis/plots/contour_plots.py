"""
Contour plot functions with indicators
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from power_analysis.plots.utils import (
    create_slider, create_scenario_dropdown_buttons,
    PLOT_HEIGHT, SLIDER_Y_FIRST
)


def add_metric_indicators_to_subplot(fig, metrics, metric_names, num_metrics, scenario_idx, visible, row_heights):
    """Add indicator traces using domain positioning to match subplot grid
    Layout: Row 1 = 3 number indicators, Row 2 = bullet (rows), Row 3 = bullet (metric), Row 4 = contour
    
    Args:
        fig: Plotly figure
        metrics: Dictionary of metric data (metric_name -> metric_data)
        metric_names: List of metric names in the order they appear in Z_groups (must match contour trace order)
        num_metrics: Number of metrics
        scenario_idx: Index of the scenario
        visible: Whether indicators should be visible
        row_heights: List of row heights for domain calculation
    """
    # Calculate cumulative row heights for domain positioning
    # row_heights = [0.08, 0.08, 0.08, 0.76]
    total_height = sum(row_heights)
    row_starts = [0]
    for i, h in enumerate(row_heights):
        row_starts.append(row_starts[i] + h / total_height)
    
    # Use metric_names order to match contour trace order
    for metric_idx, metric_name in enumerate(metric_names):
        metric_data = metrics.get(metric_name, {})
        mean = metric_data.get('mean', 0)
        std = metric_data.get('std', 0)
        sample_size = metric_data.get('sample_size', 0)
        rows_retained = metric_data.get('rows_retained', 1.0) * 100
        metric_retained = metric_data.get('metric_retained', 1.0) * 100
        
        # Calculate x domain for this metric column
        # Account for horizontal spacing (0.1) between subplots
        horizontal_spacing = 0.1
        total_spacing = horizontal_spacing * (num_metrics - 1)
        available_width = 1.0 - total_spacing
        col_width = available_width / num_metrics
        x_start = metric_idx * (col_width + horizontal_spacing)
        x_end = x_start + col_width
        
        # Row 1: 3 number indicators (Mean, Std, Sample Size) in one row using domain
        # Split the column into 3 equal parts
        for stat_idx, (stat_value, stat_title) in enumerate([
            (mean, "Mean"),
            (std, "Std"),
            (int(sample_size), "Sample Size")
        ]):
            x_indicator_start = x_start + (stat_idx / 3.0) * col_width
            x_indicator_end = x_start + ((stat_idx + 1) / 3.0) * col_width
            
            # Row 1 domain: from top, height = row_heights[0]
            # Add extra space at top to avoid overlap with title
            y_bottom = 1.0 - row_starts[1] - 0.00  # Top of row 1, with extra margin
            y_top = 1.0 - row_starts[0] - 0.00     # Top of figure, with extra margin
            
            indicator = go.Indicator(
                mode="number",
                value=stat_value,
                title={"text": stat_title, "font": {"size": 11}},
                domain={'x': [x_indicator_start, x_indicator_end], 'y': [y_bottom, y_top]},
                number={"font": {"size": 14}}
            )
            indicator.visible = visible
            fig.add_trace(indicator)
        
        # Row 2: Bullet indicator for Rows Retained (70% width to account for labels)
        y_bottom = 1.0 - row_starts[2]  # Top of row 2
        y_top = 1.0 - row_starts[1]     # Bottom of row 1
        
        # Make bullet 70% width, centered
        bullet_width = col_width * 0.9
        bullet_x_start = x_start + (col_width - bullet_width) / 1.5
        bullet_x_end = bullet_x_start + bullet_width
        
        # Calculate color: red at 0%, green at 100% (simple gradient)
        red_component = int(255 * (1 - rows_retained / 100))
        green_component = int(255 * (rows_retained / 100))
        color = f"rgb({red_component}, {green_component}, 0)"
        
        indicator = go.Indicator(
            mode="number+gauge",
            value=rows_retained,
            title={"text": "Rows Retained", "font": {"size": 11}},
            domain={'x': [bullet_x_start, bullet_x_end], 'y': [y_bottom, y_top]},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': color},
                'bgcolor': "white",
                'bordercolor': "gray",
                'borderwidth': 1
            },
            number={"font": {"size": 14}, "suffix": "%"}
        )
        indicator.visible = visible
        fig.add_trace(indicator)
        
        # Row 3: Bullet indicator for Metric Retained (70% width to account for labels)
        y_bottom = 1.0 - row_starts[3]  # Top of row 3
        y_top = 1.0 - row_starts[2]     # Bottom of row 2
        
        # Make bullet 70% width, centered
        bullet_width = col_width * 0.9
        bullet_x_start = x_start + (col_width - bullet_width) / 1.5
        bullet_x_end = bullet_x_start + bullet_width
        
        # Calculate color: red at 0%, green at 100% (simple gradient)
        red_component = int(255 * (1 - metric_retained / 100))
        green_component = int(255 * (metric_retained / 100))
        color = f"rgb({red_component}, {green_component}, 0)"
        
        indicator = go.Indicator(
            mode="number+gauge",
            value=metric_retained,
            title={"text": "Metric Retained", "font": {"size": 11}},
            domain={'x': [bullet_x_start, bullet_x_end], 'y': [y_bottom, y_top]},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': color},
                'bgcolor': "white",
                'bordercolor': "gray",
                'borderwidth': 1
            },
            number={"font": {"size": 14}, "suffix": "%"}
        )
        indicator.visible = visible
        fig.add_trace(indicator)


def apply_contour_plot_layout(fig, slider, title, dropdown_buttons, scenarios, 
                                first_scenario, num_metrics, x_label, y_label, metric_names):
    """Apply common layout for contour plots with scenario dropdown and indicator info"""
    # Calculate column positions for annotations
    horizontal_spacing = 0.1
    total_spacing = horizontal_spacing * (num_metrics - 1)
    available_width = 1.0 - total_spacing
    col_width = available_width / num_metrics
    
    # Create annotations for column titles (metric names) at the top
    annotations = []
    for i, metric_name in enumerate(metric_names):
        x_center = i * (col_width + horizontal_spacing) + col_width / 2
        annotations.append(
            dict(
                text=f"<b>{metric_name}</b>",
                xref="paper",
                yref="paper",
                x=x_center,
                y=0.96,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=16, color="black")
            )
        )
    
    fig.update_layout(
        sliders=[slider],
        title=title,
        height=PLOT_HEIGHT,
        margin=dict(b=140, t=180),  # Increased top margin for column titles and dropdown
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.05,
                yanchor="bottom"
            )
        ],
        annotations=annotations
    )
    
    # Update axis labels for contour plots (row 4 only)
    for i in range(num_metrics):
        fig.update_xaxes(title_text=x_label, row=4, col=i+1)
        fig.update_yaxes(title_text=y_label, row=4, col=i+1)
        
        # Hide axes for indicator rows
        for row in [1, 2, 3]:
            fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=i+1)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=i+1)


def create_contour_trace(z_data, x_data, y_data, showscale, max_sample_size, contour_bins, 
                          x_label="x", y_label="y", z_label="Sample Size", available_n=None):
    """Create a contour trace with standard settings and custom hover labels"""
    # Create custom colorscale if available_n is provided
    if available_n is not None and available_n > 0:
        # Calculate the transition point (0 to 1 scale)
        transition = min(available_n / max_sample_size, 1.0)
        
        # Blues (light to dark) for achievable, greys (light to dark) for not achievable
        colorscale = [
            [0, '#e3f2fd'],           # Very light blue (lowest, most achievable)
            [transition * 0.5, '#90caf9'],  # Light blue
            [transition * 0.8, '#42a5f5'],  # Medium blue
            [transition * 0.99, '#1976d2'], # Dark blue (approaching boundary)
            [transition, "#FF0000"],   # Light grey (transition point)
            [min(transition * 1.01, 1.0), "#7C7C7C"],  # Medium grey
            [1.0, "#e1e1e1"]           # Dark grey (highest, not achievable)
        ]
    else:
        colorscale = 'Blues'  # Default blue colorscale
    
    return go.Contour(
        z=z_data,
        x=x_data,
        y=y_data,
        colorscale=colorscale,
        colorbar=dict(title="N", x=1.02) if showscale else None,
        showscale=showscale,
        zmin=0,
        zmax=max_sample_size,
        ncontours=contour_bins,
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=10)
        ),
        hovertemplate=f'<b>{x_label}</b>: %{{x}}<br><b>{y_label}</b>: %{{y}}<br><b>{z_label}</b>: %{{z:.0f}}<extra></extra>'
    )


def create_contour_plot_power(uplifts, alphas, powers, Z_scenarios, scenarios, max_sample_size, contour_bins, n_groups=2):
    """Create contour plot for uplift-alpha view with power slider and scenario dropdown"""
    scenario_names = list(Z_scenarios.keys())
    num_scenarios = len(scenario_names)
    
    # Create figure - we'll add all scenarios but show only the first one initially
    first_scenario = scenario_names[0]
    first_Z_groups = Z_scenarios[first_scenario]
    first_metrics = scenarios[first_scenario]
    num_metrics = len(first_Z_groups)
    metric_names = list(first_Z_groups.keys())
    
    # Create subplots: 4 rows (3 short for indicators, 1 tall for contour), num_metrics columns
    # Row heights: first 3 rows are short (0.08 each), last row is tall (0.76)
    row_heights = [0.15, 0.08, 0.08, 0.69]
    
    # Create subplot titles: no titles for any row (we'll add column titles as annotations)
    subplot_titles = [""] * (4 * num_metrics)
    
    fig = make_subplots(
        rows=4, cols=num_metrics,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.01,
        horizontal_spacing=0.1
    )
    
    # Track which traces belong to which scenario
    traces_per_scenario = num_metrics
    indicators_per_scenario = num_metrics * 5  # 3 number + 2 bullet per metric
    
    # Add traces organized by subplot position (column first, then scenario)
    # This matches how Plotly organizes traces in fig.data when using subplots
    # First add all contour traces (row 4) organized by column
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metrics = scenarios[scenario_name]
            metric_names = list(Z_groups.keys())
            metric_name = metric_names[col_idx]
            Z_group = Z_groups[metric_name]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            showscale = (col_idx == num_metrics - 1)  # Show scale on last column
            
            # Calculate available sample size for this metric
            available_n = None
            if metric_name in metrics:
                sample_size = metrics[metric_name].get('sample_size', 0)
                if sample_size > 0:
                    available_n = sample_size / n_groups
            
            trace = create_contour_trace(
                Z_group[0], uplifts, alphas, 
                showscale, max_sample_size, contour_bins,
                x_label="Uplift", y_label="Alpha", z_label="Sample Size",
                available_n=available_n
            )
            trace.visible = visible
            fig.add_trace(trace, row=4, col=col_idx+1)
    
    # Then add all indicators organized by scenario (indicators don't use subplots, so order doesn't matter)
    for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
        metrics = scenarios[scenario_name]
        metric_names = list(Z_groups.keys())
        visible = (scenario_idx == 0)  # Only first scenario visible initially
        add_metric_indicators_to_subplot(fig, metrics, metric_names, num_metrics, scenario_idx, visible, row_heights)
    
    # Create power slider steps (updates all visible traces)
    # Traces are organized by column (metric) first, then scenario
    def make_slider_step(k):
        z_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names = list(Z_groups.keys())
                metric_name = metric_names[metric_idx]
                z_updates.append(Z_groups[metric_name][k])
        
        return dict(
            method="update",
            args=[{
                "z": z_updates,
                "zmin": [0] * (num_scenarios * traces_per_scenario),
                "zmax": [max_sample_size] * (num_scenarios * traces_per_scenario)
            }],
            label=f"{powers[k]:.2f}"
        )
    
    steps = [make_slider_step(k) for k in range(len(powers))]
    slider = create_slider(0, "Power: ", steps, SLIDER_Y_FIRST)
    
    # Create scenario dropdown - same approach as contour plots
    dropdown_buttons = create_scenario_dropdown_buttons(
        scenario_names, scenarios, num_scenarios, traces_per_scenario, first_scenario, num_metrics, indicators_per_scenario
    )
    
    apply_contour_plot_layout(
        fig, slider, "Sample size vs uplift & alpha (fixed power)",
        dropdown_buttons, scenarios, first_scenario, num_metrics,
        "Assumed uplift", "Alpha", metric_names
    )
    
    return fig


def create_contour_plot_uplift(uplifts, alphas, powers, Z_scenarios, scenarios, max_sample_size, contour_bins, n_groups=2):
    """Create contour plot for alpha-power view with uplift slider and scenario dropdown"""
    scenario_names = list(Z_scenarios.keys())
    num_scenarios = len(scenario_names)
    
    # Create figure - we'll add all scenarios but show only the first one initially
    first_scenario = scenario_names[0]
    first_Z_groups = Z_scenarios[first_scenario]
    first_metrics = scenarios[first_scenario]
    num_metrics = len(first_Z_groups)
    metric_names = list(first_Z_groups.keys())
    
    # Create subplots: 4 rows (3 short for indicators, 1 tall for contour), num_metrics columns
    row_heights = [0.15, 0.08, 0.08, 0.69]
    
    # Create subplot titles: no titles for any row (we'll add column titles as annotations)
    subplot_titles = [""] * (4 * num_metrics)
    
    fig = make_subplots(
        rows=4, cols=num_metrics,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.01,
        horizontal_spacing=0.1
    )
    
    # Track which traces belong to which scenario
    traces_per_scenario = num_metrics
    indicators_per_scenario = num_metrics * 5  # 3 number + 2 bullet per metric
    
    # Add traces organized by subplot position (column first, then scenario)
    # First add all contour traces (row 4) organized by column
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metrics = scenarios[scenario_name]
            metric_names = list(Z_groups.keys())
            metric_name = metric_names[col_idx]
            Z_group = Z_groups[metric_name]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            z_slice_init = np.array([Z_group[k][:, 0] for k in range(len(powers))])
            showscale = (col_idx == num_metrics - 1)  # Show scale on last column
            
            # Calculate available sample size for this metric
            available_n = None
            if metric_name in metrics:
                sample_size = metrics[metric_name].get('sample_size', 0)
                if sample_size > 0:
                    available_n = sample_size / n_groups
            
            trace = create_contour_trace(
                z_slice_init, alphas, powers,
                showscale, max_sample_size, contour_bins,
                x_label="Alpha", y_label="Power", z_label="Sample Size",
                available_n=available_n
            )
            trace.visible = visible
            fig.add_trace(trace, row=4, col=col_idx+1)
    
    # Then add all indicators organized by scenario (indicators don't use subplots, so order doesn't matter)
    for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
        metrics = scenarios[scenario_name]
        metric_names = list(Z_groups.keys())
        visible = (scenario_idx == 0)  # Only first scenario visible initially
        add_metric_indicators_to_subplot(fig, metrics, metric_names, num_metrics, scenario_idx, visible, row_heights)
    
    # Create uplift slider steps (updates all visible traces)
    # Traces are organized by column (metric) first, then scenario
    def make_slider_step(j):
        z_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names = list(Z_groups.keys())
                metric_name = metric_names[metric_idx]
                z_slice = np.array([Z_groups[metric_name][k][:, j] for k in range(len(powers))])
                z_updates.append(z_slice)
        
        return dict(
            method="update",
            args=[{
                "z": z_updates,
                "zmin": [0] * (num_scenarios * traces_per_scenario),
                "zmax": [max_sample_size] * (num_scenarios * traces_per_scenario)
            }],
            label=f"{uplifts[j]:.4f}"
        )
    
    steps = [make_slider_step(j) for j in range(len(uplifts))]
    slider = create_slider(0, "Assumed uplift: ", steps, SLIDER_Y_FIRST)
    
    # Create scenario dropdown - same simple approach as contour plots
    dropdown_buttons = create_scenario_dropdown_buttons(
        scenario_names, scenarios, num_scenarios, traces_per_scenario, first_scenario, num_metrics, indicators_per_scenario
    )
    
    apply_contour_plot_layout(
        fig, slider, "Sample size vs alpha & power (fixed uplift)",
        dropdown_buttons, scenarios, first_scenario, num_metrics,
        "Alpha", "Power", metric_names
    )
    
    return fig


def create_contour_plot_alpha(uplifts, alphas, powers, Z_scenarios, scenarios, max_sample_size, contour_bins, n_groups=2):
    """Create contour plot for uplift-power view with alpha slider and scenario dropdown"""
    scenario_names = list(Z_scenarios.keys())
    num_scenarios = len(scenario_names)
    
    # Create figure - we'll add all scenarios but show only the first one initially
    first_scenario = scenario_names[0]
    first_Z_groups = Z_scenarios[first_scenario]
    first_metrics = scenarios[first_scenario]
    num_metrics = len(first_Z_groups)
    metric_names = list(first_Z_groups.keys())
    
    # Create subplots: 4 rows (3 short for indicators, 1 tall for contour), num_metrics columns
    row_heights = [0.15, 0.08, 0.08, 0.69]
    
    # Create subplot titles: no titles for any row (we'll add column titles as annotations)
    subplot_titles = [""] * (4 * num_metrics)
    
    fig = make_subplots(
        rows=4, cols=num_metrics,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.1
    )
    
    # Track which traces belong to which scenario
    traces_per_scenario = num_metrics
    indicators_per_scenario = num_metrics * 5  # 3 number + 2 bullet per metric
    
    # Add traces organized by subplot position (column first, then scenario)
    # First add all contour traces (row 4) organized by column
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metrics = scenarios[scenario_name]
            metric_names = list(Z_groups.keys())
            metric_name = metric_names[col_idx]
            Z_group = Z_groups[metric_name]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            z_slice_init = np.array([Z_group[k][0, :] for k in range(len(powers))])
            showscale = (col_idx == num_metrics - 1)  # Show scale on last column
            
            # Calculate available sample size for this metric
            available_n = None
            if metric_name in metrics:
                sample_size = metrics[metric_name].get('sample_size', 0)
                if sample_size > 0:
                    available_n = sample_size / n_groups
            
            trace = create_contour_trace(
                z_slice_init, uplifts, powers,
                showscale, max_sample_size, contour_bins,
                x_label="Uplift", y_label="Power", z_label="Sample Size",
                available_n=available_n
            )
            trace.visible = visible
            fig.add_trace(trace, row=4, col=col_idx+1)
    
    # Then add all indicators organized by scenario (indicators don't use subplots, so order doesn't matter)
    for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
        metrics = scenarios[scenario_name]
        metric_names = list(Z_groups.keys())
        visible = (scenario_idx == 0)  # Only first scenario visible initially
        add_metric_indicators_to_subplot(fig, metrics, metric_names, num_metrics, scenario_idx, visible, row_heights)
    
    # Create alpha slider steps (updates all visible traces)
    # Traces are organized by column (metric) first, then scenario
    def make_slider_step(i):
        z_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names = list(Z_groups.keys())
                metric_name = metric_names[metric_idx]
                z_slice = np.array([Z_groups[metric_name][k][i, :] for k in range(len(powers))])
                z_updates.append(z_slice)
        
        return dict(
            method="update",
            args=[{
                "z": z_updates,
                "zmin": [0] * (num_scenarios * traces_per_scenario),
                "zmax": [max_sample_size] * (num_scenarios * traces_per_scenario)
            }],
            label=f"{alphas[i]:.3f}"
        )
    
    steps = [make_slider_step(i) for i in range(len(alphas))]
    slider = create_slider(0, "Alpha: ", steps, SLIDER_Y_FIRST)
    
    # Create scenario dropdown - same simple approach as contour plots
    dropdown_buttons = create_scenario_dropdown_buttons(
        scenario_names, scenarios, num_scenarios, traces_per_scenario, first_scenario, num_metrics, indicators_per_scenario
    )
    
    apply_contour_plot_layout(
        fig, slider, "Sample size vs uplift & power (fixed alpha)",
        dropdown_buttons, scenarios, first_scenario, num_metrics,
        "Assumed uplift", "Power", metric_names
    )
    
    return fig

