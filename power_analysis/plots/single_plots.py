"""
Single parameter plot functions (line plots)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from power_analysis.plots.utils import (
    get_color, create_slider,
    create_scenario_dropdown_buttons, PLOT_HEIGHT,
    SLIDER_Y_FIRST, SLIDER_Y_SECOND
)
from power_analysis.plots.contour_plots import add_metric_indicators_to_subplot


def apply_single_plot_layout(fig, sliders, title, dropdown_buttons, scenarios, 
                             first_scenario, num_metrics, x_label, y_label, metric_names, max_sample_size=None):
    """Apply common layout for single plots with scenario dropdown and indicators"""
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
        sliders=sliders,
        title=title,
        height=PLOT_HEIGHT,
        margin=dict(b=140, t=180),  # Match contour plots
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="bottom"
            )
        ],
        annotations=annotations
    )
    
    # Update axis labels for line plots (row 4 only)
    for i in range(num_metrics):
        fig.update_xaxes(title_text=x_label, row=4, col=i+1)
        fig.update_yaxes(title_text=y_label, row=4, col=i+1)
        
        if max_sample_size:
            fig.update_yaxes(range=[0, max_sample_size], row=4, col=i+1)
        
        # Hide axes for indicator rows
        for row in [1, 2, 3]:
            fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=i+1)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=i+1)


def create_single_plot_uplift(uplifts, alphas, powers, Z_scenarios, scenarios, max_sample_size=None):
    """Create single plot for Sample Size vs Uplift with sliders and scenario dropdown"""
    scenario_names = list(Z_scenarios.keys())
    num_scenarios = len(scenario_names)
    
    # Initial indices
    init_alpha_idx = len(alphas) // 2
    init_power_idx = int(len(powers) * 0.6)
    
    # Get first scenario info
    first_scenario = scenario_names[0]
    first_Z_groups = Z_scenarios[first_scenario]
    first_metrics = scenarios[first_scenario]
    num_metrics = len(first_Z_groups)
    metric_names = list(first_Z_groups.keys())
    
    # Create subplots: 4 rows (3 short for indicators, 1 tall for line plot), num_metrics columns
    row_heights = [0.15, 0.08, 0.08, 0.69]
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
    # First add all line traces (row 4) organized by column
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metric_names_list = list(Z_groups.keys())
            metric_name = metric_names_list[col_idx]
            Z_group = Z_groups[metric_name]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            trace = go.Scatter(
                x=uplifts * 100,
                y=Z_group[init_power_idx][init_alpha_idx, :],
                name=metric_name,
                mode='lines',
                line=dict(width=3, color=get_color(col_idx)),
                visible=visible
            )
            fig.add_trace(trace, row=4, col=col_idx+1)
    
    # Add horizontal dashed lines at available sample size (after all traces are added)
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metrics = scenarios[scenario_name]
            metric_names_list = list(Z_groups.keys())
            metric_name = metric_names_list[col_idx]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            # Calculate available sample size for this metric
            available_n = None
            if metric_name in metrics:
                sample_size = metrics[metric_name].get('sample_size', 0)
                if sample_size > 0:
                    available_n = sample_size / 2  # n_groups = 2 (control + treatment)
            
            # Add horizontal line if available_n is valid
            if available_n is not None and available_n > 0:
                fig.add_hline(
                    y=available_n,
                    line_dash="dash",
                    line_color="#FF0000",
                    line_width=2,
                    row=4,
                    col=col_idx+1,
                    opacity=0.8 if visible else 0.0,
                    layer="below"  # Draw behind the main lines
                )
    
    # Then add all indicators organized by scenario
    for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
        metrics = scenarios[scenario_name]
        metric_names_list = list(Z_groups.keys())
        visible = (scenario_idx == 0)  # Only first scenario visible initially
        add_metric_indicators_to_subplot(fig, metrics, metric_names_list, num_metrics, scenario_idx, visible, row_heights)
    
    # Create alpha slider steps (updates all visible traces)
    # Traces are organized by column (metric) first, then scenario
    def make_alpha_step(i):
        y_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names_list = list(Z_groups.keys())
                metric_name = metric_names_list[metric_idx]
                y_updates.append(Z_groups[metric_name][init_power_idx][i, :])
        return dict(
            method="update",
            args=[{"y": y_updates}],
            label=f"{alphas[i]:.3f}"
        )
    
    # Create power slider steps
    def make_power_step(j):
        y_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names_list = list(Z_groups.keys())
                metric_name = metric_names_list[metric_idx]
                y_updates.append(Z_groups[metric_name][j][init_alpha_idx, :])
        return dict(
            method="update",
            args=[{"y": y_updates}],
            label=f"{powers[j]:.2f}"
        )
    
    alpha_steps = [make_alpha_step(i) for i in range(len(alphas))]
    power_steps = [make_power_step(j) for j in range(len(powers))]
    
    sliders = [
        create_slider(init_alpha_idx, "Fixed Alpha: ", alpha_steps, SLIDER_Y_FIRST),
        create_slider(init_power_idx, "Fixed Power: ", power_steps, SLIDER_Y_SECOND)
    ]
    
    # Create scenario dropdown - same approach as contour plots
    dropdown_buttons = create_scenario_dropdown_buttons(
        scenario_names, scenarios, num_scenarios, traces_per_scenario, first_scenario, num_metrics, indicators_per_scenario
    )
    
    apply_single_plot_layout(
        fig, sliders, "Sample Size vs Uplift",
        dropdown_buttons, scenarios, first_scenario, num_metrics,
        "Assumed Uplift (%)", "Required Sample Size", metric_names, max_sample_size
    )
    
    return fig


def create_single_plot_alpha(uplifts, alphas, powers, Z_scenarios, scenarios, max_sample_size=None):
    """Create single plot for Sample Size vs Alpha with sliders and scenario dropdown"""
    scenario_names = list(Z_scenarios.keys())
    num_scenarios = len(scenario_names)
    
    # Initial indices
    init_uplift_idx = len(uplifts) // 4
    init_power_idx = int(len(powers) * 0.6)
    
    # Get first scenario info
    first_scenario = scenario_names[0]
    first_Z_groups = Z_scenarios[first_scenario]
    first_metrics = scenarios[first_scenario]
    num_metrics = len(first_Z_groups)
    metric_names = list(first_Z_groups.keys())
    
    # Create subplots: 4 rows (3 short for indicators, 1 tall for line plot), num_metrics columns
    row_heights = [0.15, 0.08, 0.08, 0.69]
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
    # First add all line traces (row 4) organized by column
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metric_names_list = list(Z_groups.keys())
            metric_name = metric_names_list[col_idx]
            Z_group = Z_groups[metric_name]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            trace = go.Scatter(
                x=alphas,
                y=Z_group[init_power_idx][:, init_uplift_idx],
                name=metric_name,
                mode='lines',
                line=dict(width=3, color=get_color(col_idx)),
                visible=visible
            )
            fig.add_trace(trace, row=4, col=col_idx+1)
    
    # Add horizontal dashed lines at available sample size (after all traces are added)
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metrics = scenarios[scenario_name]
            metric_names_list = list(Z_groups.keys())
            metric_name = metric_names_list[col_idx]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            # Calculate available sample size for this metric
            available_n = None
            if metric_name in metrics:
                sample_size = metrics[metric_name].get('sample_size', 0)
                if sample_size > 0:
                    available_n = sample_size / 2  # n_groups = 2 (control + treatment)
            
            # Add horizontal line if available_n is valid
            if available_n is not None and available_n > 0:
                fig.add_hline(
                    y=available_n,
                    line_dash="dash",
                    line_color="#FF0000",
                    line_width=2,
                    row=4,
                    col=col_idx+1,
                    opacity=0.8 if visible else 0.0,
                    layer="below"  # Draw behind the main lines
                )
    
    # Then add all indicators organized by scenario
    for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
        metrics = scenarios[scenario_name]
        metric_names_list = list(Z_groups.keys())
        visible = (scenario_idx == 0)  # Only first scenario visible initially
        add_metric_indicators_to_subplot(fig, metrics, metric_names_list, num_metrics, scenario_idx, visible, row_heights)
    
    # Create uplift slider steps (updates all visible traces)
    # Traces are organized by column (metric) first, then scenario
    def make_uplift_step(j):
        y_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names_list = list(Z_groups.keys())
                metric_name = metric_names_list[metric_idx]
                y_updates.append(Z_groups[metric_name][init_power_idx][:, j])
        return dict(
            method="update",
            args=[{"y": y_updates}],
            label=f"{uplifts[j]*100:.2f}"
        )
    
    # Create power slider steps
    def make_power_step(k):
        y_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names_list = list(Z_groups.keys())
                metric_name = metric_names_list[metric_idx]
                y_updates.append(Z_groups[metric_name][k][:, init_uplift_idx])
        return dict(
            method="update",
            args=[{"y": y_updates}],
            label=f"{powers[k]:.2f}"
        )
    
    uplift_steps = [make_uplift_step(j) for j in range(len(uplifts))]
    power_steps = [make_power_step(k) for k in range(len(powers))]
    
    sliders = [
        create_slider(init_uplift_idx, "Fixed Uplift: ", uplift_steps, SLIDER_Y_FIRST, "%"),
        create_slider(init_power_idx, "Fixed Power: ", power_steps, SLIDER_Y_SECOND)
    ]
    
    # Create scenario dropdown - same approach as contour plots
    dropdown_buttons = create_scenario_dropdown_buttons(
        scenario_names, scenarios, num_scenarios, traces_per_scenario, first_scenario, num_metrics, indicators_per_scenario
    )
    
    apply_single_plot_layout(
        fig, sliders, "Sample Size vs Alpha",
        dropdown_buttons, scenarios, first_scenario, num_metrics,
        "Alpha (Significance Level)", "Required Sample Size", metric_names, max_sample_size
    )
    
    return fig


def create_single_plot_power(uplifts, alphas, powers, Z_scenarios, scenarios, max_sample_size=None):
    """Create single plot for Sample Size vs Power with sliders and scenario dropdown"""
    scenario_names = list(Z_scenarios.keys())
    num_scenarios = len(scenario_names)
    
    # Initial indices
    init_uplift_idx = len(uplifts) // 4
    init_alpha_idx = len(alphas) // 10
    
    # Get first scenario info
    first_scenario = scenario_names[0]
    first_Z_groups = Z_scenarios[first_scenario]
    first_metrics = scenarios[first_scenario]
    num_metrics = len(first_Z_groups)
    metric_names = list(first_Z_groups.keys())
    
    # Create subplots: 4 rows (3 short for indicators, 1 tall for line plot), num_metrics columns
    row_heights = [0.15, 0.08, 0.08, 0.69]
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
    # First add all line traces (row 4) organized by column
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metric_names_list = list(Z_groups.keys())
            metric_name = metric_names_list[col_idx]
            Z_group = Z_groups[metric_name]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            y_data = np.array([Z_group[k][init_alpha_idx, init_uplift_idx] for k in range(len(powers))])
            trace = go.Scatter(
                x=powers,
                y=y_data,
                name=metric_name,
                mode='lines',
                line=dict(width=3, color=get_color(col_idx)),
                visible=visible
            )
            fig.add_trace(trace, row=4, col=col_idx+1)
    
    # Add horizontal dashed lines at available sample size (after all traces are added)
    for col_idx in range(num_metrics):
        for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
            metrics = scenarios[scenario_name]
            metric_names_list = list(Z_groups.keys())
            metric_name = metric_names_list[col_idx]
            visible = (scenario_idx == 0)  # Only first scenario visible initially
            
            # Calculate available sample size for this metric
            available_n = None
            if metric_name in metrics:
                sample_size = metrics[metric_name].get('sample_size', 0)
                if sample_size > 0:
                    available_n = sample_size / 2  # n_groups = 2 (control + treatment)
            
            # Add horizontal line if available_n is valid
            if available_n is not None and available_n > 0:
                fig.add_hline(
                    y=available_n,
                    line_dash="dash",
                    line_color="#FF0000",
                    line_width=2,
                    row=4,
                    col=col_idx+1,
                    opacity=0.8 if visible else 0.0,
                    layer="below"  # Draw behind the main lines
                )
    
    # Then add all indicators organized by scenario
    for scenario_idx, (scenario_name, Z_groups) in enumerate(Z_scenarios.items()):
        metrics = scenarios[scenario_name]
        metric_names_list = list(Z_groups.keys())
        visible = (scenario_idx == 0)  # Only first scenario visible initially
        add_metric_indicators_to_subplot(fig, metrics, metric_names_list, num_metrics, scenario_idx, visible, row_heights)
    
    # Create uplift slider steps (updates all visible traces)
    # Traces are organized by column (metric) first, then scenario
    def make_uplift_step(j):
        y_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names_list = list(Z_groups.keys())
                metric_name = metric_names_list[metric_idx]
                y_updates.append(np.array([Z_groups[metric_name][k][init_alpha_idx, j] for k in range(len(powers))]))
        return dict(
            method="update",
            args=[{"y": y_updates}],
            label=f"{uplifts[j]*100:.2f}"
        )
    
    # Create alpha slider steps
    def make_alpha_step(i):
        y_updates = []
        for metric_idx in range(num_metrics):
            for scenario_name in scenario_names:
                Z_groups = Z_scenarios[scenario_name]
                metric_names_list = list(Z_groups.keys())
                metric_name = metric_names_list[metric_idx]
                y_updates.append(np.array([Z_groups[metric_name][k][i, init_uplift_idx] for k in range(len(powers))]))
        return dict(
            method="update",
            args=[{"y": y_updates}],
            label=f"{alphas[i]:.3f}"
        )
    
    uplift_steps = [make_uplift_step(j) for j in range(len(uplifts))]
    alpha_steps = [make_alpha_step(i) for i in range(len(alphas))]
    
    sliders = [
        create_slider(init_uplift_idx, "Fixed Uplift: ", uplift_steps, SLIDER_Y_FIRST, "%"),
        create_slider(init_alpha_idx, "Fixed Alpha: ", alpha_steps, SLIDER_Y_SECOND)
    ]
    
    # Create scenario dropdown - same approach as contour plots
    dropdown_buttons = create_scenario_dropdown_buttons(
        scenario_names, scenarios, num_scenarios, traces_per_scenario, first_scenario, num_metrics, indicators_per_scenario
    )
    
    apply_single_plot_layout(
        fig, sliders, "Sample Size vs Power",
        dropdown_buttons, scenarios, first_scenario, num_metrics,
        "Power (1 - β)", "Required Sample Size", metric_names, max_sample_size
    )
    
    return fig

