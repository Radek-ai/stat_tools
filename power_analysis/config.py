"""
Configuration parsing and validation utilities
"""
import json
from typing import Union, Tuple


DEFAULT_STATS = {
    'No Outlier Filtering': {
        'experiment': {
            'mean': 25.85,
            'std': 133.89,
            'sample_size': 50000,
            'rows_retained': 1.0,
            'metric_retained': 1.0
        },
        'group_b': {
            'mean': 18.99,
            'std': 173.37,
            'sample_size': 48000,
            'rows_retained': 1.0,
            'metric_retained': 1.0
        }
    },
    'Percentile_1.0_99.0': {
        'experiment': {
            'mean': 24.12,
            'std': 125.45,
            'sample_size': 49500,
            'rows_retained': 0.99,
            'metric_retained': 0.37
        },
        'group_b': {
            'mean': 17.85,
            'std': 162.30,
            'sample_size': 47520,
            'rows_retained': 0.89,
            'metric_retained': 0.56
        }
    }
}


def parse_statistics(stats_input: str) -> Tuple[bool, Union[dict, str]]:
    """
    Parse statistics from JSON string.
    
    Args:
        stats_input: JSON string containing statistics
        
    Returns:
        Tuple of (is_valid, scenarios_dict)
    """
    try:
        stats = json.loads(stats_input.replace("'", '"'))
        
        # Validate structure: scenario -> metric -> {mean, std, sample_size, rows_retained, metric_retained}
        scenarios = {}
        for scenario_name, scenario_data in stats.items():
            if not isinstance(scenario_data, dict):
                raise ValueError(f"Scenario '{scenario_name}' must be a dictionary of metrics")
            
            metrics = {}
            for metric_name, metric_data in scenario_data.items():
                if not isinstance(metric_data, dict) or 'mean' not in metric_data or 'std' not in metric_data:
                    raise ValueError(f"Metric '{metric_name}' in scenario '{scenario_name}' must have 'mean' and 'std' fields")
                
                metrics[metric_name] = {
                    'mean': float(metric_data['mean']),
                    'std': float(metric_data['std']),
                    'sample_size': float(metric_data.get('sample_size', 0)),
                    'rows_retained': float(metric_data.get('rows_retained', 1.0)),
                    'metric_retained': float(metric_data.get('metric_retained', 1.0))
                }
            
            if len(metrics) < 1:
                raise ValueError(f"Scenario '{scenario_name}' must have at least one metric")
            
            scenarios[scenario_name] = metrics
        
        if len(scenarios) < 1:
            raise ValueError("At least one scenario is required")
        
        return True, scenarios
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return False, str(e)


def get_default_stats_json() -> str:
    """Get default statistics as JSON string."""
    return json.dumps(DEFAULT_STATS, indent=2)

