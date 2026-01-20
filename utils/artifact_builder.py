"""
Artifact Builder - Class for building reproducible analysis artifacts

This class allows progressive building of artifacts by adding dataframes,
plots, and log entries as the user interacts with the application.
"""
import zipfile
import io
import json
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
import pandas as pd
import plotly.graph_objects as go


class ArtifactBuilder:
    """
    Builder class for creating reproducible analysis artifacts.
    
    Stores dataframes, plots, and log entries progressively,
    then exports everything to a ZIP file.
    """
    
    def __init__(self, page_name: str):
        """
        Initialize the artifact builder.
        
        Parameters:
        -----------
        page_name : str
            Name of the page (e.g., 'power_analysis', 'group_selection')
        """
        self.page_name = page_name
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.plots: Dict[str, go.Figure] = {}
        self.logs: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        self.created_at = datetime.now()
    
    def add_df(self, name: str, df: pd.DataFrame, description: str = None):
        """
        Add a dataframe to the artifact.
        
        Parameters:
        -----------
        name : str
            Unique name for the dataframe (e.g., 'uploaded_data', 'filtered_data')
        df : pd.DataFrame
            The dataframe to store
        description : str, optional
            Description of what this dataframe represents
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        
        self.dataframes[name] = {
            'df': df,
            'description': description or f"Dataframe: {name}",
            'rows': len(df),
            'columns': len(df.columns),
            'added_at': datetime.now().isoformat()
        }
    
    def add_plot(self, name: str, fig: go.Figure, description: str = None):
        """
        Add a plot to the artifact.
        
        Parameters:
        -----------
        name : str
            Unique name for the plot (e.g., 'balance_report', 'contour_plot')
        fig : go.Figure
            The Plotly figure to store
        description : str, optional
            Description of what this plot shows
        """
        if not isinstance(fig, go.Figure):
            raise TypeError(f"Expected go.Figure, got {type(fig)}")
        
        self.plots[name] = {
            'fig': fig,
            'description': description or f"Plot: {name}",
            'added_at': datetime.now().isoformat()
        }
    
    def add_log(self, 
                category: str,
                message: str,
                details: Dict[str, Any] = None,
                log_id: str = None):
        """
        Add a log entry to the artifact.
        
        Parameters:
        -----------
        category : str
            Category of the log (e.g., 'data_upload', 'filtering', 'balancing')
        message : str
            Human-readable message
        details : dict, optional
            Additional details about the operation
        log_id : str, optional
            Unique ID for this log entry (for later removal if needed)
        """
        log_entry = {
            'id': log_id or f"{category}_{len(self.logs)}",
            'category': category,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        return log_entry['id']
    
    def remove_log(self, log_id: str = None, category: str = None):
        """
        Remove log entries by ID or category.
        
        Parameters:
        -----------
        log_id : str, optional
            Remove log entry with this ID
        category : str, optional
            Remove all log entries with this category
        
        Returns:
        --------
        int : Number of log entries removed
        """
        initial_count = len(self.logs)
        
        if log_id:
            self.logs = [log for log in self.logs if log['id'] != log_id]
        elif category:
            self.logs = [log for log in self.logs if log['category'] != category]
        else:
            raise ValueError("Must provide either log_id or category")
        
        return initial_count - len(self.logs)
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set configuration dictionary.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary (will be merged with existing config)
        """
        self.config.update(config)
    
    def update_config(self, key: str, value: Any):
        """
        Update a single configuration value.
        
        Parameters:
        -----------
        key : str
            Configuration key
        value : Any
            Configuration value
        """
        self.config[key] = value
    
    def create_zip(self, filename: Optional[str] = None) -> bytes:
        """
        Create a ZIP file containing all stored data, plots, config, and logs.
        
        Parameters:
        -----------
        filename : str, optional
            Custom filename (default: artifact_[page_name]_[timestamp].zip)
        
        Returns:
        --------
        bytes : ZIP file as bytes
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"artifact_{self.page_name}_{timestamp}.zip"
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add dataframes
            if self.dataframes:
                for name, df_info in self.dataframes.items():
                    csv_buffer = df_info['df'].to_csv(index=False)
                    zip_file.writestr(f"data/{name}.csv", csv_buffer)
            
            # Add plots
            if self.plots:
                for name, plot_info in self.plots.items():
                    html_buffer = plot_info['fig'].to_html(include_plotlyjs='cdn')
                    zip_file.writestr(f"plots/{name}.html", html_buffer)
            
            # Add configuration
            if self.config:
                config_json = json.dumps(self.config, indent=2, default=str)
                zip_file.writestr("config/configuration.json", config_json)
            
            # Add README with transformation log
            readme_content = self._generate_readme()
            zip_file.writestr("README.txt", readme_content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def _generate_readme(self) -> str:
        """
        Generate README file with transformation log.
        
        Returns:
        --------
        str : README content
        """
        lines = [
            "=" * 70,
            "ARTIFACT SAVE LOG",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Page: {self.page_name.replace('_', ' ').title()}",
            "",
            "DATA TRANSFORMATIONS",
            "-" * 70,
        ]
        
        # Group logs by category
        logs_by_category: Dict[str, List[Dict]] = {}
        for log in self.logs:
            category = log['category']
            if category not in logs_by_category:
                logs_by_category[category] = []
            logs_by_category[category].append(log)
        
        # Write logs by category
        for category, category_logs in logs_by_category.items():
            lines.append(f"\n{category.upper().replace('_', ' ')}")
            for log in category_logs:
                lines.append(f"  - {log['message']}")
                if log.get('details'):
                    for key, value in log['details'].items():
                        if isinstance(value, (dict, list)):
                            value_str = json.dumps(value, indent=4, default=str)
                            lines.append(f"    {key}:")
                            for detail_line in value_str.split('\n'):
                                lines.append(f"      {detail_line}")
                        else:
                            lines.append(f"    {key}: {value}")
        
        # Add dataframes summary
        if self.dataframes:
            lines.append("\n" + "DATA FILES INCLUDED")
            lines.append("-" * 70)
            for name, df_info in self.dataframes.items():
                lines.append(f"  - data/{name}.csv: {df_info['description']}")
                lines.append(f"    Rows: {df_info['rows']:,}, Columns: {df_info['columns']}")
        
        # Add plots summary
        if self.plots:
            lines.append("\n" + "PLOTS INCLUDED")
            lines.append("-" * 70)
            for name, plot_info in self.plots.items():
                lines.append(f"  - plots/{name}.html: {plot_info['description']}")
        
        # Add configuration summary
        if self.config:
            lines.append("\n" + "CONFIGURATION")
            lines.append("-" * 70)
            for key, value in self.config.items():
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=2, default=str)
                    lines.append(f"  {key}:")
                    for config_line in value_str.split('\n'):
                        lines.append(f"    {config_line}")
                else:
                    lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "=" * 70)
        lines.append("END OF LOG")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the artifact contents.
        
        Returns:
        --------
        dict : Summary of artifact contents
        """
        return {
            'page_name': self.page_name,
            'created_at': self.created_at.isoformat(),
            'dataframes_count': len(self.dataframes),
            'plots_count': len(self.plots),
            'logs_count': len(self.logs),
            'dataframes': list(self.dataframes.keys()),
            'plots': list(self.plots.keys()),
            'log_categories': list(set(log['category'] for log in self.logs))
        }
    
    def clear(self):
        """Clear all stored data (useful for reset operations)."""
        self.dataframes.clear()
        self.plots.clear()
        self.logs.clear()
        self.config.clear()
    
    def __repr__(self) -> str:
        """String representation of the artifact builder."""
        summary = self.get_summary()
        return (f"ArtifactBuilder(page='{self.page_name}', "
                f"dataframes={summary['dataframes_count']}, "
                f"plots={summary['plots_count']}, "
                f"logs={summary['logs_count']})")
