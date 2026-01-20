# ArtifactBuilder Class Design

## Overview

The `ArtifactBuilder` class provides a progressive, flexible way to build reproducible analysis artifacts. Instead of collecting everything at the end, you add dataframes, plots, and logs as the user interacts with the application.

## Key Features

### ✅ Progressive Building
- Add dataframes, plots, and logs as operations happen
- No need to collect everything at the end
- Natural integration with existing code flow

### ✅ Flexible Data Storage
- **Dataframes**: Store with names and descriptions
- **Plots**: Store Plotly figures with names
- **Logs**: Structured log entries with categories
- **Config**: Dictionary for configuration data

### ✅ Easy Log Management
- Add logs with unique IDs
- Remove logs by ID or category (perfect for filter resets)
- Automatic timestamping

### ✅ Self-Documenting
- Automatic README generation with transformation log
- Summary of all operations
- Human-readable format

## Class Methods

### Data Management
```python
artifact.add_df(name='uploaded_data', df=df, description='Original data')
artifact.add_plot(name='balance_report', fig=fig, description='Balance visualization')
artifact.set_config({'mode': 'Advanced', 'n_groups': 2})
artifact.update_config('max_iterations', 1000)
```

### Logging
```python
# Add log entry
log_id = artifact.add_log(
    category='filtering',
    message='Outlier filtering applied',
    details={'method': 'Percentile', 'column': 'revenue'},
    log_id='filter_revenue'  # Optional: for later removal
)

# Remove logs (useful when filters are reset)
artifact.remove_log(category='filtering')  # Remove all filtering logs
artifact.remove_log(log_id='filter_revenue')  # Remove specific log
```

### Export
```python
# Create ZIP file
zip_bytes = artifact.create_zip()
# Or with custom filename
zip_bytes = artifact.create_zip(filename='my_artifact.zip')
```

### Utilities
```python
# Get summary
summary = artifact.get_summary()
# Clear everything
artifact.clear()
```

## Usage Pattern

### 1. Initialize in Session State
```python
if 'group_selection_artifact' not in st.session_state:
    st.session_state.group_selection_artifact = ArtifactBuilder('group_selection')
artifact = st.session_state.group_selection_artifact
```

### 2. Add Data When Uploaded
```python
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    artifact.add_df('uploaded_data', df, 'Original uploaded data')
    artifact.add_log('data_upload', f'Uploaded {len(df)} rows')
```

### 3. Add Logs When Filters Applied
```python
if st.button("Apply Filters"):
    filtered_df = apply_filters(df, filter_config)
    artifact.add_df('filtered_data', filtered_df, 'Data after filtering')
    artifact.add_log(
        category='filtering',
        message='Filters applied',
        details=filter_config,
        log_id='current_filters'  # Can remove later
    )
```

### 4. Remove Logs When Filters Reset
```python
if st.button("Reset Filters"):
    artifact.remove_log(category='filtering')
    if 'filtered_data' in artifact.dataframes:
        del artifact.dataframes['filtered_data']
```

### 5. Add Plots When Generated
```python
if balance_fig:
    artifact.add_plot('balance_report', balance_fig, 'Group balance visualization')
```

### 6. Add Config When Operations Complete
```python
artifact.set_config({
    'balancing_mode': 'Advanced',
    'objectives': {...},
    'parameters': {...}
})
```

### 7. Download Button
```python
if st.button("💾 Save Artifact"):
    zip_bytes = artifact.create_zip()
    st.download_button(
        label="Download Artifact",
        data=zip_bytes,
        file_name="artifact.zip",
        mime="application/zip"
    )
```

## Artifact Structure

The generated ZIP file contains:

```
artifact_group_selection_20240115_143045.zip
├── data/
│   ├── uploaded_data.csv
│   ├── filtered_data.csv
│   └── balanced_data.csv
├── plots/
│   └── balance_report.html
├── config/
│   └── configuration.json
└── README.txt
```

## README.txt Format

The README includes:
- Generation timestamp
- Page name
- All transformation logs (grouped by category)
- Summary of data files
- Summary of plots
- Configuration details

## Benefits

1. **Progressive Building**: No need to collect everything at the end
2. **Easy Integration**: Add calls where operations happen
3. **Flexible**: Handles any number of plots/dataframes
4. **Self-Documenting**: Automatic log generation
5. **Easy Cleanup**: Remove logs when operations are undone
6. **Type Safe**: Validates dataframes and plots

## Edge Cases Handled

- Empty artifacts (no data/plots)
- Missing descriptions (uses defaults)
- Log removal (by ID or category)
- Large files (uses ZIP compression)
- Special characters in filenames (handled by ZIP)

## Integration Points

### Group Selection Page
- Add uploaded data → `add_df('uploaded_data', ...)`
- Apply filters → `add_df('filtered_data', ...)` + `add_log('filtering', ...)`
- Reset filters → `remove_log(category='filtering')`
- Balance groups → `add_df('balanced_data', ...)` + `add_plot(...)` + `set_config(...)`

### Rebalancer Page
- Similar pattern to Group Selection
- Add rebalanced data instead of balanced data

### Power Analysis Page
- Add uploaded data
- Add computed statistics JSON (as config)
- Add plots (contour plots, single plots)

### Results Analysis Page
- Add uploaded data
- Add analysis plots (Basic, CUPED, DiD)
- Add analysis config

## Next Steps

1. Integrate into each page's UI components
2. Add artifact builder initialization
3. Add calls at key operation points
4. Add download button in download sections
5. Test with real workflows
