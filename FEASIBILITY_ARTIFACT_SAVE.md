# Feasibility Analysis: Save Artifact Feature

## Overview
Create a "Save Artifact" feature that packages all data, results, plots, and transformation logs into a downloadable ZIP file for reproducibility.

## Feasibility: ✅ HIGHLY FEASIBLE

### Available Technologies
- Python `zipfile` module (built-in) - for creating ZIP archives
- `io.BytesIO` - for in-memory file creation
- Streamlit `st.download_button` - for downloading binary data
- All data already stored in `st.session_state`
- Plotly figures can be exported to HTML
- Pandas DataFrames can be exported to CSV

### Data Available Per Page

#### 1. Power Analysis Page
- ✅ `st.session_state.uploaded_df` - Original uploaded data
- ✅ `st.session_state.computed_stats` - Computed statistics (JSON)
- ✅ `st.session_state.generated_json` - Configuration JSON
- ✅ `st.session_state.selected_metrics` - Selected metrics
- ✅ Plots: Contour plots, single plots (Plotly figures)
- ⚠️ Filtering: Outlier removal applied during computation (need to track)

#### 2. Group Selection Page
- ✅ `st.session_state.uploaded_data_raw` - Original uploaded data
- ✅ `st.session_state.filtered_data` - Filtered data (if filters applied)
- ✅ `st.session_state.balanced_data` - Final balanced groups
- ✅ `st.session_state.balancing_config` - Balancing configuration (objectives, parameters, loss history)
- ✅ Balance plots (Plotly figures)
- ✅ Filtering parameters tracked in UI

#### 3. Rebalancer Page
- ✅ `st.session_state.rebalancer_uploaded_data` - Original uploaded data
- ✅ `st.session_state.rebalancer_filtered_data` - Filtered data (if filters applied)
- ✅ `st.session_state.rebalanced_data` - Final rebalanced groups
- ✅ `st.session_state.rebalancing_config` - Rebalancing configuration (parameters, loss history, middle/odd groups)
- ✅ Balance plots (Plotly figures)
- ✅ Filtering parameters tracked in UI

#### 4. Results Analysis Page
- ✅ `st.session_state.results_uploaded_data` - Original uploaded data
- ✅ `st.session_state.results_group_column` - Group column name
- ✅ `st.session_state.results_groups` - Group names
- ✅ Analysis results: Basic, CUPED, DiD (computed on-the-fly, need to save)
- ✅ Plots for each analysis type (Plotly figures)
- ⚠️ No filtering currently (but could be added)

## Implementation Plan

### Architecture

```
utils/
  artifact_saver.py
    - create_artifact_zip(page_name, session_state) -> BytesIO
    - generate_transformation_log(page_name, session_state) -> str
    - save_plot_to_html(fig, filename) -> bytes
    - format_config_as_json(config) -> str
```

### Artifact Structure

```
artifact_power_analysis_YYYYMMDD_HHMMSS.zip
├── data/
│   ├── 01_uploaded_data.csv
│   └── 02_computed_statistics.json
├── plots/
│   ├── contour_plot_power.html
│   ├── contour_plot_uplift.html
│   └── single_plots.html
├── config/
│   └── configuration.json
└── README.txt (transformation log)

artifact_group_selection_YYYYMMDD_HHMMSS.zip
├── data/
│   ├── 01_uploaded_data.csv
│   ├── 02_filtered_data.csv (if filters applied)
│   └── 03_balanced_data.csv
├── plots/
│   └── balance_report.html
├── config/
│   └── balancing_config.json
└── README.txt (transformation log)

artifact_rebalancer_YYYYMMDD_HHMMSS.zip
├── data/
│   ├── 01_uploaded_data.csv
│   ├── 02_filtered_data.csv (if filters applied)
│   └── 03_rebalanced_data.csv
├── plots/
│   └── balance_report.html
├── config/
│   └── rebalancing_config.json
└── README.txt (transformation log)

artifact_results_analysis_YYYYMMDD_HHMMSS.zip
├── data/
│   └── 01_uploaded_data.csv
├── plots/
│   ├── basic_analysis.html
│   ├── cuped_analysis.html
│   └── did_analysis.html
├── config/
│   └── analysis_config.json
└── README.txt (transformation log)
```

### Transformation Log Format

```
ARTIFACT SAVE LOG
=================
Generated: 2024-01-15 14:30:45
Page: Group Selection
App Version: 1.0

DATA TRANSFORMATIONS
--------------------
1. Data Upload
   - Source: uploaded_data.csv
   - Rows: 10,000
   - Columns: 8
   - Timestamp: 2024-01-15 14:25:00

2. Data Filtering
   - Method: Percentile-based outlier removal
   - Column: revenue
   - Lower percentile: 1.0%
   - Upper percentile: 99.0%
   - Rows removed: 200 (2.0%)
   - Metric removed: 5.2% of total revenue

3. Group Balancing
   - Mode: Advanced
   - Number of groups: 2
   - Group names: Control, Treatment
   - Proportions: 50.0%, 50.0%
   - Value columns: revenue, engagement_score
   - Stratification columns: region, device_type
   - Objectives:
     * revenue: p-value target = 0.95
     * engagement_score: p-value target = 0.90
     * region: max imbalance = 5.0%
     * device_type: max imbalance = 5.0%
   - Parameters:
     * Max iterations: 1000
     * Top K candidates: 20
     * Random candidates: 200
     * Gain threshold: 0.001
   - Final loss: 0.0234
   - Iterations: 456

FILES INCLUDED
--------------
- data/01_uploaded_data.csv: Original uploaded data
- data/02_filtered_data.csv: Data after filtering
- data/03_balanced_data.csv: Final balanced groups
- plots/balance_report.html: Interactive balance visualization
- config/balancing_config.json: Complete balancing configuration

REPRODUCIBILITY
--------------
To reproduce these results:
1. Load data/01_uploaded_data.csv
2. Apply filters as specified above
3. Run balancing with parameters from config/balancing_config.json
```

### Implementation Steps

1. **Create `utils/artifact_saver.py`**
   - Generic artifact creation function
   - Page-specific artifact generators
   - Transformation log generator
   - Plot HTML exporter

2. **Add "Save Artifact" button to each page**
   - Power Analysis: After computation
   - Group Selection: After balancing
   - Rebalancer: After rebalancing
   - Results Analysis: After analysis

3. **Track transformations**
   - Store filter parameters in session state
   - Store balancing/rebalancing parameters
   - Store analysis parameters

4. **Handle edge cases**
   - No data uploaded
   - No filters applied (skip filtered_data.csv)
   - No plots generated
   - Large files (memory management)

### Challenges & Solutions

1. **Challenge**: Plotly figures stored in session state may not be available
   - **Solution**: Regenerate plots if needed, or save plot configuration

2. **Challenge**: Large datasets may cause memory issues
   - **Solution**: Use streaming ZIP creation, or warn user for large files

3. **Challenge**: Transformation tracking requires capturing all UI interactions
   - **Solution**: Store transformation parameters in session state as they're applied

4. **Challenge**: Results Analysis computes on-the-fly
   - **Solution**: Save computed results to session state, or regenerate in artifact

### Benefits

1. **Reproducibility**: Complete snapshot of analysis
2. **Sharing**: Easy to share with colleagues
3. **Documentation**: Self-documenting with log file
4. **Backup**: Save work for later reference
5. **Audit Trail**: Track all transformations

### Estimated Implementation Time

- Core artifact saver utility: 2-3 hours
- Page-specific implementations: 1-2 hours per page (4 pages = 4-8 hours)
- Testing and edge cases: 2-3 hours
- **Total: 8-14 hours**

## Recommendation

✅ **PROCEED WITH IMPLEMENTATION**

This feature is highly feasible and provides significant value for reproducibility and collaboration. The implementation is straightforward using existing Python libraries and Streamlit capabilities.
