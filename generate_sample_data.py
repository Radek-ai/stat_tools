"""
Generate sample CSV data for testing Power Analysis tool
Creates realistic customer statistics with multiple numeric columns, outliers, and some NaN values
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers/rows
n_customers = 10000

print(f"Generating sample data for {n_customers} customers...")

# Generate customer IDs
customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]

# Generate base data with realistic distributions

# 1. Revenue (right-skewed, typical for revenue data)
# Base revenue with some high-value outliers
base_revenue = np.random.lognormal(mean=3.5, sigma=1.2, size=n_customers)
# Add some extreme outliers (top 1%)
outlier_indices = np.random.choice(n_customers, size=int(n_customers * 0.01), replace=False)
base_revenue[outlier_indices] *= np.random.uniform(5, 20, size=len(outlier_indices))
revenue = np.round(base_revenue, 2)

# 2. Clicks (Poisson-like distribution)
clicks = np.random.negative_binomial(n=10, p=0.3, size=n_customers)
clicks = np.maximum(clicks, 0).astype(float)  # Ensure non-negative and convert to float

# 3. Conversions (lower than clicks, binomial-like)
conversion_rate = np.random.beta(5, 95, size=n_customers)  # Varying conversion rates
conversions = np.random.binomial(clicks.astype(int), conversion_rate).astype(float)

# 4. Session Duration (minutes, normal-like with some outliers)
session_duration = np.random.normal(loc=15, scale=8, size=n_customers)
session_duration = np.maximum(session_duration, 0)  # No negative durations
# Add some long sessions
long_session_indices = np.random.choice(n_customers, size=int(n_customers * 0.02), replace=False)
session_duration[long_session_indices] += np.random.uniform(30, 120, size=len(long_session_indices))
session_duration = np.round(session_duration, 1)

# 5. Page Views (right-skewed)
page_views = np.random.lognormal(mean=2.5, sigma=0.8, size=n_customers)
page_views = np.maximum(page_views, 1).astype(float)

# 6. Bounce Rate (0-1, beta distribution)
bounce_rate = np.random.beta(2, 3, size=n_customers)
bounce_rate = np.round(bounce_rate, 3).astype(float)

# 7. Time on Site (seconds, correlated with session duration)
time_on_site = session_duration * 60 + np.random.normal(0, 300, size=n_customers)
time_on_site = np.maximum(time_on_site, 0)
time_on_site = np.round(time_on_site, 0).astype(float)

# Ensure all numeric arrays are float before adding NaNs
revenue = revenue.astype(float)
session_duration = session_duration.astype(float)

# Add some NaN values (5% missing data randomly distributed)
nan_percentage = 0.05
for col_data in [revenue, clicks, conversions, session_duration, page_views, bounce_rate, time_on_site]:
    nan_indices = np.random.choice(n_customers, size=int(n_customers * nan_percentage), replace=False)
    col_data[nan_indices] = np.nan

# Create DataFrame
df = pd.DataFrame({
    'customer_id': customer_ids,
    'revenue': revenue,
    'clicks': clicks,
    'conversions': conversions,
    'session_duration': session_duration,
    'page_views': page_views,
    'bounce_rate': bounce_rate,
    'time_on_site': time_on_site,
})

# Add some categorical columns for context (optional, not used in power analysis)
df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_customers)
df['device_type'] = np.random.choice(['Desktop', 'Mobile', 'Tablet'], size=n_customers, p=[0.5, 0.4, 0.1])

# Reorder columns
column_order = ['customer_id', 'revenue', 'clicks', 'conversions', 'session_duration', 
                'page_views', 'bounce_rate', 'time_on_site', 'region', 'device_type']
df = df[column_order]

# Save to CSV
output_filename = 'sample_customer_data.csv'
df.to_csv(output_filename, index=False)

print(f"\n✅ Sample data generated successfully!")
print(f"📁 Saved to: {output_filename}")
print(f"\n📊 Data Summary:")
print(f"   - Total rows: {len(df):,}")
print(f"   - Total columns: {len(df.columns)}")
print(f"\n📈 Numeric Columns (for periods):")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    non_null = df[col].notna().sum()
    pct_missing = (1 - non_null / len(df)) * 100
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"   - {col:20s}: mean={mean_val:10.2f}, std={std_val:10.2f}, missing={pct_missing:5.1f}%")

print(f"\n💡 Usage:")
print(f"   1. Upload '{output_filename}' in the Scenarios Design tab")
print(f"   2. Create scenarios with different outlier filtering (e.g., Percentile 1-99)")
print(f"   3. Create periods by selecting columns like 'revenue', 'clicks', 'conversions'")
print(f"   4. Compute statistics and generate JSON")
print(f"   5. Use the JSON in Power Analysis tab")

