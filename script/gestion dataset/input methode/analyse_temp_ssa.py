
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pysdkit import SSA

def analyze_temperature_ssa():
    # Configuration
    dataset_path = "/home/sofiane/Data_set_StL/CEGEPSI_Predictions/dataset/OMDS-CTD-meteogc-data.parquet"
    target_col = "temperature (°C)"
    depth_col = "depth (m)"
    depth_target = 1.0
    depth_tol = 0.1
    start_date = "2000-01-01"
    end_date = "2020-12-31"
    
    # SSA Parameters
    # window_size (M/L): window length
    window_size = 91
    
    output_img_base = "results/plots/SSA/temperature_ssa_analysis"
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure Date/Time is datetime and UTC
    if 'time (UTC)' in df.columns:
        df['time (UTC)'] = pd.to_datetime(df['time (UTC)'], utc=True)
        df = df.sort_values('time (UTC)')
    else:
        print("Column 'time (UTC)' not found.")
        return

    # Filter by Depth
    print(f"Filtering by depth: {depth_target}m +/- {depth_tol}m")
    if depth_col in df.columns:
        df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
        mask_depth = (df[depth_col] >= (depth_target - depth_tol)) & (df[depth_col] <= (depth_target + depth_tol))
        df = df[mask_depth]
    else:
        print(f"Column '{depth_col}' not found. Skipping depth filter.")

    # Filter by Date
    print(f"Filtering by date: {start_date} to {end_date}")
    ts_start = pd.Timestamp(start_date).tz_localize('UTC')
    ts_end = pd.Timestamp(end_date).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    
    df = df[(df['time (UTC)'] >= ts_start) & (df['time (UTC)'] <= ts_end)]

    if df.empty:
        print("No data found after filtering.")
        return

    # Group by Day (Daily Mean)
    print("Calculating daily means...")
    df['date'] = df['time (UTC)'].dt.floor('D')
    daily_df = df.groupby('date')[target_col].mean()

    # Reindex
    full_idx = pd.date_range(start=ts_start.floor('D'), end=ts_end.floor('D'), freq='D')
    daily_df = daily_df.reindex(full_idx)
    
    # Interpolation
    if daily_df.isna().sum() > 0:
        print(f"Interpolating {daily_df.isna().sum()} missing daily values...")
        daily_df = daily_df.interpolate(method='time', limit_direction='both')
        daily_df = daily_df.bfill().ffill()
    
    if daily_df.isna().any():
        print("Data still contains NaNs after interpolation. Dropping remaining NaNs.")
        daily_df = daily_df.dropna()
        
    values = daily_df.values
    
    # Apply SSA
    print(f"Applying SSA (window_size/lags={window_size})...")
    try:
        # PySDKit SSA signature: SSA(self, K=3, mode='covar', lags=None, ...)
        # K seems to be number of components to keep/return?
        # lags is the window size (embedding dimension)
        
        # We want to inspect multiple components, let's ask for e.g. 10 or 20
        # If we want all, we might need to set K=lags?
        # Let's try setting K to a large number or the window size if sensible, 
        # but for plotting we only need top N.
        
        n_to_keep = 20 # Extract top 20 components
        ssa = SSA(lags=window_size, K=n_to_keep)
        
        # Check API: usually fit_transform
        components = ssa.fit_transform(values)
        
        # Check dimensions
        if components.shape[0] == len(values):
            components = components.T
            
    except Exception as e:
        print(f"SSA failed: {e}")
        return

    n_components = components.shape[0]
    print(f"SSA extracted {n_components} components.")
    
    # Plotting first few components mainly (trend + seasonalities)
    max_plot = min(10, n_components)
    
    output_img = f"{output_img_base}_window{window_size}_K{max_plot}.png"

    print("Generating plot...")
    plt.figure(figsize=(12, 2 * (max_plot + 1)))
    
    plt.subplot(max_plot + 1, 1, 1)
    plt.plot(daily_df.index, values, color='black', label='Original Temperature')
    plt.title(f"Temperature (Depth {depth_target}m) & SSA Decomposition")
    plt.ylabel('Temp (°C)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    for i in range(max_plot):
        plt.subplot(max_plot + 1, 1, i + 2)
        plt.plot(daily_df.index, components[i,:], label=f'Component {i+1}')
        plt.ylabel(f'Comp {i+1}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")

if __name__ == "__main__":
    analyze_temperature_ssa()