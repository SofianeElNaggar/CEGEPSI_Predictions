
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
import os

def analyze_temperature_ceemdan():
    # Configuration
    dataset_path = "/home/sofiane/Data_set_StL/CEGEPSI_Predictions/dataset/OMDS-CTD-meteogc-data.parquet"
    target_col = "temperature (°C)"
    depth_col = "depth (m)"
    depth_target = 1.0
    depth_tol = 0.1
    start_date = "2000-01-01"
    end_date = "2020-12-31"
    
    # CEEMDAN Parameters
    trials = 100       # Number of realizations
    epsilon = 0.1      # Noise level
    # Note: CEEMDAN finds the number of modes automatically. 
    # We will plot whatever it finds, or limit if too many.

    output_img_base = "results/plots/CEEMDAN/temperature_ceemdan_analysis"
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure Date/Time is datetime and UTC
    if 'time (UTC)' in df.columns:
        df['time (UTC)'] = pd.to_datetime(df['time (UTC)'], utc=True)
        # Sort by time just in case
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

    # Filter by Date (2010 to 2020)
    print(f"Filtering by date: {start_date} to {end_date}")
    # Localize start/end dates to UTC for comparison
    ts_start = pd.Timestamp(start_date).tz_localize('UTC')
    ts_end = pd.Timestamp(end_date).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) # End of the day
    
    df = df[(df['time (UTC)'] >= ts_start) & (df['time (UTC)'] <= ts_end)]

    if df.empty:
        print("No data found after filtering.")
        return

    # Group by Day (Daily Mean)
    print("Calculating daily means...")
    # Create a 'date' column for grouping
    df['date'] = df['time (UTC)'].dt.floor('D')
    daily_df = df.groupby('date')[target_col].mean()

    # Reindex to ensure continuous daily time series
    full_idx = pd.date_range(start=ts_start.floor('D'), end=ts_end.floor('D'), freq='D')
    daily_df = daily_df.reindex(full_idx)
    
    # Handle NaNs (Interpolation)
    # CEEMDAN requires arrays without NaNs.
    if daily_df.isna().sum() > 0:
        print(f"Interpolating {daily_df.isna().sum()} missing daily values...")
        daily_df = daily_df.interpolate(method='time', limit_direction='both')
        # Fill any remaining NaNs at edges with bfill/ffill if necessary
        daily_df = daily_df.bfill().ffill()
    
    if daily_df.isna().any():
        print("Data still contains NaNs after interpolation. Dropping remaining NaNs.")
        daily_df = daily_df.dropna()
        
    values = daily_df.values
    
    # Apply CEEMDAN
    print(f"Applying CEEMDAN (trials={trials}, epsilon={epsilon})...")
    try:
        ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
        # imfs shape: (n_imfs, n_samples)
        imfs = ceemdan(values)
    except Exception as e:
        print(f"CEEMDAN failed: {e}")
        return

    n_imfs = imfs.shape[0]
    print(f"CEEMDAN found {n_imfs} IMFs.")
    
    output_img = f"{output_img_base}_trials{trials}_eps{epsilon}_IMFs{n_imfs}.png"

    # Plotting
    print("Generating plot...")
    # Adjust figure height based on number of IMFs
    plt.figure(figsize=(12, 2 * (n_imfs + 1)))
    
    # Original Signal
    plt.subplot(n_imfs + 1, 1, 1)
    plt.plot(daily_df.index, values, color='black', label='Original Temperature')
    plt.title(f"Temperature (Depth {depth_target}m) & CEEMDAN Decomposition")
    plt.ylabel('Temp (°C)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # CEEMDAN Modes
    for i in range(n_imfs):
        plt.subplot(n_imfs + 1, 1, i + 2)
        plt.plot(daily_df.index, imfs[i,:], label=f'IMF {i+1}')
        plt.ylabel(f'IMF {i+1}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")

if __name__ == "__main__":
    analyze_temperature_ceemdan()