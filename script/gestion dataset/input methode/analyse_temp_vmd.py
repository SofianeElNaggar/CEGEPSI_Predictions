
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import os

def analyze_temperature_vmd():
    # Configuration
    dataset_path = "/home/sofiane/Data_set_StL/CEGEPSI_Predictions/dataset/OMDS-CTD-meteogc-data.parquet"
    target_col = "temperature (°C)"
    depth_col = "depth (m)"
    depth_target = 1.0
    depth_tol = 0.1
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    
    # VMD Parameters
    alpha = 2000       # Bandwidth constraint
    tau = 0.01         # Noise-tolerance (no strict fidelity enforcement)
    K = 8              # Number of modes
    DC = 0             # No DC part imposed
    init = 0           # Initialize omegas uniformly
    tol = 1e-12         # Tolerance

    output_img = "results/plots/temperature_vmd_analysis_tau"+str(tau)+"_alpha"+str(alpha)+"_K"+str(K)+"_DC"+str(DC)+"_init"+str(init)+"_tol"+str(tol)+".png"
    
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

    # Reindex to ensure continuous daily time series (important for VMD)
    full_idx = pd.date_range(start=ts_start.floor('D'), end=ts_end.floor('D'), freq='D')
    daily_df = daily_df.reindex(full_idx)
    
    # Handle NaNs (Interpolation)
    # VMD requires arrays without NaNs. We interpolate linearly.
    if daily_df.isna().sum() > 0:
        print(f"Interpolating {daily_df.isna().sum()} missing daily values...")
        daily_df = daily_df.interpolate(method='time', limit_direction='both')
        # Fill any remaining NaNs at edges with bfill/ffill if necessary
        daily_df = daily_df.bfill().ffill()
    
    if daily_df.isna().any():
        print("Data still contains NaNs after interpolation. Dropping remaining NaNs.")
        daily_df = daily_df.dropna()
        
    values = daily_df.values
    
    if len(values) < 2 * K:
         print("Not enough data points for VMD.")
         return

    # Apply VMD
    print(f"Applying VMD (K={K})...")
    # VMD(f, alpha, tau, K, DC, init, tol)
    try:
        u, u_hat, omega = VMD(values, alpha, tau, K, DC, init, tol)
    except Exception as e:
        print(f"VMD failed: {e}")
        return

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 12))
    
    # Original Signal
    plt.subplot(K + 1, 1, 1)
    plt.plot(daily_df.index, values, color='black', label='Original Temperature')
    plt.title(f"Temperature (Depth {depth_target}m) & VMD Decomposition")
    plt.ylabel('Temp (°C)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # VMD Modes
    for i in range(K):
        plt.subplot(K + 1, 1, i + 2)
        plt.plot(daily_df.index, u[i,:], label=f'Mode {i+1}')
        plt.ylabel(f'IMF {i+1}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    
    
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")
    
    # Optional: Show plot if running in an environment that supports it (often not in headless agents)
    # plt.show()

if __name__ == "__main__":
    analyze_temperature_vmd()
