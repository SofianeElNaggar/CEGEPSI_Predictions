
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pysdkit import TVF_EMD

def analyze_temperature_tvf_emd():
    # Configuration
    dataset_path = "/home/sofiane/Data_set_StL/CEGEPSI_Predictions/dataset/OMDS-CTD-meteogc-data.parquet"
    target_col = "temperature (°C)"
    depth_col = "depth (m)"
    depth_target = 1.0
    depth_tol = 0.1
    start_date = "2000-01-01"
    end_date = "2020-12-31"
    
    # TVF_EMD Parameters
    # Adjust based on defaults or needs
    # TVF_EMD(signal, bandwidth_threshold=0.1, B_spline_order=26) typically
    bandwidth_threshold = 0.1
    b_spline_order = 26
    
    max_imfs = None # None = automatic

    output_img_base = "results/plots/TVF_EMD/temperature_tvf_emd_analysis"
    
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
    
    # Apply TVF_EMD
    print(f"Applying TVF_EMD (bandwidth_threshold={bandwidth_threshold}, b_spline_order={b_spline_order})...")
    try:
        tvf_emd = TVF_EMD()
        # Check API: usually fit_transform(signal) or just call(signal)
        # Based on PySDKit general pattern:
        imfs_found = tvf_emd.fit_transform(values)
        
        # Ensure shape (n_imfs, n_samples)
        # PySDKit often returns (n_samples, n_imfs) -> transpose if needed
        # Or (n_imfs, n_samples). Let's check dimensions.
        if imfs_found.shape[0] == len(values):
             imfs_found = imfs_found.T
             
    except Exception as e:
        print(f"TVF_EMD failed: {e}")
        return

    n_found = imfs_found.shape[0]
    print(f"TVF_EMD originally found {n_found} IMFs.")
    
    if max_imfs is not None:
        final_imfs_count = max_imfs
        imfs = np.zeros((final_imfs_count, imfs_found.shape[1]))
        if n_found >= final_imfs_count:
            imfs = imfs_found[:final_imfs_count, :]
        else:
            imfs[:n_found, :] = imfs_found
            print(f"Warning: TVF_EMD found fewer IMFs ({n_found}) than requested ({max_imfs}). Padding with zeros.")
    else:
        imfs = imfs_found
        final_imfs_count = n_found
    
    output_img = f"{output_img_base}_bw{bandwidth_threshold}_order{b_spline_order}_IMFs{final_imfs_count}.png"

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 2 * (final_imfs_count + 1)))
    
    plt.subplot(final_imfs_count + 1, 1, 1)
    plt.plot(daily_df.index, values, color='black', label='Original Temperature')
    plt.title(f"Temperature (Depth {depth_target}m) & TVF_EMD Decomposition")
    plt.ylabel('Temp (°C)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    for i in range(final_imfs_count):
        plt.subplot(final_imfs_count + 1, 1, i + 2)
        is_padding = np.all(imfs[i,:] == 0)
        label = f'IMF {i+1}' + (' (Padding)' if is_padding else '')
        plt.plot(daily_df.index, imfs[i,:], label=label)
        plt.ylabel(f'IMF {i+1}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")

if __name__ == "__main__":
    analyze_temperature_tvf_emd()