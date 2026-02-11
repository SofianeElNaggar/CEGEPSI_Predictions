
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import os
import sys
import torch

# Adjust path to import CNN model
# script/gestion dataset/input methode/ -> script/model/GRU/PINN/
# Relative: ../../../model/GRU/PINN
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

try:
    from script.model.GRU.PINN.CNN import CNNFeatureExtractor
except ImportError:
    # Fallback if the package structure is different
    model_dir = os.path.join(project_root, "script/model/GRU/PINN")
    sys.path.append(model_dir)
    from CNN import CNNFeatureExtractor

def analyze_temperature_vmd_cnn():
    # Configuration
    dataset_path = "/home/sofiane/Data_set_StL/CEGEPSI_Predictions/dataset/OMDS-CTD-meteogc-data.parquet"
    target_col = "temperature (°C)"
    depth_col = "depth (m)"
    depth_target = 1.0
    depth_tol = 0.1
    start_date = "2013-01-01"
    end_date = "2017-12-31"
    
    # VMD Parameters
    alpha = 2000       # Bandwidth constraint
    tau = 0.01         # Noise-tolerance
    K = 3              # Number of modes
    DC = 0             # No DC part imposed
    init = 0           # Initialize omegas uniformly
    tol = 1e-12        # Tolerance

    # Output paths
    output_vmd_img = "results/plots/vmd_cnn_vmd_output.png"
    output_cnn_img = "results/plots/vmd_cnn_cnn_output.png"
    os.makedirs("results/plots", exist_ok=True)
    
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
        print("Data still contains NaNs. Dropping...")
        daily_df = daily_df.dropna()
        
    values = daily_df.values
    
    if len(values) < 2 * K:
         print("Not enough data points for VMD.")
         return

    # Apply VMD
    print(f"Applying VMD (K={K})...")
    try:
        u, u_hat, omega = VMD(values, alpha, tau, K, DC, init, tol)
    except Exception as e:
        print(f"VMD failed: {e}")
        return

    # u shape is (K, N)
    print(f"VMD Output Dimensions (u): {u.shape}")

    # Plot VMD
    print("Generating VMD plot...")
    plt.figure(figsize=(12, 12))
    plt.subplot(K + 1, 1, 1)
    plt.plot(daily_df.index, values, color='black', label='Original Temperature')
    plt.title(f"Temperature & VMD Decomposition")
    plt.legend()
    plt.grid(True, alpha=0.3)

    for i in range(K):
        plt.subplot(K + 1, 1, i + 2)
        plt.plot(daily_df.index, u[i,:], label=f'Mode {i+1}')
        plt.ylabel(f'IMF {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_vmd_img)
    print(f"VMD Graph saved to {output_vmd_img}")
    plt.close()

    # Prepare for CNN
    # CNN expects (batch, seq_len, n_features)
    # u is (K, N) -> (n_features, seq_len)
    # Transpose to (N, K) -> (seq_len, n_features)
    # Add batch dim -> (1, seq_len, n_features)
    
    input_tensor = torch.tensor(u.T, dtype=torch.float32).unsqueeze(0)
    print(f"CNN Input Tensor Dimensions: {input_tensor.shape}")
    
    # Initialize CNN
    out_channels = 64
    cnn_model = CNNFeatureExtractor(n_features=K, out_channels=out_channels, padding='same')
    
    # Run CNN
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(input_tensor)
        
    # Output is (batch, seq_len, out_channels)
    print(f"CNN Output Dimensions: {output.shape}")
    
    # Plot CNN Output
    # We will plot it as a heatmap: (out_channels, seq_len)
    output_np = output.squeeze(0).T.numpy() # (out_channels, seq_len)
    
    print("Generating CNN output plot...")
    plt.figure(figsize=(15, 6))
    plt.imshow(output_np, aspect='auto', cmap='viridis', origin='lower', extent=[0, output_np.shape[1], 0, output_np.shape[0]])
    plt.colorbar(label='Activation')
    plt.title(f"CNN Output Features (Channels={out_channels})")
    plt.xlabel("Time Steps")
    plt.ylabel("Channels")
    plt.savefig(output_cnn_img)
    print(f"CNN Graph saved to {output_cnn_img}")
    plt.close()

if __name__ == "__main__":
    analyze_temperature_vmd_cnn()
