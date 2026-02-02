
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import re

def analyze_temperature_wavelet():
    # Configuration
    dataset_path = "/home/sofiane/Data_set_StL/CEGEPSI_Predictions/dataset/OMDS-CTD-meteogc-data.parquet"
    target_col = "temperature (°C)"
    depth_col = "depth (m)"
    depth_target = 1.0
    depth_tol = 0.1
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    
    level = 5             # Decomposition level

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
    if daily_df.isna().sum() > 0:
        print(f"Interpolating {daily_df.isna().sum()} missing daily values...")
        daily_df = daily_df.interpolate(method='time', limit_direction='both')
        # Fill any remaining NaNs at edges with bfill/ffill if necessary
        daily_df = daily_df.bfill().ffill()
    
    if daily_df.isna().any():
        print("Data still contains NaNs after interpolation. Dropping remaining NaNs.")
        daily_df = daily_df.dropna()
        
    values = daily_df.values.copy()
    
    if len(values) < 20: 
         print("Not enough data points for Wavelet analysis.")
         return

    # Get all available discrete wavelets
    wavelet_list = pywt.wavelist(kind='discrete')
    print(f"Found {len(wavelet_list)} wavelets to test.")
    
    # helper for family name extraction
    def get_family(w_name):
        # usually just remove numbers
        match = re.match(r"([a-zA-Z]+)", w_name)
        if match:
            return match.group(1)
        return "other"

    for wavelet_name in wavelet_list:
        family = get_family(wavelet_name)
        output_dir = f"results/plots/Ondelette/{family}"
        output_img = f"{output_dir}/temperature_wavelet_analysis_{wavelet_name}_level{level}.png"
        
        # skip if already exists? maybe not, overwrite for now.
        
        print(f"Processing {wavelet_name} (Family: {family})...")
        
        try:
            # Apply Wavelet MRA (Multi-Resolution Analysis)
            # pywt.mra requires PyWavelets >= 1.1.0
            try:
                # mra returns [Approximation, Detail_Level, Detail_Level-1, ..., Detail_1]
                mra_coeffs = pywt.mra(values, wavelet_name, level=level, transform='dwt')
            except AttributeError:
                # Fallback
                coeffs = pywt.wavedec(values, wavelet_name, level=level)
                mra_coeffs = []
                c_zeroed = [np.zeros_like(c) for c in coeffs]
                c_zeroed[0] = coeffs[0]
                mra_coeffs.append(pywt.waverec(c_zeroed, wavelet_name))
                for i in range(1, len(coeffs)):
                    c_zeroed = [np.zeros_like(c) for c in coeffs]
                    c_zeroed[i] = coeffs[i]
                    mra_coeffs.append(pywt.waverec(c_zeroed, wavelet_name))
                    
                # Ensure lengths match
                for i in range(len(mra_coeffs)):
                    if len(mra_coeffs[i]) > len(values):
                        mra_coeffs[i] = mra_coeffs[i][:len(values)]
            except Exception as e:
                # Some wavelets might fail if they are valid but constraints (like signal length vs filter length) aren't met
                print(f"  Skipping {wavelet_name}: {e}")
                continue

            num_components = len(mra_coeffs)
            
            # Plotting
            plt.figure(figsize=(12, 3 * (num_components + 1)))
            
            # Original Signal
            plt.subplot(num_components + 1, 1, 1)
            plt.plot(daily_df.index, values, color='black', label='Original Temperature')
            plt.title(f"Temperature & Wavelet MRA ({wavelet_name})")
            plt.ylabel('Temp (°C)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)

            # Components
            plt.subplot(num_components + 1, 1, 2)
            plt.plot(daily_df.index, mra_coeffs[0], color='red', label=f'Approximation (A{level})')
            plt.ylabel(f'A{level}')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            for i in range(1, num_components):
                d_level = level - i + 1 
                plt.subplot(num_components + 1, 1, i + 2)
                plt.plot(daily_df.index, mra_coeffs[i], label=f'Detail (D{d_level})')
                plt.ylabel(f'D{d_level}')
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)

            plt.xlabel('Date')
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            plt.savefig(output_img)
            plt.close() # Close memory
            
        except Exception as e:
            print(f"  Error processing {wavelet_name}: {e}")
            plt.close() # Ensure close even on error

if __name__ == "__main__":
    analyze_temperature_wavelet()
