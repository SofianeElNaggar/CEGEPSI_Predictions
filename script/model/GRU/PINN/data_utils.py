# data_utils.py
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from vmdpy import VMD
from pysdkit import SSA
from PyEMD import CEEMDAN

from sklearn.preprocessing import StandardScaler

def apply_vmd(df, config):
    if not config.VMD_ENABLED:
        return df
    
    if VMD is None:
        print("VMD introuvable. Skip.")
        return df

    print("\n" + "-"*40)
    print("Traitement VMD...")
    
    # Parametres VMD
    alpha = config.VMD_ALPHA
    tau = config.VMD_TAU
    K = config.VMD_K
    DC = config.VMD_DC
    init = config.VMD_INIT
    tol = config.VMD_TOL
    
    df_new = df.copy()
    
    for col in config.VMD_COLS:
        if col not in df.columns:
            print(f"Colonne {col} absente pour VMD. Skip.")
            continue
            
        print(f" -> Decomposition VMD de '{col}' en {K} modes")
        f = df[col].values
        
        # VMD requires 1D array, even length preferred but not strictly required by python impl often?
        # vmdpy: u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
        try:
            u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
            
            # u shape: (K, N)
            for k in range(K):
                mode_name = f"{col}_mode{k+1}"
                df_new[mode_name] = u[k, :]
        except Exception as e:
            print(f"Erreur VMD sur {col}: {e}")
            
    print("-"*40 + "\n")
    return df_new

def apply_ceemdan(df, config):
    if not config.CEEMDAN_ENABLED:
        return df
    
    if CEEMDAN is None:
        print("CEEMDAN introuvable. Skip.")
        return df

    print("\n" + "-"*40)
    print("Traitement CEEMDAN...")
    
    # Parametres CEEMDAN
    trials = getattr(config, 'CEEMDAN_TRIALS', 100)
    epsilon = getattr(config, 'CEEMDAN_EPSILON', 0.2)
    max_imfs = getattr(config, 'CEEMDAN_MAX_IMFS', None)
    
    df_new = df.copy()
    
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
    # Pour éviter trop de logs si PyEMD est verbeux
    # ceemdan.verbose = False 

    for col in config.CEEMDAN_COLS:
        if col not in df.columns:
            print(f"Colonne {col} absente pour CEEMDAN. Skip.")
            continue
            
        print(f" -> Decomposition CEEMDAN de '{col}'")
        S = df[col].values
        
        # CEEMDAN: IMFs = ceemdan(S) => shape (n_imfs, n_samples)
        try:
            # S doit être un tableau numpy 1D
            imfs = ceemdan(S)
            
            n_found = imfs.shape[0]
            
            if max_imfs is not None:
                # Force exact number (pad or truncate)
                for k in range(max_imfs):
                    mode_name = f"{col}_mode{k+1}"
                    if k < n_found:
                        df_new[mode_name] = imfs[k, :]
                    else:
                        df_new[mode_name] = np.zeros_like(S)
            else:
                # Keep all found IMFs
                for k in range(n_found):
                     mode_name = f"{col}_mode{k+1}"
                     df_new[mode_name] = imfs[k, :]
                    
        except Exception as e:
            print(f"Erreur CEEMDAN sur {col}: {e}")
            
    print("-"*40 + "\n")
    return df_new

def apply_ssa(df, config):
    if not config.SSA_ENABLED:
        return df
    
    if SSA is None:
        print("SSA introuvable (pysdkit missing). Skip.")
        return df

    print("\n" + "-"*40)
    print("Traitement SSA...")
    
    # Parametres SSA
    window_size = getattr(config, 'SSA_WINDOW', 365)
    
    df_new = df.copy()
    
    # Initialize SSA
    # SSA(lags=window_size)
    ssa = SSA(lags=window_size)

    for col in config.SSA_COLS:
        if col not in df.columns:
            print(f"Colonne {col} absente pour SSA. Skip.")
            continue
            
        print(f" -> Decomposition SSA de '{col}' (L={window_size})")
        S = df[col].values
        
        try:
            # SSA fit_transform
            # components shape: (K, n_samples) ideally, if not check transpose
            components = ssa.fit_transform(S)
            
            # Verify shape
            if components.shape[0] == len(S) and components.shape[1] == n_components:
                # Transpose needed to get (K, n_samples)
                components = components.T
            
            n_found = components.shape[0]
            
            for k in range(n_found):
                mode_name = f"{col}_comp{k+1}"
                df_new[mode_name] = components[k, :]
                    
        except Exception as e:
            print(f"Erreur SSA sur {col}: {e}")
            
    print("-"*40 + "\n")
    return df_new

def get_next_pdf_path(template_path_str):
    p = Path(template_path_str)
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    base = p.stem
    m = re.match(r"^(.*?)(?:_(\d+))?$", base)
    if m:
        base_clean = m.group(1)
    else:
        base_clean = base
    pattern = re.compile(rf"^{re.escape(base_clean)}_(\d+)\.pdf$", re.IGNORECASE)
    max_n = 0
    try:
        for f in p.parent.iterdir():
            if f.is_file() and f.suffix.lower() == ".pdf":
                mm = pattern.match(f.name)
                if mm:
                    try:
                        n = int(mm.group(1))
                        if n > max_n:
                            max_n = n
                    except Exception:
                        pass
    except FileNotFoundError:
        max_n = 0
    next_n = max_n + 1
    next_name = f"{base_clean}_{next_n}.pdf"
    return str(p.parent / next_name)

def load_and_filter(parquet_path, start_date, depth_center, depth_tol, target_cols):
    print("Chargement du fichier...", parquet_path)
    df = pd.read_parquet(parquet_path)
    # Normalisation des noms de colonnes utilisés dans ce projet
    # On suppose que le fichier a des colonnes 'time (UTC)' et 'depth (m)'
    df['time (UTC)'] = pd.to_datetime(df['time (UTC)'], errors='coerce', utc=True)
    if 'depth (m)' in df.columns:
        df['depth (m)'] = pd.to_numeric(df['depth (m)'], errors='coerce')
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    start_ts = pd.to_datetime(start_date, errors='coerce')
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize('UTC')
    else:
        start_ts = start_ts.tz_convert('UTC')
    df = df[df['time (UTC)'] >= start_ts]

    if 'depth (m)' in df.columns:
        depth_mask = df['depth (m)'].notna() & (np.abs(df['depth (m)'] - depth_center) <= depth_tol)
        df = df[depth_mask]
        print(f"Filtrage profondeur: centre={depth_center} tol={depth_tol} -> {len(df)} lignes restantes")
    else:
        print("Aucune colonne 'depth (m)' détectée; aucune sélection par profondeur appliquée.")
    return df

def aggregate_daily(df, target_cols, agg_method='median'):
    df2 = df.copy()
    df2['date'] = df2['time (UTC)'].dt.floor('D')
    if agg_method == 'median':
        agg = df2.groupby('date')[target_cols].median()
    else:
        agg = df2.groupby('date')[target_cols].mean()
    agg.index = pd.to_datetime(agg.index)
    return agg

def reindex_and_impute(daily_df, start, end):
    idx = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq='D', tz='UTC')
    daily = daily_df.reindex(idx)
    daily_inter = daily.interpolate(method='time', limit_direction='both')
    return daily_inter

def create_sequences_multivar(values, seq_len, target_cols_count):
    n = values.shape[0]
    n_features = values.shape[1]
    n_samples = n - seq_len
    if n_samples <= 0:
        return np.empty((0, seq_len, n_features)), np.empty((0, target_cols_count))
    X = np.zeros((n_samples, seq_len, n_features), dtype=float)
    y = np.zeros((n_samples, target_cols_count), dtype=float)
    for i in range(n_samples):
        X[i] = values[i:i + seq_len]
        y[i] = values[i + seq_len, :target_cols_count]
    return X, y

def add_time_features(df_):
    df = df_.copy()
    doy = df.index.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return df
