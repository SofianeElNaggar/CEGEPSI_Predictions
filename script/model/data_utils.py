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
    """Décompose les colonnes de config.DECOMPOSITION_COLS en modes VMD et les ajoute au DataFrame."""
    if config.DECOMPOSITION_METHOD != "VMD":
        return df

    if VMD is None:
        print("VMD introuvable. Skip.")
        return df

    print("\n" + "-"*40)
    print("Traitement VMD...")

    alpha = config.VMD_ALPHA
    tau   = config.VMD_TAU
    K     = config.VMD_K
    DC    = config.VMD_DC
    init  = config.VMD_INIT
    tol   = config.VMD_TOL

    df_new = df.copy()

    for col in config.DECOMPOSITION_COLS:
        if col not in df.columns:
            print(f"Colonne {col} absente pour VMD. Skip.")
            continue

        print(f" -> Décomposition VMD de '{col}' en {K} modes")
        f = df[col].values

        try:
            u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
            # u shape : (K, N) — un mode par ligne
            for k in range(K):
                df_new[f"{col}_mode{k+1}"] = u[k, :]
        except Exception as e:
            print(f"Erreur VMD sur {col}: {e}")

    print("-"*40 + "\n")
    return df_new

def apply_ceemdan(df, config):
    """Décompose les colonnes de config.DECOMPOSITION_COLS en IMFs CEEMDAN et les ajoute au DataFrame."""
    if config.DECOMPOSITION_METHOD != "CEEMDAN":
        return df

    if CEEMDAN is None:
        print("CEEMDAN introuvable. Skip.")
        return df

    print("\n" + "-"*40)
    print("Traitement CEEMDAN...")

    trials   = config.CEEMDAN_TRIALS
    epsilon  = config.CEEMDAN_EPSILON
    max_imfs = config.CEEMDAN_MAX_IMFS

    df_new = df.copy()
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)

    for col in config.DECOMPOSITION_COLS:
        if col not in df.columns:
            print(f"Colonne {col} absente pour CEEMDAN. Skip.")
            continue

        print(f" -> Décomposition CEEMDAN de '{col}'")
        S = df[col].values

        try:
            imfs = ceemdan(S)
            n_found = imfs.shape[0]

            if max_imfs is not None:
                # Tronque ou complète à max_imfs avec des zéros
                for k in range(max_imfs):
                    df_new[f"{col}_mode{k+1}"] = imfs[k, :] if k < n_found else np.zeros_like(S)
            else:
                for k in range(n_found):
                    df_new[f"{col}_mode{k+1}"] = imfs[k, :]

        except Exception as e:
            print(f"Erreur CEEMDAN sur {col}: {e}")

    print("-"*40 + "\n")
    return df_new

def apply_ssa(df, config):
    """Décompose les colonnes de config.DECOMPOSITION_COLS en composantes SSA et les ajoute au DataFrame."""
    if config.DECOMPOSITION_METHOD != "SSA":
        return df

    if SSA is None:
        print("SSA introuvable (pysdkit manquant). Skip.")
        return df

    print("\n" + "-"*40)
    print("Traitement SSA...")

    window_size = config.SSA_WINDOW
    df_new = df.copy()
    ssa = SSA(lags=window_size)

    for col in config.DECOMPOSITION_COLS:
        if col not in df.columns:
            print(f"Colonne {col} absente pour SSA. Skip.")
            continue

        print(f" -> Décomposition SSA de '{col}' (L={window_size})")
        S = df[col].values

        try:
            components = ssa.fit_transform(S)

            # Vérification de l'orientation : on s'attend à (K, n_samples)
            if components.shape[0] == len(S):
                components = components.T

            n_found = components.shape[0]
            for k in range(n_found):
                df_new[f"{col}_comp{k+1}"] = components[k, :]

        except Exception as e:
            print(f"Erreur SSA sur {col}: {e}")

    print("-"*40 + "\n")
    return df_new

def get_next_pdf_path(template_path_str):
    """
    Retourne un chemin PDF non conflictuel en incrémentant un suffixe numérique.
    Ex. : rapport_1.pdf, rapport_2.pdf, …
    """
    p = Path(template_path_str)
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    base = p.stem
    m = re.match(r"^(.*?)(?:_(\d+))?$", base)
    base_clean = m.group(1) if m else base
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
    return str(p.parent / f"{base_clean}_{next_n}.pdf")

def load_and_filter(parquet_path, start_date, depth_center, depth_tol, target_cols, use_depth_filter=True):
    """Charge le fichier parquet, filtre par date de début et par profondeur."""
    print("Chargement du fichier...", parquet_path)
    df = pd.read_parquet(parquet_path)
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
        if use_depth_filter:
            depth_mask = df['depth (m)'].notna() & (np.abs(df['depth (m)'] - depth_center) <= depth_tol)
            df = df[depth_mask]
            print(f"Filtrage profondeur: centre={depth_center} tol={depth_tol} -> {len(df)} lignes restantes")
        else:
            print("Prétraitement désactivé : toutes les profondeurs sont conservées.")
    else:
        print("Aucune colonne 'depth (m)' détectée; aucun filtrage par profondeur appliqué.")
    return df

def aggregate_daily(df, target_cols, agg_method='median'):
    """Agrège les mesures sub-journalières en valeurs journalières (médiane ou moyenne)."""
    df2 = df.copy()
    df2['date'] = df2['time (UTC)'].dt.floor('D')
    if agg_method == 'median':
        agg = df2.groupby('date')[target_cols].median()
    else:
        agg = df2.groupby('date')[target_cols].mean()
    agg.index = pd.to_datetime(agg.index)
    return agg

def reindex_and_impute(df, start, end, freq='D'):
    """Réindexe sur une grille temporelle continue (si freq est fourni) et interpole les valeurs manquantes."""
    if freq is not None:
        idx = pd.date_range(start=start, end=end - pd.Timedelta(seconds=1), freq=freq, tz='UTC')
        df = df.reindex(idx)
    return df.interpolate(method='time', limit_direction='both')

def create_sequences_multivar(values, seq_len, target_cols_count):
    """
    Crée des séquences glissantes (X, y) pour l'entraînement du modèle.
    X[i] = values[i : i+seq_len]  (toutes les features)
    y[i] = values[i+seq_len, :target_cols_count]  (cibles du pas suivant)
    """
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
    """Ajoute les encodages sinusoïdaux du jour de l'année (doy_sin, doy_cos)."""
    df = df_.copy()
    doy = df.index.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return df
