# -*- coding: utf-8 -*-
"""
Pipeline LSTM multivarié — version PyTorch
- remplace TensorFlow/Keras par PyTorch
- conserve le pipeline original (chargement, agrégation, imputation, features temporelles,
  standardisation, OLS post-traitement, export PDF)
- supporte forecast récursif ou walk-forward
- early stopping manuel + restauration des meilleurs poids
- vise la reproductibilité (seeds, variables d'environnement, cudnn flags)
"""

import os
# --- deterministic env-vars (avant imports ML) ---
os.environ.setdefault('PYTHONHASHSEED', '42')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import random
from pathlib import Path
from datetime import datetime
import re
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sklearn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Reproductibilité ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# pour GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# cudnn determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- PARAMÈTRES UTILISATEUR ----------------------
PARQUET_PATH = "../../../../dataset/OMDS-CTD-meteogc-data.parquet"  # chemin vers .parquet

# profondeur ciblée (ex 1.0m +/- tol)
DEPTH_CENTER = 1.0
DEPTH_TOLERANCE = 0.1

# Agg: 'median' ou 'mean'
AGG_METHOD = "mean"

# template pour les PDFs de sortie (on injectera le nom de la combinaison)
OUTPUT_PDF_TEMPLATE = f"../../../results/prediction/LSTM/pytorch/{AGG_METHOD}/LSTM_predictions_bis_{AGG_METHOD}.pdf"

# Période d'utilisation des données
START_DATE = "2000-01-01"
TRAIN_END = "2020-01-01"
TEST_END = "2025-01-01"

# LSTM / entrainement
SEQUENCE_LENGTH = 30
N_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8

# Fraction du jeu d'entraînement à réserver comme validation (contiguë)
VALIDATION_FRAC = 0.05

RECURSIVE_FORECAST = True

# --- liste complète des targets (la combinaison utilisée telle quelle) ---
ALL_TARGETS = [
    "temperature (°C)",
    #"chlorophyll (mg m-3)",
    #"turbidity (NTU)",
    #"pH",
    #"salinity (PSS-78)",
    #"potential_density (kg m-3)",
    #"dissolved_oxygen (ml l-1)",
]

# --------------------------------------------------------------------

# ---------------------- Fonctions utilitaires ----------------------

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
    # colonnes nommées avec espace comme dans ton exemple
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

# ---------------- PyTorch dataset ----------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        # X: (n_samples, seq_len, n_features)
        # y: (n_samples, n_outputs)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- PyTorch model ----------------
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=128, hidden2=64, n_outputs=1, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, n_outputs)
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm1(x)            # out: (batch, seq_len, hidden1)
        out, _ = self.lstm2(out)          # out: (batch, seq_len, hidden2)
        last = out[:, -1, :]              # (batch, hidden2)
        last = self.dropout(last)
        out = self.fc(last)               # (batch, n_outputs)
        return out

# ---------------- weighted MSE loss util (PyTorch) ----------------
def weighted_mse_loss(pred, target, weights):
    # pred/target: (batch, n_outputs)
    # weights: numpy array length n_outputs -> torch tensor
    w = torch.tensor(weights, dtype=pred.dtype, device=pred.device).reshape(1, -1)
    se = (pred - target) ** 2
    return torch.mean(se * w)

# ---------------------- Fonctions de prédiction PyTorch ----------------------
def predict_walk_forward_torch(model, scaler_X, scaler_y, daily_full_tf, feature_cols, target_count, seq_len, train_end_dt, test_df_tf, device, batch_size):
    values_full = daily_full_tf[feature_cols].values
    scaled_full = scaler_X.transform(values_full)
    X_all, y_all = create_sequences_multivar(scaled_full, seq_len, target_count)
    first_test_date = test_df_tf.index.min()
    pos = daily_full_tf.index.get_indexer([first_test_date])[0]
    n_train_samples = pos - seq_len + 1
    if n_train_samples < 0:
        raise ValueError("Pas assez de données historiques avant test pour créer des séquences.")
    X_test = X_all[n_train_samples:]
    if X_test.shape[0] == 0:
        return np.empty((0, target_count)), np.empty((0, target_count)), np.array([], dtype='datetime64[ns]')

    ds_test = SeqDataset(X_test, np.zeros((X_test.shape[0], target_count)))
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)  # (batch, n_outputs)
            preds_scaled.append(out.cpu().numpy())
    y_pred_scaled = np.vstack(preds_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    # true targets from y_all
    y_true = scaler_y.inverse_transform(y_all[n_train_samples:])
    dates = daily_full_tf.index[seq_len + n_train_samples: seq_len + n_train_samples + y_pred.shape[0]]
    return y_pred, y_true, dates

def predict_recursive_torch(model, scaler_X, scaler_y, history_orig, feature_cols, target_count, seq_len, future_dates, device):
    # history_orig: numpy array (seq_len, n_features) in original scale
    n_features = history_orig.shape[1]
    n_steps = len(future_dates)
    preds_targets = np.zeros((n_steps, target_count), dtype=float)
    current = history_orig.copy()
    model.eval()
    with torch.no_grad():
        for t in range(n_steps):
            cur_scaled = scaler_X.transform(current)  # (seq_len, n_features)
            xb = torch.from_numpy(cur_scaled.astype(np.float32)).unsqueeze(0).to(device)  # (1, seq_len, n_features)
            out = model(xb)  # (1, n_outputs) scaled for targets
            p_scaled = out.cpu().numpy()[0]
            p_inv = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
            preds_targets[t] = p_inv
            # slide window: remove first row, append new row where target positions replaced with p_inv
            row = np.array(current[-1, :], copy=True)
            row[:target_count] = p_inv
            if 'doy_sin' in feature_cols and 'doy_cos' in feature_cols:
                dt = pd.to_datetime(future_dates[t])
                idx_sin = feature_cols.index('doy_sin'); idx_cos = feature_cols.index('doy_cos')
                row[idx_sin] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
                row[idx_cos] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
            current = np.vstack([current[1:], row])
    return preds_targets

# ---------------------- Pipeline réutilisable pour la combinaison ALL_TARGETS ----------------------
def run_pipeline(target_cols, output_pdf_template):
    """
    Pipeline principal pour une combinaison donnée (PyTorch).
    Correction: imputation robuste des NaN dans values_train / values_full
    juste avant le fit des StandardScaler pour éviter les warnings sklearn.
    """
    print("\n" + "="*80)
    print(f"DÉBUT pipeline pour combinaison: {target_cols}")
    print("="*80 + "\n")
    # ré-init poids composants (1.0 par composante)
    component_weights = np.ones(len(target_cols), dtype=float)

    # Charge / filtre
    df = load_and_filter(PARQUET_PATH, START_DATE, DEPTH_CENTER, DEPTH_TOLERANCE, target_cols)
    if df.empty:
        raise ValueError("Aucune donnée après filtres. Vérifie profondeur/start_date.")

    daily = aggregate_daily(df, target_cols, agg_method=AGG_METHOD)
    if daily.empty:
        raise ValueError("Aucune donnée après agrégation journalière.")

    train_end_dt = pd.to_datetime(TRAIN_END, errors='coerce')
    if train_end_dt.tzinfo is None: train_end_dt = train_end_dt.tz_localize('UTC')
    else: train_end_dt = train_end_dt.tz_convert('UTC')
    test_end_dt = pd.to_datetime(TEST_END, errors='coerce')
    if test_end_dt.tzinfo is None: test_end_dt = test_end_dt.tz_localize('UTC')
    else: test_end_dt = test_end_dt.tz_convert('UTC')

    start = daily.index.min()
    if start.tzinfo is None: start = start.tz_localize('UTC')

    daily = daily[daily.index < test_end_dt]
    daily_full = reindex_and_impute(daily, start, test_end_dt)
    if daily_full.index.tz is None:
        daily_full.index = daily_full.index.tz_localize('UTC')

    train_df = daily_full[daily_full.index < train_end_dt]
    test_df = daily_full[(daily_full.index >= train_end_dt) & (daily_full.index < test_end_dt)]

    print(f"Total days: {len(daily_full)}, train: {len(train_df)}, test: {len(test_df)}")

    if len(train_df) <= SEQUENCE_LENGTH:
        raise ValueError("Pas assez de jours en entraînement pour la longueur de séquence choisie.")

    def add_time_features(df_):
        df = df_.copy()
        doy = df.index.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
        return df

    daily_full_tf = add_time_features(daily_full)
    train_df_tf = daily_full_tf[daily_full_tf.index < train_end_dt]
    test_df_tf = daily_full_tf[(daily_full_tf.index >= train_end_dt) & (daily_full_tf.index < test_end_dt)]

    feature_cols = list(target_cols) + ['doy_sin', 'doy_cos']
    values_full = daily_full_tf[feature_cols].values    # tableau complet (n_days_total, n_features)
    values_train = train_df_tf[feature_cols].values     # tableau d'entraînement (n_days_train, n_features)

    # ------------------ IMPUTATION ROBUSTE DES NaN (patch) ------------------
    # calcule les moyennes colonne-par-colonne sur l'entraînement en ignorant les NaN
    col_means = np.nanmean(values_train, axis=0)  # shape (n_features,)
    # fallback: si une colonne est entièrement NaN dans l'entraînement -> remplacer par 0.0 (modifiable)
    nan_mean_mask = np.isnan(col_means)
    if np.any(nan_mean_mask):
        print(f"Warning: {np.sum(nan_mean_mask)} colonnes entièrement NaN dans l'entraînement -> fallback mean=0.0 pour ces colonnes.")
        col_means[nan_mean_mask] = 0.0

    # comptage avant imputation (debug)
    nans_train_before = int(np.sum(np.isnan(values_train)))
    nans_full_before = int(np.sum(np.isnan(values_full)))
    if nans_train_before > 0 or nans_full_before > 0:
        print(f"Imputation: {nans_train_before} NaNs dans values_train, {nans_full_before} NaNs dans values_full -> remplacement par moyennes d'entraînement.")

    # remplissage des NaN dans values_train par les moyennes correspondantes
    inds_train = np.where(np.isnan(values_train))
    if inds_train[0].size > 0:
        values_train[inds_train] = np.take(col_means, inds_train[1])

    # remplissage des NaN dans values_full par les mêmes moyennes d'entraînement
    inds_full = np.where(np.isnan(values_full))
    if inds_full[0].size > 0:
        values_full[inds_full] = np.take(col_means, inds_full[1])

    # vérification après imputation (debug)
    nans_train_after = int(np.sum(np.isnan(values_train)))
    nans_full_after = int(np.sum(np.isnan(values_full)))
    print(f"Après imputation, NaNs dans values_train: {nans_train_after}, NaNs dans values_full: {nans_full_after}")
    if nans_train_after > 0 or nans_full_after > 0:
        raise RuntimeError("Des NaNs subsistent après imputation — vérifier les données / fallback.")

    # ------------------ FIN PATCH IMPUTATION ------------------

    # Now safe to fit scalers (aucun NaN présent)
    scaler_X = StandardScaler().fit(values_train)
    # pour scaler_y, on utilise les colonnes cibles tirées de train_df_tf : s'assurer qu'il n'y a pas de NaN
    y_train_df = train_df_tf[target_cols].values
    # remplir NaN dans y_train_df par la moyenne d'entraînement sur chaque colonne (mêmes moyennes calculées sur values_train
    # mais les positions de target dans feature_cols sont les premières colonnes)
    target_count = len(target_cols)
    target_means = col_means[:target_count]  # moyennes d'entraînement pour les colonnes cibles
    inds_y_train = np.where(np.isnan(y_train_df))
    if inds_y_train[0].size > 0:
        y_train_df[inds_y_train] = np.take(target_means, inds_y_train[1])
    scaler_y = StandardScaler().fit(y_train_df)

    # Continue pipeline : création des séquences à partir de values_full (déjà imputé)
    values_full_scaled = scaler_X.transform(values_full)

    X_all, y_all_scaled_targets = create_sequences_multivar(values_full_scaled, SEQUENCE_LENGTH, len(target_cols))
    _, y_all_targets_orig = create_sequences_multivar(values_full, SEQUENCE_LENGTH, len(target_cols))
    if y_all_targets_orig.size == 0:
        raise ValueError("Pas assez de séquences pour créer X/y. Réduis SEQUENCE_LENGTH.")

    y_all_targets_scaled = scaler_y.transform(y_all_targets_orig)

    L_train = len(train_df_tf)
    n_train_samples = max(0, L_train - SEQUENCE_LENGTH)
    if n_train_samples <= 0:
        raise ValueError("Aucun échantillon d'entraînement après découpage. Réduis SEQUENCE_LENGTH ou agrandis l'entraînement.")

    n_val_samples = max(1, int(np.ceil(VALIDATION_FRAC * n_train_samples)))
    if n_train_samples - n_val_samples <= 0:
        n_val_samples = max(1, n_train_samples // 10)

    train_end_idx = n_train_samples
    val_start_idx = train_end_idx - n_val_samples
    val_end_idx = train_end_idx

    X_train = X_all[:val_start_idx]
    y_train = y_all_targets_scaled[:val_start_idx]
    X_val = X_all[val_start_idx:val_end_idx]
    y_val = y_all_targets_scaled[val_start_idx:val_end_idx]

    X_test = X_all[n_train_samples:]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}  (contigu: last {n_val_samples} train samples)")
    print(f"X_test: {X_test.shape}")

    # création datasets / loaders
    ds_train = SeqDataset(X_train, y_train)
    ds_val = SeqDataset(X_val, y_val)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False pour séries temporelles contiguës
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    model = LSTMModel(n_features=n_features, hidden_size=128, hidden2=64, n_outputs=n_outputs, dropout=0.2).to(device)

    # optimiser + loss (on utilisera weighted_mse_loss)
    # normaliser poids comme avant
    w = component_weights.copy()
    w = w / (np.mean(w) + 1e-12)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping variables
    best_val = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(1, N_EPOCHS + 1):
        # training
        model.train()
        train_losses = []
        for xb, yb in loader_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)  # (batch, n_outputs) scaled
            loss = weighted_mse_loss(out, yb, w)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses)) if train_losses else float('nan')

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = weighted_mse_loss(out, yb, w)
                val_losses.append(loss.item())
        avg_val = float(np.mean(val_losses)) if val_losses else float('nan')

        print(f"Epoch {epoch}/{N_EPOCHS} — train loss: {avg_train:.6f}  val loss: {avg_val:.6f}")

        # early stopping check
        if not math.isnan(avg_val) and avg_val < best_val - 1e-12:
            best_val = avg_val
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping triggered (patience={PATIENCE}). Restauration des meilleurs poids (val loss={best_val:.6f}).")
            break

    # restore best weights if present
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict
    if RECURSIVE_FORECAST:
        first_test_date = test_df_tf.index.min()
        pos = daily_full_tf.index.get_indexer([first_test_date])[0]
        hist_start_idx = pos - SEQUENCE_LENGTH
        if hist_start_idx < 0:
            raise ValueError("Pas assez d'historique pour prévision récursive.")
        history_orig = daily_full_tf.iloc[hist_start_idx: hist_start_idx + SEQUENCE_LENGTH][feature_cols].values
        future_dates = pd.date_range(start=first_test_date, periods=len(test_df_tf), freq='D', tz=first_test_date.tz)
        preds_targets = predict_recursive_torch(model, scaler_X, scaler_y, history_orig, feature_cols, len(target_cols), SEQUENCE_LENGTH, future_dates, device)
        true_targets = test_df_tf[target_cols].values[:len(preds_targets)]
        dates = future_dates
    else:
        preds_targets, true_targets, dates = predict_walk_forward_torch(model, scaler_X, scaler_y, daily_full_tf, feature_cols, len(target_cols), SEQUENCE_LENGTH, train_end_dt, test_df_tf, device, batch_size=BATCH_SIZE)

    # --- Post-traitement OLS + métriques + PDF similaire à l'original ---
    CLAMP_TO_BOUNDS = False
    X_MIN, X_MAX = 1.0, 2.0
    Y_MIN, Y_MAX = 0.0, 2.0

    best_params = {}
    transformed_preds = np.zeros_like(preds_targets)

    for i, col in enumerate(target_cols):
        p = preds_targets[:, i].astype(float)
        t = true_targets[:, i].astype(float)
        valid = np.isfinite(p) & np.isfinite(t)
        p_v = p[valid]
        t_v = t[valid]
        if p_v.size == 0:
            bx, by = 1.5, 1.0
            br2 = np.nan
            best_params[col] = (bx, by, br2)
            transformed_preds[:, i] = p * bx + by
            print(f"[OLS] {col}: pas de valeurs valides, fallback x={bx}, y={by}")
            continue
        var_p = np.var(p_v, ddof=0)
        if var_p == 0:
            bx = 0.0
            by = float(np.mean(t_v))
            ss_tot = np.sum((t_v - np.mean(t_v))**2)
            ss_res = np.sum((t_v - by)**2)
            br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            best_params[col] = (bx, by, float(br2))
            transformed_preds[:, i] = p * bx + by
            print(f"[OLS] {col}: p constant -> x=0, y=mean(t)={by:.6f}, R2={br2:.6f}")
            continue
        cov = np.mean((p_v - np.mean(p_v)) * (t_v - np.mean(t_v)))
        bx = float(cov / var_p)
        by = float(np.mean(t_v) - bx * np.mean(p_v))
        if CLAMP_TO_BOUNDS:
            bx_clamped = float(np.clip(bx, X_MIN, X_MAX))
            by_clamped = float(np.clip(by, Y_MIN, Y_MAX))
            if (bx_clamped != bx) or (by_clamped != by):
                q_v = p_v * bx_clamped + by_clamped
                ss_res = np.sum((t_v - q_v) ** 2)
                ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
                br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
                bx, by = bx_clamped, by_clamped
            else:
                q_v = p_v * bx + by
                ss_res = np.sum((t_v - q_v) ** 2)
                ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
                br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        else:
            q_v = p_v * bx + by
            ss_res = np.sum((t_v - q_v) ** 2)
            ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
            br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        best_params[col] = (bx, by, float(br2))
        transformed_preds[:, i] = p * bx + by
        print(f"[OLS] {col}: x={bx:.6f}, y={by:.6f}, R2={br2:.6f} (n={p_v.size})")

    rmses = {}
    r2s = {}
    for i, col in enumerate(target_cols):
        valid = np.isfinite(transformed_preds[:, i]) & np.isfinite(true_targets[:, i])
        if np.sum(valid) == 0:
            rmses[col] = np.nan
            r2s[col] = np.nan
            continue
        rmse = math.sqrt(mean_squared_error(true_targets[valid, i], transformed_preds[valid, i]))
        r2 = r2_score(true_targets[valid, i], transformed_preds[valid, i])
        rmses[col] = rmse
        r2s[col] = r2
        print(f"{col}: RMSE={rmse:.4f}, R2={r2:.4f} (après OLS)")

    # préparer nom de fichier et sauvegarder PDF
    names_safe = "_".join([re.sub(r'[^A-Za-z0-9]+', '', c) for c in target_cols])
    out_template = output_pdf_template.format(names=names_safe)
    out_pdf = get_next_pdf_path(out_template)

    try:
        with PdfPages(out_pdf) as pdf:
            fig_all, axs = plt.subplots(nrows=len(target_cols), ncols=1, figsize=(10, 3 * len(target_cols)), constrained_layout=True)
            if len(target_cols) == 1:
                axs = [axs]
            for i, col in enumerate(target_cols):
                ax = axs[i]
                ax.plot(dates, true_targets[:, i], label='Réel')
                ax.plot(dates, transformed_preds[:, i], label=f'Prédit (x={best_params[col][0]:.4f}, y={best_params[col][1]:.4f})')
                ax.set_title(f"{col} — RMSE={rmses[col]:.3f}  R2={r2s[col]:.3f}")
                ax.legend()
                ax.set_ylabel(col)
            pdf.savefig(fig_all)
            plt.close(fig_all)

            fig_sum = plt.figure(figsize=(8.27, 11.69))
            fig_sum.clf()
            txt = "Résultats LSTM multivarié (après OLS)\n\n"
            txt += f"Période entraînement : {train_df.index.min().date()} -> {train_df.index.max().date()}\n"
            txt += f"Période test : {test_df.index.min().date()} -> {test_df.index.max().date()}\n\n"
            txt += "Metrics par variable (après transformation optimale) :\n"
            for col in target_cols:
                bx, by, br2 = best_params[col]
                txt += f" - {col}: RMSE={rmses[col]:.4f}, R2={r2s[col]:.4f}  (x={bx:.6f}, y={by:.6f}, R2_opt={br2:.6f})\n"
            fig_sum.text(0.01, 0.99, txt, fontsize=10, va='top')
            pdf.savefig()
            plt.close(fig_sum)
        print(f"PDF saved: {out_pdf}")
    except Exception as e:
        print("Erreur lors de la sauvegarde du PDF:", e)
        raise
    print(f"FIN pipeline pour combinaison: {target_cols}\n")



# ---------------------- Main: exécution unique sur ALL_TARGETS ----------------------
def main():
    target_cols = ALL_TARGETS.copy()
    print(f"Exécution unique sur la combinaison (ALL_TARGETS) : {target_cols}")
    try:
        run_pipeline(target_cols, OUTPUT_PDF_TEMPLATE)
    except Exception as e:
        print(f"Erreur lors de l'exécution pour {target_cols}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
