# pipeline.py
import math
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader

from utils import (PARQUET_PATH, DEPTH_CENTER, DEPTH_TOLERANCE, AGG_METHOD,
                   START_DATE, TRAIN_END, TEST_END, SEQUENCE_LENGTH, N_EPOCHS,
                   BATCH_SIZE, LEARNING_RATE, PATIENCE, VALIDATION_FRAC,
                   RECURSIVE_FORECAST, INPUT_ONLY_COLS, TIME_FEATURE_COLS)
from data_utils import (load_and_filter, aggregate_daily, reindex_and_impute,
                        create_sequences_multivar, add_time_features)
from model import LSTMModel, SeqDataset, weighted_mse_loss, predict_walk_forward_torch, predict_recursive_torch
from train import train_model
from pdf_utils import save_results_pdf
from pinns import CosSinPINN, DissolvedOxygenPINN, pHPINN  

def run_pipeline(target_cols, output_pdf_template):
    """
    Pipeline principal en version modulaire.
    Contient l'imputation robuste des NaNs (même méthode que précédemment).
    Intègre un point pour appeler des PINNs (si définis)
    """
    print("\n" + "="*80)
    print(f"DÉBUT pipeline pour combinaison: {target_cols}")
    print("="*80 + "\n")
    component_weights = np.ones(len(target_cols), dtype=float)

    df = load_and_filter(PARQUET_PATH, START_DATE, DEPTH_CENTER, DEPTH_TOLERANCE, target_cols)
    if df.empty:
        raise ValueError("Aucune donnée après filtres. Vérifie profondeur/start_date.")

    agg_cols = list(set(target_cols + INPUT_ONLY_COLS))
    daily = aggregate_daily(df, agg_cols, agg_method=AGG_METHOD)
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

    daily_full_tf = add_time_features(daily_full)
    train_df_tf = daily_full_tf[daily_full_tf.index < train_end_dt]
    test_df_tf = daily_full_tf[(daily_full_tf.index >= train_end_dt) & (daily_full_tf.index < test_end_dt)]

    feature_cols = list(target_cols) + list(INPUT_ONLY_COLS) + list(TIME_FEATURE_COLS)
    values_full = daily_full_tf[feature_cols].values
    values_train = train_df_tf[feature_cols].values

    # ------------ IMPUTATION ROBUSTE NaNs ----------
    col_means = np.nanmean(values_train, axis=0)
    nan_mean_mask = np.isnan(col_means)
    if np.any(nan_mean_mask):
        print(f"Warning: {np.sum(nan_mean_mask)} colonnes entièrement NaN dans l'entraînement -> fallback mean=0.0 pour ces colonnes.")
        col_means[nan_mean_mask] = 0.0

    nans_train_before = int(np.sum(np.isnan(values_train)))
    nans_full_before = int(np.sum(np.isnan(values_full)))
    if nans_train_before > 0 or nans_full_before > 0:
        print(f"Imputation: {nans_train_before} NaNs dans values_train, {nans_full_before} NaNs dans values_full -> remplacement par moyennes d'entraînement.")

    inds_train = np.where(np.isnan(values_train))
    if inds_train[0].size > 0:
        values_train[inds_train] = np.take(col_means, inds_train[1])
    inds_full = np.where(np.isnan(values_full))
    if inds_full[0].size > 0:
        values_full[inds_full] = np.take(col_means, inds_full[1])

    nans_train_after = int(np.sum(np.isnan(values_train)))
    nans_full_after = int(np.sum(np.isnan(values_full)))
    print(f"Après imputation, NaNs dans values_train: {nans_train_after}, NaNs dans values_full: {nans_full_after}")
    if nans_train_after > 0 or nans_full_after > 0:
        raise RuntimeError("Des NaNs subsistent après imputation — vérifier les données / fallback.")

    scaler_X = StandardScaler().fit(values_train)
    y_train_df = train_df_tf[target_cols].values
    target_count = len(target_cols)
    target_means = col_means[:target_count]
    inds_y_train = np.where(np.isnan(y_train_df))
    if inds_y_train[0].size > 0:
        y_train_df[inds_y_train] = np.take(target_means, inds_y_train[1])
    scaler_y = StandardScaler().fit(y_train_df)

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

    ds_train = SeqDataset(X_train, y_train)
    ds_val = SeqDataset(X_val, y_val)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    model = LSTMModel(n_features=n_features, hidden_size=128, hidden2=64, n_outputs=n_outputs, dropout=0.2).to(device)

    w = component_weights.copy()
    w = w / (np.mean(w) + 1e-12)
    
    # Création des PINNs avant l'optimizer pour pouvoir ajouter leurs paramètres
    pinns = [
        CosSinPINN(
            'doy_sin',
            'doy_cos',
            in_targets=False,
            weight=1.0
        ),
        DissolvedOxygenPINN(
            do_name="dissolved_oxygen (ml l-1)", # TARGET
            temp_water_name="temperature (°C)",  # INPUT
            temp_air_name="Mean Temp (°C)",      # INPUT
            chl_name="chlorophyll (mg m-3)",     # INPUT
            wind_name="Spd of Max Gust (km/h)",  # INPUT
            sal_name="salinity (PSS-78)",        # INPUT
            tide_name="tide_range (m)",          # INPUT
            weight=1
        ),
        #pHPINN(
        #    ph_name="pH",                        # TARGET
        #    temp_water_name="temperature (°C)",  # INPUT
        #    sal_name="salinity (PSS-78)",        # INPUT
        #    chl_name="chlorophyll (mg m-3)",     # INPUT
        #    do_name="dissolved_oxygen (ml l-1)", # INPUT
        #    wind_name="Spd of Max Gust (km/h)",  # INPUT
        #    tide_name="tide_range (m)",          # INPUT
        #    weight=1
        #)
    ]
    
    # Déplacer les PINNs qui sont des nn.Module sur le device
    for pinn in pinns:
        if isinstance(pinn, torch.nn.Module):
            pinn.to(device)
    
    # Ajouter les paramètres du modèle ET des PINNs à l'optimizer
    all_params = list(model.parameters())
    for pinn in pinns:
        if isinstance(pinn, torch.nn.Module):
            all_params.extend(list(pinn.parameters()))
    
    optimizer = torch.optim.Adam(all_params, lr=LEARNING_RATE)

    # Remarque: les PINNs prennent batch_inputs (torch tensor) et batch_preds (torch) + meta dict
    best_val = float('inf')
    best_state = None
    best_pinn_states = {}
    patience_cnt = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_losses = []
        train_pinn_losses = []
        for xb, yb in loader_train:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = weighted_mse_loss(out, yb, w)
            # compute PINN losses on this batch (if any)
            # IMPORTANT: passer xb directement (tensor) pour préserver le graphe de calcul
            pinn_loss = 0.0
            meta = {'feature_cols': feature_cols, 'target_names': target_cols, 'scaler_y': scaler_y, 'device': device}
            for pinn in pinns:
                l = pinn.compute_pinn_loss(xb, out, meta)
                if l is not None:
                    pinn_loss = pinn_loss + l
                    train_pinn_losses.append(l.item())
            if pinn_loss != 0.0:
                loss = loss + pinn_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses)) if train_losses else float('nan')
        avg_pinn_train = float(np.mean(train_pinn_losses)) if train_pinn_losses else 0.0

        # validation
        model.eval()
        val_losses = []
        val_pinn_losses = []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                loss = weighted_mse_loss(out, yb, w)
                pinn_loss = 0.0
                meta = {'feature_cols': feature_cols, 'target_names': target_cols, 'scaler_y': scaler_y, 'device': device}
                for pinn in pinns:
                    l = pinn.compute_pinn_loss(xb, out, meta)
                    if l is not None:
                        pinn_loss = pinn_loss + l
                        val_pinn_losses.append(l.item())
                if pinn_loss != 0.0:
                    loss = loss + pinn_loss
                val_losses.append(loss.item())
        avg_val = float(np.mean(val_losses)) if val_losses else float('nan')
        avg_pinn_val = float(np.mean(val_pinn_losses)) if val_pinn_losses else 0.0
        
        print(f"Epoch {epoch}/{N_EPOCHS} — train loss: {avg_train:.6f}  val loss: {avg_val:.6f}  PINN train: {avg_pinn_train:.6f}  PINN val: {avg_pinn_val:.6f}")

        if not math.isnan(avg_val) and avg_val < best_val - 1e-12:
            best_val = avg_val
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            # Sauvegarder aussi les états des PINNs
            best_pinn_states = {}
            for i, pinn in enumerate(pinns):
                if isinstance(pinn, torch.nn.Module):
                    best_pinn_states[i] = {k:v.cpu().clone() for k,v in pinn.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping triggered (patience={PATIENCE}). Restauration des meilleurs poids (val loss={best_val:.6f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        # Restaurer aussi les états des PINNs
        for i, pinn in enumerate(pinns):
            if isinstance(pinn, torch.nn.Module) and i in best_pinn_states:
                pinn.load_state_dict(best_pinn_states[i])

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
        preds_targets, true_targets, dates = predict_walk_forward_torch(model, scaler_X, scaler_y, daily_full_tf, feature_cols, len(target_cols), SEQUENCE_LENGTH, train_end_dt, test_df_tf, device, batch_size=BATCH_SIZE, create_sequences_fn=create_sequences_multivar)

    # Post-traitement OLS + métriques + PDF
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

    # Métriques brutes (avant OLS) pour évaluation réelle
    rmses_raw = {}
    r2s_raw = {}
    rmses = {}
    r2s = {}
    for i, col in enumerate(target_cols):
        valid = np.isfinite(preds_targets[:, i]) & np.isfinite(true_targets[:, i])
        if np.sum(valid) == 0:
            rmses_raw[col] = np.nan
            r2s_raw[col] = np.nan
            rmses[col] = np.nan
            r2s[col] = np.nan
            continue
        # Métriques brutes
        rmse_raw = math.sqrt(mean_squared_error(true_targets[valid, i], preds_targets[valid, i]))
        r2_raw = r2_score(true_targets[valid, i], preds_targets[valid, i])
        rmses_raw[col] = rmse_raw
        r2s_raw[col] = r2_raw
        print(f"{col}: RMSE brut={rmse_raw:.4f}, R2 brut={r2_raw:.4f}")
        
        # Métriques après OLS
        valid_transformed = np.isfinite(transformed_preds[:, i]) & np.isfinite(true_targets[:, i])
        if np.sum(valid_transformed) == 0:
            rmses[col] = np.nan
            r2s[col] = np.nan
            continue
        rmse = math.sqrt(mean_squared_error(true_targets[valid_transformed, i], transformed_preds[valid_transformed, i]))
        r2 = r2_score(true_targets[valid_transformed, i], transformed_preds[valid_transformed, i])
        rmses[col] = rmse
        r2s[col] = r2
        print(f"{col}: RMSE={rmse:.4f}, R2={r2:.4f} (après OLS)")

    # sauvegarde PDF
    save_results_pdf(output_pdf_template, target_cols, dates, true_targets, transformed_preds, best_params, rmses, r2s, train_df, test_df)

    print(f"FIN pipeline pour combinaison: {target_cols} - {INPUT_ONLY_COLS}\n")
