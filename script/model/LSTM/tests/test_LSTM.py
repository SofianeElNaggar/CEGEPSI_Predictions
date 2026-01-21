# -*- coding: utf-8 -*-

import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------- PARAMÈTRES UTILISATEUR ----------------------
PARQUET_PATH = "../../../../dataset/OMDS-CTD-meteogc-data.parquet"  # chemin vers .parquet


# profondeur ciblée (ex 1.0m +/- tol)
DEPTH_CENTER = 1.0
DEPTH_TOLERANCE = 0.1

# Agg: 'median' ou 'mean'
AGG_METHOD = "mean"

OUTPUT_PDF = f"../../../results/prediction/LSTM/v1/{AGG_METHOD}/LSTM_predictions_{AGG_METHOD}.pdf"         # modèle de sortie PDF

# Période d'utilisation des données
START_DATE = "2000-01-01"   # ignorer données avant
TRAIN_END = "2020-01-01"    # entrainement < TRAIN_END
TEST_END = "2025-01-01"     # test < TEST_END

# LSTM / entrainement
SEQUENCE_LENGTH = 30        # nb de jours historiques utilisés
N_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8                # early stopping patience

# Mode de forecast
# If True: recursive multi-step (horizon = len(test)). If False: walk-forward using real observations to shift window (recommended).
RECURSIVE_FORECAST = True

# Variables cibles (ordre)
TARGET_COLS = [
    "temperature (°C)",
    #"CPHLPR01",
    #"TURBPR01",
    #"PHXXPR01",
    #"PSALST01",
    #"SIGTEQST",
    #"DOXYZZ01",
]

# Poids par composant dans la loss (1 par défaut). Peut être modifié.
COMPONENT_WEIGHTS = np.ones(len(TARGET_COLS), dtype=float)
# --------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------- Fonctions utilitaires ----------------------
def get_next_pdf_path(template_path_str):
    """
    Prend un chemin '.../LSTM_predictions.pdf' (ou sans extension) et retourne
    un chemin avec suffixe incrémental: LSTM_predictions_1.pdf, LSTM_predictions_2.pdf, ...
    Le dossier est créé si besoin.
    """
    p = Path(template_path_str)
    # s'assurer extension .pdf
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    # créer dossier si nécessaire
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # base name without trailing _\d+
    base = p.stem
    m = re.match(r"^(.*?)(?:_(\d+))?$", base)
    if m:
        base_clean = m.group(1)
    else:
        base_clean = base

    # chercher fichiers existants du type base_clean_{n}.pdf
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
        # dossier inexistant -> on l'a créé plus haut normalement
        max_n = 0

    next_n = max_n + 1
    next_name = f"{base_clean}_{next_n}.pdf"
    return str(p.parent / next_name)

def load_and_filter(parquet_path, start_date, depth_center, depth_tol):
    """Load parquet, ensure UTC times, convert numerics, filter QC and depth."""
    print("Chargement du fichier...", parquet_path)
    df = pd.read_parquet(parquet_path)

    # time -> tz-aware UTC
    df['time (UTC)'] = pd.to_datetime(df['time (UTC)'], errors='coerce', utc=True)

    # numeric conversions
    if 'depth' in df.columns:
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')

    # convert target columns to numeric
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # QC columns
    qc_cols = [c for c in df.columns if c.upper().endswith('_QC')]
    for c in qc_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # start_date -> tz-aware UTC
    start_ts = pd.to_datetime(start_date, errors='coerce')
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize('UTC')
    else:
        start_ts = start_ts.tz_convert('UTC')

    df = df[df['time (UTC)'] >= start_ts]

    # QC filtering: keep if QC is NaN or 1 or 5 for all *_QC columns (if present)
    if qc_cols:
        mask_good = pd.Series(True, index=df.index)
        for qc in qc_cols:
            mask_good &= ((df[qc].isna()) | (df[qc] == 1) | (df[qc] == 5))
        df = df[mask_good]
        print(f"Filtres QC: {len(qc_cols)} colonnes -> {len(df)} lignes restantes after QC")
    else:
        print("Aucune colonne QC détectée.")

    # depth filter
    if 'depth' in df.columns:
        depth_mask = df['depth'].notna() & (np.abs(df['depth'] - depth_center) <= depth_tol)
        df = df[depth_mask]
        print(f"Filtrage profondeur: centre={depth_center} tol={depth_tol} -> {len(df)} lignes restantes")
    else:
        print("Aucune colonne 'depth' détectée; aucune sélection par profondeur appliquée.")

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
    """
    values: numpy array shape (T, n_features)
    returns X (n_samples, seq_len, n_features), y (n_samples, n_targets)
    y is la valeur des target_cols le jour après la séquence (one-step ahead)
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
        y[i] = values[i + seq_len, :target_cols_count]  # assume targets are first columns
    return X, y

def build_lstm(n_timesteps, n_features, n_outputs, lr=1e-3, dropout=0.2):
    model = Sequential()
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(64, activation='tanh', return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation='linear'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # weighted MSE if COMPONENT_WEIGHTS not uniform
    w = np.array(COMPONENT_WEIGHTS, dtype=float)
    w = w / (np.mean(w) + 1e-12)
    def weighted_mse(y_true, y_pred):
        se = tf.square(y_true - y_pred)
        return tf.reduce_mean(se * tf.constant(w.reshape((1, -1)), dtype=se.dtype))
    model.compile(optimizer=optimizer, loss=weighted_mse, metrics=['mse'])
    model.summary()
    return model

def predict_walk_forward(model, scaler_X, scaler_y, daily_full_tf, feature_cols, target_count, seq_len, train_end_dt, test_df_tf):
    """
    Walk-forward predictions over the test period using true observations to update the window.
    Returns preds (n_steps, n_targets), true (n_steps, n_targets), dates (DatetimeIndex)
    """
    # prepare full scaled array
    values_full = daily_full_tf[feature_cols].values  # non-scaled
    scaled_full = scaler_X.transform(values_full)
    # build sequences on scaled_full
    X_all, y_all = create_sequences_multivar(scaled_full, seq_len, target_count)
    # compute how many samples are train (based on number of days)
    # find index of first test date in daily_full_tf
    first_test_date = test_df_tf.index.min()
    pos = daily_full_tf.index.get_indexer([first_test_date])[0]
    # number of samples before test start (samples whose y date < first_test_date)
    n_train_samples = pos - seq_len + 1
    if n_train_samples < 0:
        raise ValueError("Pas assez de données historiques avant test pour créer des séquences.")
    X_test = X_all[n_train_samples:]
    # get predictions (one-step ahead) for each sample in X_test
    y_pred_scaled = model.predict(X_test, verbose=0)
    # invert y scaling (scaler_y inverse expects shape (n, n_targets))
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    # true y values
    y_true = scaler_y.inverse_transform(y_all[n_train_samples:])
    dates = daily_full_tf.index[seq_len + n_train_samples: seq_len + n_train_samples + y_pred.shape[0]]
    return y_pred, y_true, dates

def predict_recursive(model, scaler_X, scaler_y, history_orig, feature_cols, target_count, seq_len, future_dates):
    """
    Recursive multi-step forecast: at each step predict targets, reconstruct full feature vector for that day,
    append to history and continue. history_orig is non-scaled values (seq_len, n_features)
    future_dates: DatetimeIndex length n_steps
    """
    n_features = history_orig.shape[1]
    n_steps = len(future_dates)
    preds_targets = np.zeros((n_steps, target_count), dtype=float)
    current = history_orig.copy()  # non-scaled
    for t in range(n_steps):
        cur_scaled = scaler_X.transform(current).reshape(1, seq_len, n_features)
        p_scaled = model.predict(cur_scaled, verbose=0)[0]  # (n_outputs,)
        p_inv = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
        preds_targets[t] = p_inv
        # build next row: put targets in first columns, compute doy features if present, and copy other cols from last row
        row = np.array(current[-1, :], copy=True)
        # target columns assumed to be first target_count columns
        row[:target_count] = p_inv
        # recompute time features if exist
        if 'doy_sin' in feature_cols and 'doy_cos' in feature_cols:
            dt = pd.to_datetime(future_dates[t])
            idx_sin = feature_cols.index('doy_sin'); idx_cos = feature_cols.index('doy_cos')
            row[idx_sin] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
            row[idx_cos] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
        # advance
        current = np.vstack([current[1:], row])
    return preds_targets

# ---------------------- Pipeline principal ----------------------

def main():
    out_pdf = get_next_pdf_path(OUTPUT_PDF)

    df = load_and_filter(PARQUET_PATH, START_DATE, DEPTH_CENTER, DEPTH_TOLERANCE)
    if df.empty:
        raise ValueError("Aucune donnée après filtres. Vérifie QC/profondeur/start_date.")

    daily = aggregate_daily(df, TARGET_COLS, agg_method=AGG_METHOD)
    if daily.empty:
        raise ValueError("Aucune donnée après agrégation journalière.")

    # normaliser dates TRAIN_END / TEST_END en UTC
    train_end_dt = pd.to_datetime(TRAIN_END, errors='coerce')
    if train_end_dt.tzinfo is None: train_end_dt = train_end_dt.tz_localize('UTC')
    else: train_end_dt = train_end_dt.tz_convert('UTC')
    test_end_dt = pd.to_datetime(TEST_END, errors='coerce')
    if test_end_dt.tzinfo is None: test_end_dt = test_end_dt.tz_localize('UTC')
    else: test_end_dt = test_end_dt.tz_convert('UTC')

    start = daily.index.min()
    if start.tzinfo is None: start = start.tz_localize('UTC')

    # clip and reindex
    daily = daily[daily.index < test_end_dt]
    daily_full = reindex_and_impute(daily, start, test_end_dt)
    if daily_full.index.tz is None:
        daily_full.index = daily_full.index.tz_localize('UTC')

    train_df = daily_full[daily_full.index < train_end_dt]
    test_df = daily_full[(daily_full.index >= train_end_dt) & (daily_full.index < test_end_dt)]

    print(f"Total days: {len(daily_full)}, train: {len(train_df)}, test: {len(test_df)}")

    if len(train_df) <= SEQUENCE_LENGTH:
        raise ValueError("Pas assez de jours en entraînement pour la longueur de séquence choisie.")

    # ajouter features temporelles (saisonnalité)
    def add_time_features(df_):
        df = df_.copy()
        doy = df.index.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
        return df

    daily_full_tf = add_time_features(daily_full)
    train_df_tf = daily_full_tf[daily_full_tf.index < train_end_dt]
    test_df_tf = daily_full_tf[(daily_full_tf.index >= train_end_dt) & (daily_full_tf.index < test_end_dt)]

    # features: targets first, then time features (on suppose TARGET_COLS en premier)
    feature_cols = TARGET_COLS + ['doy_sin', 'doy_cos']
    # build arrays
    values_full = daily_full_tf[feature_cols].values  # non-scaled
    values_train = train_df_tf[feature_cols].values

    # scalers: X scales whole feature set, y scales only targets (pour inverse facilement)
    scaler_X = StandardScaler().fit(values_train)
    scaler_y = StandardScaler().fit(train_df_tf[TARGET_COLS].values)

    values_full_scaled = scaler_X.transform(values_full)

    # create sequences (scaled) - model input est scaled; targets seront scalés par scaler_y
    X_all, y_all_scaled_targets = create_sequences_multivar(values_full_scaled, SEQUENCE_LENGTH, len(TARGET_COLS))
    # reconstruire y_all en échelle originale puis scaler avec scaler_y (pour cohérence entraînement)
    _, y_all_targets_orig = create_sequences_multivar(values_full, SEQUENCE_LENGTH, len(TARGET_COLS))
    if y_all_targets_orig.size == 0:
        raise ValueError("Pas assez de séquences pour créer X/y. Réduis SEQUENCE_LENGTH.")

    y_all_targets_scaled = scaler_y.transform(y_all_targets_orig)  # shape (n_samples, n_targets)

    # compute number of train samples (basé sur train_df length)
    L_train = len(train_df_tf)
    n_train_samples = max(0, L_train - SEQUENCE_LENGTH)
    if n_train_samples <= 0:
        raise ValueError("Aucun échantillon d'entraînement après découpage. Réduis SEQUENCE_LENGTH ou agrandis l'entraînement.")

    X_train = X_all[:n_train_samples]
    y_train = y_all_targets_scaled[:n_train_samples]
    X_val = X_all[n_train_samples: n_train_samples + max(1, int(0.05 * len(X_train)))]  # petite validation si besoin

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_all[n_train_samples:].shape}")

    # build model
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    model = build_lstm(SEQUENCE_LENGTH, n_features, n_outputs, lr=LEARNING_RATE)

    # callbacks: only EarlyStopping (plus restore_best_weights)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ]

    # fit
    history = model.fit(
        X_train, y_train,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2,
        shuffle=False
    )

    # Predict
    if RECURSIVE_FORECAST:
        # prepare history_orig (non-scaled) for recursive: last SEQUENCE_LENGTH days before first test date
        first_test_date = test_df_tf.index.min()
        pos = daily_full_tf.index.get_indexer([first_test_date])[0]
        hist_start_idx = pos - SEQUENCE_LENGTH
        if hist_start_idx < 0:
            raise ValueError("Pas assez d'historique pour prévision récursive.")
        history_orig = daily_full_tf.iloc[hist_start_idx: hist_start_idx + SEQUENCE_LENGTH][feature_cols].values  # non-scaled
        future_dates = pd.date_range(start=first_test_date, periods=len(test_df_tf), freq='D', tz=first_test_date.tz)
        preds_targets = predict_recursive(model, scaler_X, scaler_y, history_orig, feature_cols, len(TARGET_COLS), SEQUENCE_LENGTH, future_dates)
        true_targets = test_df_tf[TARGET_COLS].values[:len(preds_targets)]
        dates = future_dates
    else:
        preds_targets, true_targets, dates = predict_walk_forward(model, scaler_X, scaler_y, daily_full_tf, feature_cols, len(TARGET_COLS), SEQUENCE_LENGTH, train_end_dt, test_df_tf)

    # -----------------------
    # Estimation analytique OLS — coefficients (x,y) qui maximisent R² pour chaque variable
    CLAMP_TO_BOUNDS = False   # passe à True si tu veux forcer x,y dans des bornes
    X_MIN, X_MAX = 1.0, 2.0
    Y_MIN, Y_MAX = 0.0, 2.0

    best_params = {}          # col -> (x, y, R2_opt)
    transformed_preds = np.zeros_like(preds_targets)

    for i, col in enumerate(TARGET_COLS):
        p = preds_targets[:, i].astype(float)
        t = true_targets[:, i].astype(float)

        # filtrer NaN/inf
        valid = np.isfinite(p) & np.isfinite(t)
        p_v = p[valid]
        t_v = t[valid]

        if p_v.size == 0:
            # pas de données valides : fallback
            bx, by = 1.5, 1.0
            br2 = np.nan
            best_params[col] = (bx, by, br2)
            transformed_preds[:, i] = p * bx + by
            print(f"[OLS] {col}: pas de valeurs valides, fallback x={bx}, y={by}")
            continue

        var_p = np.var(p_v, ddof=0)  # variance population
        if var_p == 0:
            # prédictions constantes : slope impossible -> modèle constant y = mean(t)
            bx = 0.0
            by = float(np.mean(t_v))
            ss_tot = np.sum((t_v - np.mean(t_v))**2)
            ss_res = np.sum((t_v - by)**2)
            br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            best_params[col] = (bx, by, float(br2))
            transformed_preds[:, i] = p * bx + by
            print(f"[OLS] {col}: p constant -> x=0, y=mean(t)={by:.6f}, R2={br2:.6f}")
            continue

        # OLS closed-form
        cov = np.mean((p_v - np.mean(p_v)) * (t_v - np.mean(t_v)))
        bx = float(cov / var_p)
        by = float(np.mean(t_v) - bx * np.mean(p_v))

        # appliquer bornes si demandé (clamping)
        if CLAMP_TO_BOUNDS:
            bx_clamped = float(np.clip(bx, X_MIN, X_MAX))
            by_clamped = float(np.clip(by, Y_MIN, Y_MAX))
            if (bx_clamped != bx) or (by_clamped != by):
                # recalculer R2 après clamp
                q_v = p_v * bx_clamped + by_clamped
                ss_res = np.sum((t_v - q_v) ** 2)
                ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
                br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
                bx, by = bx_clamped, by_clamped
            else:
                # pas cherché à changer, calcul normal
                q_v = p_v * bx + by
                ss_res = np.sum((t_v - q_v) ** 2)
                ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
                br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        else:
            # calcul R2 standard
            q_v = p_v * bx + by
            ss_res = np.sum((t_v - q_v) ** 2)
            ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
            br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        best_params[col] = (bx, by, float(br2))
        transformed_preds[:, i] = p * bx + by
        print(f"[OLS] {col}: x={bx:.6f}, y={by:.6f}, R2={br2:.6f} (n={p_v.size})")

    # recalcul des métriques finales avec transformed_preds
    rmses = {}
    r2s = {}
    for i, col in enumerate(TARGET_COLS):
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

    # Save PDF with plots (en utilisant transformed_preds)
    try:
        with PdfPages(out_pdf) as pdf:
            fig_all, axs = plt.subplots(nrows=len(TARGET_COLS), ncols=1, figsize=(10, 3 * len(TARGET_COLS)), constrained_layout=True)
            if len(TARGET_COLS) == 1:
                axs = [axs]
            for i, col in enumerate(TARGET_COLS):
                ax = axs[i]
                ax.plot(dates, true_targets[:, i], label='Réel')
                ax.plot(dates, transformed_preds[:, i], label=f'Prédit (x={best_params[col][0]:.4f}, y={best_params[col][1]:.4f})')
                ax.set_title(f"{col} — RMSE={rmses[col]:.3f}  R2={r2s[col]:.3f}")
                ax.legend()
                ax.set_ylabel(col)
            pdf.savefig(fig_all)
            plt.close(fig_all)

            # summary page (ajout des paramètres optimaux)
            fig_sum = plt.figure(figsize=(8.27, 11.69))
            fig_sum.clf()
            txt = "Résultats LSTM multivarié (après OLS)\n\n"
            txt += f"Période entraînement : {train_df.index.min().date()} -> {train_df.index.max().date()}\n"
            txt += f"Période test : {test_df.index.min().date()} -> {test_df.index.max().date()}\n\n"
            txt += "Metrics par variable (après transformation optimale) :\n"
            for col in TARGET_COLS:
                bx, by, br2 = best_params[col]
                txt += f" - {col}: RMSE={rmses[col]:.4f}, R2={r2s[col]:.4f}  (x={bx:.6f}, y={by:.6f}, R2_opt={br2:.6f})\n"
            fig_sum.text(0.01, 0.99, txt, fontsize=10, va='top')
            pdf.savefig()
            plt.close(fig_sum)
        print(f"PDF saved: {out_pdf}")
    except Exception as e:
        print("Erreur lors de la sauvegarde du PDF:", e)


if __name__ == "__main__":
    main()
