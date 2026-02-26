# data_manager.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data_utils import (
    load_and_filter, aggregate_daily, reindex_and_impute,
    add_time_features, create_sequences_multivar,
    apply_vmd, apply_ceemdan, apply_ssa
)
from model import SeqDataset

class DataManager:
    """
    Orchestre le pipeline de préparation des données :
    chargement, agrégation, décomposition de signal, normalisation et création des séquences.
    """
    def __init__(self, config):
        self.config = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.train_df     = None
        self.test_df      = None
        self.feature_cols = None
        self.target_cols  = None

        self.train_loader = None
        self.val_loader   = None

        self.daily_full_tf = None
        self.train_end_dt  = None

    def prepare_data(self, target_cols=None):
        if target_cols is None:
            target_cols = self.config.ALL_TARGETS
        self.target_cols = target_cols

        print("\n" + "="*80)
        print(f"Préparation des données pour cibles: {target_cols}")
        print("="*80 + "\n")

        # 1. Chargement et filtrage
        df = load_and_filter(
            self.config.PARQUET_PATH,
            self.config.START_DATE,
            self.config.DEPTH_CENTER,
            self.config.DEPTH_TOLERANCE,
            target_cols,
            use_depth_filter=self.config.USE_PREPROCESSING
        )
        if df.empty:
            raise ValueError("Aucune donnée après filtres.")

        # 2. Gestion de la fréquence temporelle
        agg_cols = list(set(target_cols + self.config.INPUT_ONLY_COLS))
        if self.config.USE_PREPROCESSING:
            # Agrégation journalière forcée
            data_df = aggregate_daily(df, agg_cols, agg_method=self.config.AGG_METHOD)
            freq = 'D'
        else:
            # On garde les données brutes, mais on s'assure qu'elles sont indexées par temps et triées
            df = df.set_index('time (UTC)').sort_index()
            data_df = df[agg_cols]
            freq = None
        
        if data_df.empty:
            raise ValueError("Aucune donnée après filtrage/agrégation.")

        # 3. Réindexation (si nécessaire) et imputation
        self.train_end_dt = pd.to_datetime(self.config.TRAIN_END).tz_localize('UTC')
        test_end_dt = pd.to_datetime(self.config.TEST_END).tz_localize('UTC')
        start = data_df.index.min().tz_localize('UTC') if data_df.index.min().tzinfo is None else data_df.index.min()

        data_df = data_df[data_df.index < test_end_dt]
        data_full = reindex_and_impute(data_df, start, test_end_dt, freq=freq)
        if data_full.index.tz is None:
            data_full.index = data_full.index.tz_localize('UTC')

        # 4. Décomposition de signal et ingénierie des features
        method = self.config.DECOMPOSITION_METHOD
        decomp_cols = []

        if method == "VMD":
            data_full = apply_vmd(data_full, self.config)
            for col in self.config.DECOMPOSITION_COLS:
                for k in range(self.config.VMD_K):
                    decomp_cols.append(f"{col}_mode{k+1}")

        elif method == "CEEMDAN":
            data_full = apply_ceemdan(data_full, self.config)
            n_imfs = self.config.CEEMDAN_MAX_IMFS
            if n_imfs is not None:
                for col in self.config.DECOMPOSITION_COLS:
                    for k in range(n_imfs):
                        decomp_cols.append(f"{col}_mode{k+1}")
            else:
                # Nombre d'IMFs inconnu à l'avance
                for col in self.config.DECOMPOSITION_COLS:
                    decomp_cols += [c for c in data_full.columns if c.startswith(f"{col}_mode")]

        elif method == "SSA":
            data_full = apply_ssa(data_full, self.config)
            for col in self.config.DECOMPOSITION_COLS:
                decomp_cols += [c for c in data_full.columns if c.startswith(f"{col}_comp")]

        self.daily_full_tf = add_time_features(data_full)
        self.feature_cols = list(target_cols) + list(self.config.INPUT_ONLY_COLS) + list(self.config.TIME_FEATURE_COLS) + decomp_cols

        train_df_tf = self.daily_full_tf[self.daily_full_tf.index < self.train_end_dt]
        self.test_df = self.daily_full_tf[(self.daily_full_tf.index >= self.train_end_dt) & (self.daily_full_tf.index < test_end_dt)]

        # 5. Imputation des NaN résiduels
        values_train = train_df_tf[self.feature_cols].values
        values_full  = self.daily_full_tf[self.feature_cols].values
        self._impute_nans(values_train, values_full)

        # 6. Normalisation
        self.scaler_X.fit(values_train)
        y_train_raw = train_df_tf[target_cols].values

        y_means = np.nanmean(y_train_raw, axis=0)
        inds = np.where(np.isnan(y_train_raw))
        if inds[0].size > 0:
            y_train_raw[inds] = np.take(y_means, inds[1])
        self.scaler_y.fit(y_train_raw)

        # 7. Création des séquences et constitution des loaders
        values_full_scaled = self.scaler_X.transform(values_full)
        X_all, y_all_scaled_targets = create_sequences_multivar(values_full_scaled, self.config.SEQUENCE_LENGTH, len(target_cols))
        _, y_all_targets_orig = create_sequences_multivar(values_full, self.config.SEQUENCE_LENGTH, len(target_cols))

        L_train = len(train_df_tf)
        n_train_samples = max(0, L_train - self.config.SEQUENCE_LENGTH)
        n_val_samples   = max(1, int(np.ceil(self.config.VALIDATION_FRAC * n_train_samples)))
        val_start_idx   = n_train_samples - n_val_samples

        X_train = X_all[:val_start_idx]
        y_train = self.scaler_y.transform(y_all_targets_orig[:val_start_idx])

        X_val = X_all[val_start_idx:n_train_samples]
        y_val = self.scaler_y.transform(y_all_targets_orig[val_start_idx:n_train_samples])

        self.train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.val_loader   = DataLoader(SeqDataset(X_val, y_val),   batch_size=self.config.BATCH_SIZE, shuffle=False)

        self.n_features = X_train.shape[2]
        self.n_outputs  = y_train.shape[1]

        print(f"Données prêtes. Train shapes: X={X_train.shape}, y={y_train.shape}")

    def _impute_nans(self, values_train, values_full):
        """Remplace les NaN par la moyenne de la colonne calculée sur le train."""
        col_means = np.nanmean(values_train, axis=0)
        nan_mean_mask = np.isnan(col_means)
        if np.any(nan_mean_mask):
            col_means[nan_mean_mask] = 0.0

        inds_train = np.where(np.isnan(values_train))
        if inds_train[0].size > 0:
            values_train[inds_train] = np.take(col_means, inds_train[1])

        inds_full = np.where(np.isnan(values_full))
        if inds_full[0].size > 0:
            values_full[inds_full] = np.take(col_means, inds_full[1])
