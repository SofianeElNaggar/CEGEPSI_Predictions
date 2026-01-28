# data_manager.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data_utils import (
    load_and_filter, aggregate_daily, reindex_and_impute,
    add_time_features, create_sequences_multivar, apply_vmd
)
from model import SeqDataset # We might need to move SeqDataset or import it

class DataManager:
    def __init__(self, config):
        self.config = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # State
        self.train_df = None
        self.test_df = None
        self.feature_cols = None
        self.target_cols = None
        
        # Loaders
        self.train_loader = None
        self.val_loader = None
        
        # Data for evaluation
        self.daily_full_tf = None
        self.train_end_dt = None
        
    def prepare_data(self, target_cols=None):
        if target_cols is None:
            target_cols = self.config.ALL_TARGETS
        self.target_cols = target_cols
        
        print("\n" + "="*80)
        print(f"Preparation des donnees pour cibles: {target_cols}")
        print("="*80 + "\n")

        # 1. Load & Filter
        df = load_and_filter(
            self.config.PARQUET_PATH, 
            self.config.START_DATE, 
            self.config.DEPTH_CENTER, 
            self.config.DEPTH_TOLERANCE, 
            target_cols
        )
        if df.empty:
            raise ValueError("Aucune donnée après filtres.")

        # 2. Aggregate
        agg_cols = list(set(target_cols + self.config.INPUT_ONLY_COLS))
        daily = aggregate_daily(df, agg_cols, agg_method=self.config.AGG_METHOD)
        if daily.empty:
            raise ValueError("Aucune donnée après agrégation journalière.")

        # 3. Impute & Split
        self.train_end_dt = pd.to_datetime(self.config.TRAIN_END).tz_localize('UTC')
        test_end_dt = pd.to_datetime(self.config.TEST_END).tz_localize('UTC')
        start = daily.index.min().tz_localize('UTC') if daily.index.min().tzinfo is None else daily.index.min()
        
        daily = daily[daily.index < test_end_dt]
        daily_full = reindex_and_impute(daily, start, test_end_dt)
        if daily_full.index.tz is None:
             daily_full.index = daily_full.index.tz_localize('UTC')

        # 4. Feature Engineering
        daily_full = apply_vmd(daily_full, self.config)
        
        self.daily_full_tf = add_time_features(daily_full)
        
        # Identification des colonnes VMD
        vmd_cols = []
        if self.config.VMD_ENABLED:
            for col in self.config.VMD_COLS:
                for k in range(self.config.VMD_K):
                    vmd_cols.append(f"{col}_mode{k+1}")
        
        self.feature_cols = list(target_cols) + list(self.config.INPUT_ONLY_COLS) + list(self.config.TIME_FEATURE_COLS) + vmd_cols
        
        train_df_tf = self.daily_full_tf[self.daily_full_tf.index < self.train_end_dt]
        self.test_df = self.daily_full_tf[(self.daily_full_tf.index >= self.train_end_dt) & (self.daily_full_tf.index < test_end_dt)]
        
        # 5. Robust Imputation (Target-specific logic from pipeline)
        values_train = train_df_tf[self.feature_cols].values
        values_full = self.daily_full_tf[self.feature_cols].values
        
        self._impute_nans(values_train, values_full)
        
        # 6. Scaling
        self.scaler_X.fit(values_train)
        y_train_raw = train_df_tf[target_cols].values
        
        # Impute NaNs in y_train for scaler fitting
        y_means = np.nanmean(y_train_raw, axis=0)
        inds = np.where(np.isnan(y_train_raw))
        if inds[0].size > 0:
            y_train_raw[inds] = np.take(y_means, inds[1])
            
        self.scaler_y.fit(y_train_raw)
        
        # 7. Create Sequences
        values_full_scaled = self.scaler_X.transform(values_full)
        X_all, y_all_scaled_targets = create_sequences_multivar(values_full_scaled, self.config.SEQUENCE_LENGTH, len(target_cols))
        _, y_all_targets_orig = create_sequences_multivar(values_full, self.config.SEQUENCE_LENGTH, len(target_cols))
        
        L_train = len(train_df_tf)
        n_train_samples = max(0, L_train - self.config.SEQUENCE_LENGTH)
        
        # Validation Split
        n_val_samples = max(1, int(np.ceil(self.config.VALIDATION_FRAC * n_train_samples)))
        val_start_idx = n_train_samples - n_val_samples
        
        X_train = X_all[:val_start_idx]
        y_train = self.scaler_y.transform(y_all_targets_orig[:val_start_idx])
        
        X_val = X_all[val_start_idx:n_train_samples]
        y_val = self.scaler_y.transform(y_all_targets_orig[val_start_idx:n_train_samples])
        
        self.train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        self.n_features = X_train.shape[2]
        self.n_outputs = y_train.shape[1]
        
        print(f"Données prêtes. Train shapes: X={X_train.shape}, y={y_train.shape}")
        
    def _impute_nans(self, values_train, values_full):
        col_means = np.nanmean(values_train, axis=0)
        nan_mean_mask = np.isnan(col_means) 
        if np.any(nan_mean_mask):
             col_means[nan_mean_mask] = 0.0
             
        # Apply to train
        inds_train = np.where(np.isnan(values_train))
        if inds_train[0].size > 0:
            values_train[inds_train] = np.take(col_means, inds_train[1])
            
        # Apply to full
        inds_full = np.where(np.isnan(values_full))
        if inds_full[0].size > 0:
            values_full[inds_full] = np.take(col_means, inds_full[1])
