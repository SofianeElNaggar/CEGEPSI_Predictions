# evaluator.py
import numpy as np
import pandas as pd
import torch
import math
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from pdf_utils import save_results_pdf
from model import SeqDataset
from data_utils import create_sequences_multivar

class Evaluator:
    def __init__(self, model, data_manager, config):
        self.model = model
        self.dm = data_manager
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self):
        print("\n" + "="*80)
        print("DÉBUT évaluation...")
        print("="*80 + "\n")
        
        # Prediction
        if self.config.RECURSIVE_FORECAST:
            preds_targets, true_targets, dates = self._predict_recursive()
        else:
            preds_targets, true_targets, dates = self._predict_walk_forward()
            
        # OLS Correction & Metrics
        best_params = {}
        transformed_preds = np.zeros_like(preds_targets)
        target_cols = self.dm.target_cols
        feature_cols = self.dm.feature_cols
        
        # Ranges for clamping (legacy constants)
        X_MIN, X_MAX = 1.0, 2.0
        Y_MIN, Y_MAX = 0.0, 2.0
        CLAMP_TO_BOUNDS = False

        rmses = {}
        r2s = {}

        for i, col in enumerate(target_cols):
            p = preds_targets[:, i].astype(float)
            t = true_targets[:, i].astype(float)
            valid = np.isfinite(p) & np.isfinite(t)
            
            p_v = p[valid]
            t_v = t[valid]
            
            if p_v.size == 0:
                print(f"[OLS] {col}: pas de données valides.")
                best_params[col] = (1.5, 1.0, np.nan)
                continue
                
            # OLS Calculation
            var_p = np.var(p_v, ddof=0)
            if var_p == 0:
                bx = 0.0
                by = float(np.mean(t_v))
                br2 = 0.0 # simplified
            else:
                cov = np.mean((p_v - np.mean(p_v)) * (t_v - np.mean(t_v)))
                bx = float(cov / var_p)
                by = float(np.mean(t_v) - bx * np.mean(p_v))
                
                # R2
                q_v = p_v * bx + by
                ss_res = np.sum((t_v - q_v)**2)
                ss_tot = np.sum((t_v - np.mean(t_v))**2)
                br2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            best_params[col] = (bx, by, float(br2))
            transformed_preds[:, i] = p * bx + by
            
            # Metrics
            valid_tr = np.isfinite(transformed_preds[:, i]) & np.isfinite(true_targets[:, i])
            rmse = math.sqrt(mean_squared_error(true_targets[valid_tr, i], transformed_preds[valid_tr, i]))
            r2 = r2_score(true_targets[valid_tr, i], transformed_preds[valid_tr, i])
            
            rmses[col] = rmse
            r2s[col] = r2
            print(f"{col}: OLS x={bx:.4f}, y={by:.4f} -> RMSE={rmse:.4f}, R2={r2:.4f}")

        # PDF Report
        output_path = self.config.get_output_path()
        try:
            # Reconstruct train/test df for plotting context
             # Using indices from dm
             pass
        except:
             pass
             
        # Need train_df and test_df for plotting
        # dm.train_loader is processed, we need the DF
        # We stored them in dm.prepare_data
        
        # Re-split daily_full_tf based on indices logic if needed, or just use what we have
        # dm.daily_full_tf is available
        train_df = self.dm.daily_full_tf[self.dm.daily_full_tf.index < self.dm.train_end_dt]
        test_df = self.dm.test_df
        
        save_results_pdf(
            output_path, 
            target_cols, 
            feature_cols,
            dates, 
            true_targets, 
            transformed_preds, 
            best_params, 
            rmses, 
            r2s, 
            train_df, 
            test_df
        )
        print(f"Rapport PDF généré : {output_path}")

    def _predict_recursive(self):
        # ... logic from predict_recursive_torch ...
        # Need: history_orig (last window of train/full before test)
        # We can extract from dm.daily_full_tf
        
        daily_full_tf = self.dm.daily_full_tf
        feature_cols = self.dm.feature_cols
        target_cols = self.dm.target_cols
        seq_len = self.config.SEQUENCE_LENGTH
        
        first_test_date = self.dm.test_df.index.min()
        pos = daily_full_tf.index.get_indexer([first_test_date])[0]
        hist_start_idx = pos - seq_len
        
        if hist_start_idx < 0:
             raise ValueError("Pas assez d'historique pour forecast.")
             
        history_orig = daily_full_tf.iloc[hist_start_idx: hist_start_idx + seq_len][feature_cols].values
        future_dates = pd.date_range(
            start=first_test_date, 
            periods=len(self.dm.test_df), 
            freq='D', 
            tz=first_test_date.tz
        )
        
        n_steps = len(future_dates)
        target_count = len(target_cols)
        preds_targets = np.zeros((n_steps, target_count), dtype=float)
        current = history_orig.copy()
        
        with torch.no_grad():
            for t in range(n_steps):
                cur_scaled = self.dm.scaler_X.transform(current)
                xb = torch.from_numpy(cur_scaled.astype('float32')).unsqueeze(0).to(self.device)
                out = self.model(xb)
                p_scaled = out.cpu().numpy()[0]
                p_inv = self.dm.scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
                preds_targets[t] = p_inv
                
                # Update current window
                row = np.array(current[-1, :], copy=True)
                # Put predictions back into feature vector
                # Assumes targets are first columns in feature_cols (which they are)
                row[:target_count] = p_inv
                
                # Update time features
                if 'doy_sin' in feature_cols and 'doy_cos' in feature_cols:
                    dt = pd.to_datetime(future_dates[t])
                    idx_sin = feature_cols.index('doy_sin')
                    idx_cos = feature_cols.index('doy_cos')
                    row[idx_sin] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
                    row[idx_cos] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
                
                current = np.vstack([current[1:], row])

        true_targets = self.dm.test_df[target_cols].values[:n_steps]
        return preds_targets, true_targets, future_dates

    def _predict_walk_forward(self):
        daily_full_tf = self.dm.daily_full_tf
        feature_cols = self.dm.feature_cols
        target_cols = self.dm.target_cols
        seq_len = self.config.SEQUENCE_LENGTH
        target_count = len(target_cols)
        
        values_full = daily_full_tf[feature_cols].values
        scaled_full = self.dm.scaler_X.transform(values_full)
        
        # Reuse creation function
        X_all, y_all = create_sequences_multivar(scaled_full, seq_len, target_count)
        
        first_test_date = self.dm.test_df.index.min()
        pos = daily_full_tf.index.get_indexer([first_test_date])[0]
        n_train_samples = pos - seq_len + 1 # Align with pipeline logic
        
        X_test = X_all[n_train_samples:]
        if X_test.shape[0] == 0:
            return np.empty((0, target_count)), np.empty((0, target_count)), []
            
        ds_test = SeqDataset(X_test, np.zeros((X_test.shape[0], target_count)))
        loader = DataLoader(ds_test, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        preds_scaled = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                preds_scaled.append(out.cpu().numpy())
                
        y_pred_scaled = np.vstack(preds_scaled)
        y_pred = self.dm.scaler_y.inverse_transform(y_pred_scaled)
        
        # Get ground truth from FULL original values (pipeline logic)
        # But we need corresponding ground truth for the sequences
        # Warning: create_sequences_multivar returns (X, y) where y is at t+seq_len
        # We need to construct y_true aligned with X_test
        
        _, y_all_targets_orig = create_sequences_multivar(values_full, seq_len, target_count)
        y_true = y_all_targets_orig[n_train_samples : n_train_samples + y_pred.shape[0]]
        
        dates = daily_full_tf.index[seq_len + n_train_samples : seq_len + n_train_samples + y_pred.shape[0]]
        
        return y_pred, y_true, dates
