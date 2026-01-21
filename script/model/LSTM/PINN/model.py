# model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=128, hidden2=64, n_outputs=1, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, n_outputs)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]
        last = self.dropout(last)
        out = self.fc(last)
        return out

def weighted_mse_loss(pred, target, weights):
    w = torch.tensor(weights, dtype=pred.dtype, device=pred.device).reshape(1, -1)
    se = (pred - target) ** 2
    return torch.mean(se * w)

# Fonctions de prédiction (recopie des fonctions précédentes en version modulaire)
def predict_walk_forward_torch(model, scaler_X, scaler_y, daily_full_tf, feature_cols, target_count, seq_len, train_end_dt, test_df_tf, device, batch_size, create_sequences_fn):
    values_full = daily_full_tf[feature_cols].values
    scaled_full = scaler_X.transform(values_full)
    X_all, y_all = create_sequences_fn(scaled_full, seq_len, target_count)
    first_test_date = test_df_tf.index.min()
    pos = daily_full_tf.index.get_indexer([first_test_date])[0]
    n_train_samples = pos - seq_len + 1
    if n_train_samples < 0:
        raise ValueError("Pas assez de données historiques avant test pour créer des séquences.")
    X_test = X_all[n_train_samples:]
    if X_test.shape[0] == 0:
        return np.empty((0, target_count)), np.empty((0, target_count)), np.array([], dtype='datetime64[ns]')

    from torch.utils.data import DataLoader
    ds_test = SeqDataset(X_test, np.zeros((X_test.shape[0], target_count)))
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            preds_scaled.append(out.cpu().numpy())
    y_pred_scaled = np.vstack(preds_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_all[n_train_samples:])
    dates = daily_full_tf.index[seq_len + n_train_samples: seq_len + n_train_samples + y_pred.shape[0]]
    return y_pred, y_true, dates

def predict_recursive_torch(model, scaler_X, scaler_y, history_orig, feature_cols, target_count, seq_len, future_dates, device):
    n_features = history_orig.shape[1]
    n_steps = len(future_dates)
    preds_targets = np.zeros((n_steps, target_count), dtype=float)
    current = history_orig.copy()
    model.eval()
    with torch.no_grad():
        for t in range(n_steps):
            cur_scaled = scaler_X.transform(current)
            xb = torch.from_numpy(cur_scaled.astype('float32')).unsqueeze(0).to(device)
            out = model(xb)
            p_scaled = out.cpu().numpy()[0]
            p_inv = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
            preds_targets[t] = p_inv
            row = np.array(current[-1, :], copy=True)
            row[:target_count] = p_inv
            if 'doy_sin' in feature_cols and 'doy_cos' in feature_cols:
                dt = pd.to_datetime(future_dates[t])
                idx_sin = feature_cols.index('doy_sin'); idx_cos = feature_cols.index('doy_cos')
                row[idx_sin] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
                row[idx_cos] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
            current = np.vstack([current[1:], row])
    return preds_targets
