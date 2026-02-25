# model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from CNN import CNNFeatureExtractor

class SeqDataset(Dataset):
    """Dataset PyTorch pour les séquences multivariées."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM à deux couches pour la prédiction multivariée (sans CNN)."""
    def __init__(self, n_features, hidden_size=128, hidden2=64, n_outputs=1, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, n_outputs)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)

class GRUModel(nn.Module):
    """GRU à deux couches pour la prédiction multivariée (sans CNN)."""
    def __init__(self, n_features, hidden_size=128, hidden2=64, n_outputs=1, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, n_outputs)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)

class CNNGRUModel(nn.Module):
    """
    Architecture CNN → GRU pour la prédiction de séries temporelles multivariées.
    Le CNN extrait des caractéristiques locales sur l'axe temporel avant le GRU.
    """
    def __init__(self, n_features, cnn_out_channels=64, hidden_size=128, hidden2=64, n_outputs=1, dropout=0.2):
        super().__init__()
        self.cnn = CNNFeatureExtractor(n_features, out_channels=cnn_out_channels)
        self.gru1 = nn.GRU(input_size=cnn_out_channels, hidden_size=hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, n_outputs)

    def forward(self, x):
        # x : (batch, seq_len, n_features)
        cnn_out = self.cnn(x)

        if not hasattr(self, '_printed'):
            print("x shape:", x.shape)
            print("cnn_out shape:", cnn_out.shape)
            self._printed = True

        out, _ = self.gru1(cnn_out)
        out, _ = self.gru2(out)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)

class CNNLSTMModel(nn.Module):
    """
    Architecture CNN → LSTM pour la prédiction de séries temporelles multivariées.
    Le CNN extrait des caractéristiques locales sur l'axe temporel avant le LSTM.
    """
    def __init__(self, n_features, cnn_out_channels=64, hidden_size=128, hidden2=64, n_outputs=1, dropout=0.2):
        super().__init__()
        self.cnn   = CNNFeatureExtractor(n_features, out_channels=cnn_out_channels)
        self.lstm1 = nn.LSTM(input_size=cnn_out_channels, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, n_outputs)

    def forward(self, x):
        # x : (batch, seq_len, n_features)
        cnn_out = self.cnn(x)

        if not hasattr(self, '_printed'):
            print("x shape:", x.shape)
            print("cnn_out shape:", cnn_out.shape)
            self._printed = True

        out, _ = self.lstm1(cnn_out)
        out, _ = self.lstm2(out)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)

def weighted_mse_loss(pred, target, weights):
    """MSE pondérée par cible, utile pour équilibrer des variables sur des échelles différentes."""
    w = torch.tensor(weights, dtype=pred.dtype, device=pred.device).reshape(1, -1)
    return torch.mean((pred - target) ** 2 * w)
