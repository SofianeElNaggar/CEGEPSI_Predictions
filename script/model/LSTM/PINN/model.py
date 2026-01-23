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

