# CNN.py
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    Extracteur de caractéristiques par convolution 1D appliqué avant le GRU.
    Opère sur l'axe temporel tout en conservant la longueur de séquence (padding='same').
    """
    def __init__(self, n_features, out_channels=64, kernel_size=3, padding='same'):
        super().__init__()
        # Conv1d attend (batch, canaux, longueur) — la permutation est faite dans forward()
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Entrée : (batch, seq_len, n_features)
        # Permutation pour Conv1d : (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = self.relu(out)
        # Retour au format GRU : (batch, seq_len, out_channels)
        out = out.permute(0, 2, 1)
        return out
