# CNN.py
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    Module CNN pour l'extraction de features avant le GRU.
    """
    def __init__(self, n_features, out_channels=64, kernel_size=3, padding='same'):
        super().__init__()
        # Conv1d attend (batch, channels, seq_len)
        # padding='same' permet de garder la même longueur de séquence
        self.conv1 = nn.Conv1d(
            in_channels=n_features, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.relu = nn.ReLU()
        # Optionnel: on peut ajouter d'autres couches ou du pooling si on veut réduire la dimension temporelle

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        
        # Pour Conv1d, on doit permuter pour avoir (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        
        out = self.conv1(x)
        out = self.relu(out)
        
        # On remet dans le format attendu par GRU: (batch, seq_len, out_channels)
        out = out.permute(0, 2, 1)
        
        return out
