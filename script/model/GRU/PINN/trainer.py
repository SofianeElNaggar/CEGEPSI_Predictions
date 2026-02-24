# trainer.py
import numpy as np
import torch
import torch.nn as nn
from model import weighted_mse_loss

class Trainer:
    """
    Boucle d'entraînement avec prise en charge des contraintes PINN.
    Intègre l'early stopping et la restauration du meilleur état.
    """
    def __init__(self, model, config, pinns=None):
        self.model = model
        self.config = config
        self.pinns = pinns if pinns else []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        for p in self.pinns:
            if isinstance(p, nn.Module):
                p.to(self.device)

        # Optimiseur commun : paramètres du GRU-CNN + paramètres appris des PINNs
        self.params = list(model.parameters())
        for p in self.pinns:
            if isinstance(p, nn.Module):
                self.params.extend(list(p.parameters()))

        self.optimizer = torch.optim.Adam(self.params, lr=config.LEARNING_RATE)

    def train(self, train_loader, val_loader, feature_cols, target_names, scaler_y):
        print(f"Device: {self.device}")

        # Poids par cible pour la MSE pondérée (uniformes par défaut)
        w = np.ones(len(target_names), dtype=float)
        w = w / (np.mean(w) + 1e-12)

        best_val = float('inf')
        best_state = None
        best_pinn_states = {}
        patience_cnt = 0

        # Métadonnées partagées avec les PINNs à chaque batch
        meta = {
            'feature_cols': feature_cols,
            'target_names': target_names,
            'scaler_y': scaler_y,
            'device': self.device
        }

        for epoch in range(1, self.config.N_EPOCHS + 1):
            # --- Entraînement ---
            self.model.train()
            train_losses = []
            train_pinn_losses = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(xb)

                # Perte RNN (MSE pondérée)
                loss = weighted_mse_loss(out, yb, w) * self.config.RNN_LOSS_WEIGHT

                # Perte PINN (somme des résidus physiques)
                pinn_loss_sum = 0.0
                for pinn in self.pinns:
                    l = pinn.compute_pinn_loss(xb, out, meta)
                    if l is not None:
                        pinn_loss_sum = pinn_loss_sum + l
                        train_pinn_losses.append(l.item())

                if isinstance(pinn_loss_sum, torch.Tensor):
                    loss = loss + (pinn_loss_sum * self.config.PINN_LOSS_WEIGHT)
                elif pinn_loss_sum > 0:
                    loss = loss + (pinn_loss_sum * self.config.PINN_LOSS_WEIGHT)

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses)) if train_losses else float('nan')
            avg_pinn_train = float(np.mean(train_pinn_losses)) if train_pinn_losses else 0.0

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_pinn_losses = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    out = self.model(xb)
                    loss = weighted_mse_loss(out, yb, w) * self.config.RNN_LOSS_WEIGHT

                    pinn_loss_sum = 0.0
                    for pinn in self.pinns:
                        l = pinn.compute_pinn_loss(xb, out, meta)
                        if l is not None:
                            pinn_loss_sum = pinn_loss_sum + l
                            val_pinn_losses.append(l.item())

                    if isinstance(pinn_loss_sum, torch.Tensor):
                        loss = loss + (pinn_loss_sum * self.config.PINN_LOSS_WEIGHT)
                    elif pinn_loss_sum > 0:
                        loss = loss + (pinn_loss_sum * self.config.PINN_LOSS_WEIGHT)

                    val_losses.append(loss.item())

            avg_val = float(np.mean(val_losses)) if val_losses else float('nan')
            avg_pinn_val = float(np.mean(val_pinn_losses)) if val_pinn_losses else 0.0

            print(f"Epoch {epoch}/{self.config.N_EPOCHS} — train: {avg_train:.6f} val: {avg_val:.6f} (PINN tr: {avg_pinn_train:.6f} val: {avg_pinn_val:.6f})")

            # --- Early stopping ---
            if not np.isnan(avg_val) and avg_val < best_val - 1e-12:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_pinn_states = {}
                for i, pinn in enumerate(self.pinns):
                    if isinstance(pinn, nn.Module):
                        best_pinn_states[i] = {k: v.cpu().clone() for k, v in pinn.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= self.config.PATIENCE:
                print(f"Early stopping déclenché. Meilleure perte val={best_val:.6f}.")
                break

        # Restauration du meilleur état
        if best_state is not None:
            self.model.load_state_dict(best_state)
            for i, pinn in enumerate(self.pinns):
                if isinstance(pinn, nn.Module) and i in best_pinn_states:
                    pinn.load_state_dict(best_pinn_states[i])
