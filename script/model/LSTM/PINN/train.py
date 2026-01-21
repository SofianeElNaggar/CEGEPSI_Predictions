# train.py
import math
import torch
import numpy as np

from model import weighted_mse_loss

def train_model(model, loader_train, loader_val, n_epochs, optimizer, component_weights, patience, device):
    best_val = float('inf')
    best_state = None
    patience_cnt = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = weighted_mse_loss(out, yb, component_weights)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses)) if train_losses else float('nan')

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = weighted_mse_loss(out, yb, component_weights)
                val_losses.append(loss.item())
        avg_val = float(np.mean(val_losses)) if val_losses else float('nan')

        print(f"Epoch {epoch}/{n_epochs} — train loss: {avg_train:.6f}  val loss: {avg_val:.6f}")

        if not math.isnan(avg_val) and avg_val < best_val - 1e-12:
            best_val = avg_val
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            print(f"Early stopping triggered (patience={patience}). Restauration des meilleurs poids (val loss={best_val:.6f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
