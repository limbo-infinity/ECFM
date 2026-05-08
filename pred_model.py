import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np



# xi        # [batch, obs_dim]
# da_true   # [batch, T, K]   day-ahead prices lam
# rt_true   # [batch, T, K]   real-time prices pi
# x         # [batch, T, K]   generated bids

## PREDICTION MODEL g_\phi(xi)
class PricePredictor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, T, K):
        super().__init__()
        self.T = T
        self.K = K

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * T * K),
        )

    def forward(self, xi):
        out = self.net(xi)  # [batch, 2*T*K]

        batch_size = xi.shape[0]
        out = out.view(batch_size, self.T, 2, self.K)

        da_pred = out[:, :, 0, :]
        rt_pred = out[:, :, 1, :]

        return da_pred, rt_pred
    
    
## Not too sure what the dims of xi are -> check this/ just change later to suit
## the implementation
def pred_loss(g_phi, xi, da_true, rt_true):
    """
    xi:       [batch, obs_dim] observations of past prices
    da_true: [batch, T, K]
    rt_true: [batch, T, K]
    g_phi: prediction model 
    """
    da_pred, rt_pred = g_phi(xi)

    loss_da = F.mse_loss(da_pred, da_true)
    loss_rt = F.mse_loss(rt_pred, rt_true)
    return loss_da + loss_rt 




def train_prediction_model(
    g_phi,
    train_loader,
    optimizer,
    device=None,
    num_epochs=20,
):
    """
    Train the prediction model on batches of (xi, da_true, rt_true).
    """
    if device is None:
        device = next(g_phi.parameters()).device

    train_loss_dict = {}

    for epoch in range(num_epochs):
        g_phi.train()
        epoch_loss = 0.0
        num_samples = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            rt_true = rt_true.to(device)
            da_true = da_true.to(device)

            optimizer.zero_grad()

            loss = pred_loss(g_phi, xi, da_true, rt_true)

            loss.backward()
            optimizer.step()

            batch_size_curr = xi.size(0)
            epoch_loss += loss.item() * batch_size_curr
            num_samples += batch_size_curr

        if num_samples == 0:
            raise ValueError("train_loader did not provide any training batches.")

        avg_loss = epoch_loss / num_samples
        train_loss_dict[epoch] = avg_loss

        print(f"Epoch {epoch + 1}/{num_epochs} | prediction loss: {avg_loss:.4f}")

    return train_loss_dict


