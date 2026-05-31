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

## PREDICTION MODEL g_phi(xi)
class PricePredictor(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        T,
        K,
        num_hidden_layers=2,
        output_mode="prices",
    ):
        super().__init__()
        if output_mode not in {"prices", "spread"}:
            raise ValueError("output_mode must be either 'prices' or 'spread'.")

        self.T = T
        self.K = K
        self.num_hidden_layers = num_hidden_layers
        self.output_mode = output_mode

        layers = []
        input_dim = obs_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
        output_dim = T * K if output_mode == "spread" else 2 * T * K
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xi):
        batch_size = xi.shape[0]
        out = self.net(xi)

        if self.output_mode == "spread":
            return out.view(batch_size, self.T, self.K)

        out = out.view(batch_size, self.T, 2, self.K)

        da_pred = out[:, :, 0, :]
        rt_pred = out[:, :, 1, :]

        return da_pred, rt_pred
    
    
def price_mse_loss(g_phi, xi, da_true, rt_true):
    """
    xi:       [batch, obs_dim] observations of past prices
    da_true: [batch, T, K]
    rt_true: [batch, T, K]
    g_phi: prediction model returning (da_pred, rt_pred)
    """
    da_pred, rt_pred = g_phi(xi)

    loss_da = F.mse_loss(da_pred, da_true)
    loss_rt = F.mse_loss(rt_pred, rt_true)
    return loss_da + loss_rt


def spread_mse_loss(g_phi, xi, da_true, rt_true):
    """
    Train the model to predict the spread DA - RT directly.

    spread_pred: [batch, T, K]
    spread_true: [batch, T, K]
    """
    model_output = g_phi(xi)
    if isinstance(model_output, tuple):
        da_pred, rt_pred = model_output
        spread_pred = da_pred - rt_pred
    else:
        spread_pred = model_output
    spread_true = da_true - rt_true
    return F.mse_loss(spread_pred, spread_true)


def pred_loss(g_phi, xi, da_true, rt_true, loss_mode="prices"):
    if loss_mode == "prices":
        return price_mse_loss(g_phi, xi, da_true, rt_true)
    if loss_mode == "spread":
        return spread_mse_loss(g_phi, xi, da_true, rt_true)
    raise ValueError("loss_mode must be either 'prices' or 'spread'.")




def train_prediction_model(
    g_phi,
    train_loader,
    optimizer,
    device=None,
    num_epochs=20,
    val_loader=None,
    verbose=True,
    loss_mode="prices",
):
    """
    Train the prediction model on batches of (xi, da_true, rt_true).
    """
    if device is None:
        device = next(g_phi.parameters()).device

    losses = {"train": [], "val": []}

    for epoch in range(num_epochs):
        g_phi.train()
        epoch_loss = 0.0
        num_samples = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            rt_true = rt_true.to(device)
            da_true = da_true.to(device)

            optimizer.zero_grad()

            loss = pred_loss(g_phi, xi, da_true, rt_true, loss_mode=loss_mode)

            loss.backward()
            optimizer.step()

            batch_size_curr = xi.size(0)
            epoch_loss += loss.item() * batch_size_curr
            num_samples += batch_size_curr

        if num_samples == 0:
            raise ValueError("train_loader did not provide any training batches.")

        avg_loss = epoch_loss / num_samples
        losses["train"].append(avg_loss)

        message = f"Epoch {epoch + 1}/{num_epochs} | prediction loss: {avg_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate_prediction_model(
                g_phi,
                val_loader,
                device=device,
                loss_mode=loss_mode,
            )
            losses["val"].append(val_loss)
            message += f" | val loss: {val_loss:.4f}"
        if verbose:
            print(message)

    return losses


def evaluate_prediction_model(g_phi, val_loader, device=None, loss_mode="prices"):
    if device is None:
        device = next(g_phi.parameters()).device

    g_phi.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for xi, da_true, rt_true in val_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            loss = pred_loss(g_phi, xi, da_true, rt_true, loss_mode=loss_mode)
            batch_size_curr = xi.size(0)
            total_loss += loss.item() * batch_size_curr
            num_samples += batch_size_curr

    if num_samples == 0:
        raise ValueError("val_loader did not provide any validation batches.")

    return total_loss / num_samples
