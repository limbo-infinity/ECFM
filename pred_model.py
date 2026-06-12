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


class FMTSPricePredictor(nn.Module):
    """
    Conditional FM-TS-style predictor.

    It learns a rectified flow from Gaussian noise to the next-day DA/RT price
    tensor, conditioned on the historical price vector xi.
    """

    def __init__(
        self,
        obs_dim,
        hidden_dim,
        T,
        K,
        num_hidden_layers=3,
        output_mode="prices",
        num_prediction_samples=4,
        num_prediction_steps=16,
        time_sampling="logit_normal",
    ):
        super().__init__()
        if output_mode not in {"prices", "spread"}:
            raise ValueError("output_mode must be either 'prices' or 'spread'.")

        self.T = T
        self.K = K
        self.output_mode = output_mode
        self.num_hidden_layers = num_hidden_layers
        self.num_prediction_samples = num_prediction_samples
        self.num_prediction_steps = num_prediction_steps
        self.time_sampling = time_sampling
        self.target_channels = 2 if output_mode == "prices" else 1
        self.target_dim = T * K * self.target_channels

        context_layers = []
        input_dim = obs_dim
        for _ in range(num_hidden_layers):
            context_layers.append(nn.Linear(input_dim, hidden_dim))
            context_layers.append(nn.GELU())
            input_dim = hidden_dim
        self.context_encoder = nn.Sequential(*context_layers)

        velocity_layers = []
        input_dim = self.target_dim + hidden_dim + 3
        for _ in range(num_hidden_layers):
            velocity_layers.append(nn.Linear(input_dim, hidden_dim))
            velocity_layers.append(nn.GELU())
            input_dim = hidden_dim
        velocity_layers.append(nn.Linear(input_dim, self.target_dim))
        self.velocity_net = nn.Sequential(*velocity_layers)

        norm_shape = (1, 1, 1, self.target_channels)
        self.register_buffer("target_mean", torch.zeros(norm_shape))
        self.register_buffer("target_std", torch.ones(norm_shape))

    def _target_from_truth(self, da_true, rt_true):
        if self.output_mode == "prices":
            return torch.stack([da_true, rt_true], dim=-1)
        return (da_true - rt_true).unsqueeze(-1)

    def fit_normalization(self, data_loader, device=None):
        if device is None:
            device = next(self.parameters()).device

        total = None
        total_sq = None
        count = 0
        with torch.no_grad():
            for _, da_true, rt_true in data_loader:
                da_true = da_true.to(device)
                rt_true = rt_true.to(device)
                target = self._target_from_truth(da_true, rt_true)
                reduce_dims = tuple(range(target.ndim - 1))
                batch_sum = target.sum(dim=reduce_dims)
                batch_sum_sq = (target**2).sum(dim=reduce_dims)
                batch_count = target[..., 0].numel()

                total = batch_sum if total is None else total + batch_sum
                total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
                count += batch_count

        if count == 0:
            raise ValueError("data_loader did not provide any targets.")

        mean = total / count
        var = (total_sq / count) - mean**2
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        self.target_mean.copy_(mean.reshape(1, 1, 1, self.target_channels))
        self.target_std.copy_(std.reshape(1, 1, 1, self.target_channels))

    def _normalize_target(self, target):
        return (target - self.target_mean) / self.target_std

    def _denormalize_target(self, target):
        return target * self.target_std + self.target_mean

    def _sample_t(self, batch_size, device):
        if self.time_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(batch_size, 1, device=device))
        return torch.rand(batch_size, 1, device=device)

    def _velocity(self, z, t, xi):
        batch_size = z.shape[0]
        context = self.context_encoder(xi)
        t_features = torch.cat(
            [
                t,
                torch.sin(2.0 * torch.pi * t),
                torch.cos(2.0 * torch.pi * t),
            ],
            dim=-1,
        )
        model_input = torch.cat(
            [z.reshape(batch_size, -1), context, t_features],
            dim=-1,
        )
        velocity = self.velocity_net(model_input)
        return velocity.reshape(batch_size, self.T, self.K, self.target_channels)

    def rectified_flow_loss(self, xi, da_true, rt_true):
        target = self._normalize_target(self._target_from_truth(da_true, rt_true))
        batch_size = xi.shape[0]

        z0 = torch.randn_like(target)
        t = self._sample_t(batch_size, xi.device)
        t_view = t.reshape(batch_size, 1, 1, 1)
        zt = (1.0 - t_view) * z0 + t_view * target
        target_velocity = target - z0
        pred_velocity = self._velocity(zt, t, xi)
        return F.mse_loss(pred_velocity, target_velocity)

    def sample(
        self,
        xi,
        num_samples=None,
        num_steps=None,
        differentiable=False,
    ):
        num_samples = self.num_prediction_samples if num_samples is None else num_samples
        num_steps = self.num_prediction_steps if num_steps is None else num_steps
        batch_size = xi.shape[0]
        device = xi.device

        xi_rep = xi.unsqueeze(1).repeat(1, num_samples, 1)
        xi_rep = xi_rep.reshape(batch_size * num_samples, -1)
        z = torch.randn(
            batch_size * num_samples,
            self.T,
            self.K,
            self.target_channels,
            device=device,
        )

        grad_context = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_context:
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t_value = step / num_steps
                t = torch.full(
                    (batch_size * num_samples, 1),
                    t_value,
                    device=device,
                )
                z = z + dt * self._velocity(z, t, xi_rep)

        z = self._denormalize_target(z)
        return z.reshape(
            batch_size,
            num_samples,
            self.T,
            self.K,
            self.target_channels,
        )

    def forward(self, xi):
        samples = self.sample(
            xi,
            num_samples=self.num_prediction_samples,
            num_steps=self.num_prediction_steps,
            differentiable=torch.is_grad_enabled(),
        )
        prediction = samples.mean(dim=1)

        if self.output_mode == "spread":
            return prediction[..., 0]

        da_pred = prediction[..., 0]
        rt_pred = prediction[..., 1]
        return da_pred, rt_pred


class DirectBidMLP(nn.Module):
    def __init__(self, T, K, hidden_dim, num_hidden_layers=2):
        super().__init__()
        self.T = T
        self.K = K
        self.num_hidden_layers = num_hidden_layers

        layers = []
        input_dim = 2 * T * K
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, T * K))
        self.net = nn.Sequential(*layers)

    def forward(self, da_cond, rt_cond):
        batch_size = da_cond.shape[0]
        cond = torch.cat(
            [
                da_cond.reshape(batch_size, -1),
                rt_cond.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        raw_bid = self.net(cond)
        return raw_bid.view(batch_size, self.T, self.K)
    
    
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


def train_fmts_prediction_model(
    g_phi,
    train_loader,
    optimizer,
    device=None,
    num_epochs=20,
    val_loader=None,
    verbose=True,
    loss_mode="spread",
):
    """
    Train an FMTSPricePredictor with rectified-flow loss.

    The validation curve reports the usual prediction MSE from sampled forecasts
    so it is directly comparable with the MLP predictor.
    """
    if device is None:
        device = next(g_phi.parameters()).device

    if hasattr(g_phi, "fit_normalization"):
        g_phi.fit_normalization(train_loader, device=device)

    losses = {"train": [], "val": []}

    for epoch in range(num_epochs):
        g_phi.train()
        epoch_loss = 0.0
        num_samples = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            optimizer.zero_grad()
            loss = g_phi.rectified_flow_loss(xi, da_true, rt_true)
            loss.backward()
            optimizer.step()

            batch_size_curr = xi.size(0)
            epoch_loss += loss.item() * batch_size_curr
            num_samples += batch_size_curr

        if num_samples == 0:
            raise ValueError("train_loader did not provide any training batches.")

        avg_loss = epoch_loss / num_samples
        losses["train"].append(avg_loss)

        message = f"Epoch {epoch + 1}/{num_epochs} | FM-TS flow loss: {avg_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate_prediction_model(
                g_phi,
                val_loader,
                device=device,
                loss_mode=loss_mode,
            )
            losses["val"].append(val_loss)
            message += f" | val {loss_mode} MSE: {val_loss:.4f}"
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
