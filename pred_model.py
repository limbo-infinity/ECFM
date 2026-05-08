import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from helper import transform_x


# xi        # [batch, obs_dim]
# da_true   # [batch, T, K]   day-ahead prices lam
# rt_true   # [batch, T, K]   real-time prices pi
# x         # [batch, T, K]   generated bids

## Included the penalalization of the bidding prices if they go outside of the 
## bounds set by the prices of the day
def trading_energy(
    x,
    da_price,
    rt_price,
    low,
    up,
    budget,
    alpha=20.0,
    penalty_weight=100.0,
):
    """
    x:        [batch, T, K] generated bid decisions
    da_price: [batch, T, K] lambda prices
    rt_price: [batch, T, K] pi prices
    budget: scalar or [batch, T]
    """
    # Transform x to be within the bounds of the bidding prices
    x = transform_x(x, low, up)
    

    # Smooth approximation of 1{x >= lambda}
    cleared_prob = torch.sigmoid(alpha * (x - da_price))

    # Payoff: sum_t (pi_t - lambda_t)^T 1{x_t >= lambda_t}
    payoff = ((rt_price - da_price) * cleared_prob).sum(dim=(1, 2))

    # Convert maximization of payoff into minimization
    base_energy = -payoff

    # # Constraint 1: x >= 0
    # nonnegative_penalty = F.relu(-x).sum(dim=(1, 2))

    # Constraint 2: ||x_t||_1 <= B for each day t
    daily_l1 = x.abs().sum(dim=-1)  # [batch, T]
    budget_penalty = F.relu(daily_l1 - budget).sum(dim=1)

    ## Removed the below zero penalty because the x already satisfies it
    total_energy = base_energy + penalty_weight * (
        + budget_penalty
    )

    return total_energy.mean(), {
        "mean_payoff": payoff.mean().item(),
        "mean_base_energy": base_energy.mean().item(),
        "mean_budget_penalty": budget_penalty.mean().item(),
        "mean_total_energy": total_energy.mean().item(),
    }


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


def build_train_loader(
    xi_train,
    da_train,
    rt_train,
    batch_size=128,
    shuffle=True,
):
    """
    Build batches of (xi, da_true, rt_true) for prediction-model training.
    """
    xi_train = torch.as_tensor(xi_train, dtype=torch.float32)
    da_train = torch.as_tensor(da_train, dtype=torch.float32)
    rt_train = torch.as_tensor(rt_train, dtype=torch.float32)

    if xi_train.size(0) != da_train.size(0) or xi_train.size(0) != rt_train.size(0):
        raise ValueError("xi_train, da_train, and rt_train must have the same sample count.")

    if da_train.shape != rt_train.shape:
        raise ValueError("da_train and rt_train must have matching [samples, T, K] shapes.")

    train_dataset = TensorDataset(xi_train, da_train, rt_train)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


def train_prediction_model(
    g_phi,
    train_loader,
    optimizer,
    device,
    num_epochs=20,
):
    """
    Train the prediction model on batches of (xi, da_true, rt_true).
    """
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


if __name__ == "__main__":
    # Placeholder hyperparameters. Match these to your actual data shapes.
    batch_size = 128
    obs_dim = 32
    hidden_dim = 256
    learning_rate = 1e-3
    num_epochs = 20
    T = 100
    K = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    g_phi = PricePredictor(obs_dim=obs_dim, hidden_dim=hidden_dim, T=T, K=K)
    g_phi.to(device)
    print(g_phi)

    optimizer = torch.optim.Adam(g_phi.parameters(), lr=learning_rate)

    num_train_samples = 1024
    xi_train = torch.randn(num_train_samples, obs_dim)
    da_train = torch.randn(num_train_samples, T, K)
    rt_train = torch.randn(num_train_samples, T, K)

    train_loader = build_train_loader(
        xi_train=xi_train,
        da_train=da_train,
        rt_train=rt_train,
        batch_size=batch_size,
    )

    train_prediction_model(
        g_phi=g_phi,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )
