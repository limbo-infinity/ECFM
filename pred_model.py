import torch
import torch.nn.functional as F
import torch.nn as nn
from helper import transform_x
import torch.optim as optim
import tdqm
import matplotlib.pyplot as plt


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
    x = transform_x(x, low, up)

    # Smooth approximation of 1{x >= lambda}
    cleared_prob = torch.sigmoid(alpha * (x - da_price))

    # Payoff: sum_t (pi_t - lambda_t)^T 1{x_t >= lambda_t}
    payoff = ((rt_price - da_price) * cleared_prob).sum(dim=(1, 2))

    # Convert maximization of payoff into minimization
    base_energy = -payoff

    # Constraint 1: x >= 0
    nonnegative_penalty = F.relu(-x).sum(dim=(1, 2))

    # Constraint 2: ||x_t||_1 <= B for each day t
    daily_l1 = x.abs().sum(dim=-1)  # [batch, T]
    budget_penalty = F.relu(daily_l1 - budget).sum(dim=1)

    total_energy = base_energy + penalty_weight * (
        nonnegative_penalty + budget_penalty
    )

    return total_energy.mean(), {
        "mean_payoff": payoff.mean().item(),
        "mean_base_energy": base_energy.mean().item(),
        "mean_nonnegative_penalty": nonnegative_penalty.mean().item(),
        "mean_budget_penalty": budget_penalty.mean().item(),
    }


## PREDICTION MODEL g_\phi(xi)
class PricePredictor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, T, K):
        super().__init__()
        self.T = T
        self.K = K

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * T * K),
        )

    def forward(self, xi):
        out = self.net(xi)  # [batch, 2*T*K]

        batch_size = xi.shape[0]
        out = out.view(batch_size, self.T, 2, self.K)

        da_pred = out[:, :, 0, :]
        rt_pred = out[:, :, 1, :]

        return da_pred, rt_pred
    
    
## Have to actually instantiate the model 

# Hyperparameters of prediction model 
batch_size = 128
obs_dim = 32
hidden_dim = 256
learning_rate = 1e-3
num_epochs = 20

# Number of trading days and number of options 
# - change this later just add some dummy numbers
T = 100
K = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

g_phi = PricePredictor(obs_dim=obs_dim, hidden_dim=hidden_dim, T=T, K=K)
print(g_phi)


# Optimizaer for prediction model 
optimizer = torch.optim.Adam(g_phi.parameters(), lr=learning_rate)

## Not too sure what the dims of xi are -> check this/ just change later to suit
## the implementation
def pred_loss(g_phi, xi, c_true):
    """
    xi:       [batch, obs_dim] observations of past prices
    c_true: [batch, 2*T*K] true RT and DA prices
    g_phi: prediction model 
    """

    c_pred = g_phi(xi)

    loss = torch.norm(c_pred - c_true, p=2, dim=1) ** 2 # Get the norm across all days and options for both DA and RT
    
    return loss.mean()

# da_pred, rt_pred = g_phi(xi)

## Training Loop
train_loss_history = []

g_phi.train()

for epoch in range(num_epochs):
    epoch_total_loss = 0.0


    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for x, _ in progress_bar:
        x = x.to(device)

        optimizer.zero_grad()

        x_recon, mu, logvar = vae(x)
        total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

        total_loss.backward()
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()

        # Show average losses per image so values are easier to interpret
        batch_size_curr = x.size(0)
        progress_bar.set_postfix(
            total_loss=f"{total_loss.item() / batch_size_curr:.4f}",
            recon_loss=f"{recon_loss.item() / batch_size_curr:.4f}",
            kl_loss=f"{kl_loss.item() / batch_size_curr:.4f}",
        )

    avg_total_loss = epoch_total_loss / len(train_loader.dataset)
    avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
    avg_kl_loss = epoch_kl_loss / len(train_loader.dataset)

    train_loss_history.append(avg_total_loss)
    train_recon_history.append(avg_recon_loss)
    train_kl_history.append(avg_kl_loss)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"avg total loss: {avg_total_loss:.4f} | "
        f"avg recon loss: {avg_recon_loss:.4f} | "
        f"avg kl loss: {avg_kl_loss:.4f}"
    )