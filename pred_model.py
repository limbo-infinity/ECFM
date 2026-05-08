import torch
import torch.nn.functional as F
import torch.nn as nn
from helper import transform_x
import torch.optim as optim
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
    # Transform x to be within the bounds of the bidding prices
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
g_phi.to(device) 
print(g_phi)


# Optimizaer for prediction model 
optimizer = torch.optim.Adam(g_phi.parameters(), lr=learning_rate)

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



## Training Loop
train_loss_dict = dict()
val_loss_dict = dict()
train_accuracy_dict = dict()
test_accuracy_dict = dict()

# Should this be put inside the training loop?
for epoch in range(num_epochs):
    g_phi.train()
    epoch_total_loss = 0.0

    # Can consider adding a training progress bar later using tqdm

    for xi, da_true, rt_true in train_loader:
        xi = xi.to(device)
        rt_true = rt_true.to(device)

        optimizer.zero_grad()

        loss = pred_loss(g_phi, xi, da_true, rt_true)

        loss.backward()
        optimizer.step()
