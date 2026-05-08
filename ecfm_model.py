import torch
import torch.nn as nn
from helper import transform_x
import torch.nn.functional as F

# v(x_t, t, c)
# We input z ~ N(0, I) conditioned on the DA and RT prices
class ConditionalFlow(nn.Module):
    def __init__(self, T, K, hidden_dim):
        super().__init__()
        self.T = T
        self.K = K

        x_dim = T * K
        cond_dim = 2 * T * K

        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, z, t, da_cond, rt_cond):
        """
        z:       [batch, T, K]
        t:       [batch, 1]
        da_cond: [batch, T, K]
        rt_cond: [batch, T, K]
        """

        batch_size = z.shape[0]

        z_flat = z.reshape(batch_size, -1)
        cond_flat = torch.cat(
            [da_cond.reshape(batch_size, -1),
             rt_cond.reshape(batch_size, -1)],
            dim=-1,
        )

        inp = torch.cat([z_flat, cond_flat, t], dim=-1)

        velocity = self.net(inp)
        velocity = velocity.reshape(batch_size, self.T, self.K)

        return velocity



    def rectified_flow_loss(self, z_1, da_cond, rt_cond):
        """
        Rectified-flow objective from noise z_0 to target sample z_1.

        z_1:     [batch, T, K]
        da_cond: [batch, T, K]
        rt_cond: [batch, T, K]
        """
        batch_size = z_1.shape[0]

        z_0 = torch.randn_like(z_1)
        t = z_1.new_empty(batch_size, 1).uniform_(0.0, 1.0)
        t_view = t.reshape(batch_size, 1, 1)

        z_t = (1.0 - t_view) * z_0 + t_view * z_1
        target_velocity = z_1 - z_0
        pred_velocity = self(z_t, t, da_cond, rt_cond)

        return ((pred_velocity - target_velocity) ** 2).mean()


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
    cleared_prob = torch.sigmoid(alpha * (x - da_price))
    payoff = ((rt_price - da_price) * cleared_prob).sum(dim=(1, 2))
    base_energy = -payoff

    # nonnegative_penalty = F.relu(-x).sum(dim=(1, 2))

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
    
    
## Energy fn is just the trading energy from (6)
def langevin_refine_trading(
    x,
    da_price,
    rt_price,
    budget,
    energy_fn,
    beta=1.0,
    step_size=1e-3,
    n_steps=20,
):
    x = x.clone().detach()

    for _ in range(n_steps):
        x.requires_grad_(True)

        energy, logs = energy_fn(x=x, da_price=da_price, rt_price=rt_price,
                                budget=budget,)

        grad_x = torch.autograd.grad(energy, x, create_graph=False)[0]

        noise = torch.randn_like(x)
        x_new = x - step_size * beta * grad_x + (2 * step_size) ** 0.5 * noise
        x = x_new.detach()

    return x