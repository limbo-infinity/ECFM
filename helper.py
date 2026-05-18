import torch
import torch.nn.functional as F


# Define a transform function that forces x to be between
# the upper and lower bounds of the biding prices
def transform_x(x, low, up):
    return low + (up - low) * torch.sigmoid(x)


def get_model_device(model):
    return next(model.parameters()).device


def get_training_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


## Included the penalalization of the bidding prices if they
# go outside of the bounds of the ISO

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


    daily_l1 = x.abs().sum(dim=-1)  # [batch, T]
    budget_penalty = F.relu(daily_l1 - budget).sum(dim=1)

    ## Removed the below zero penalty because the x already satisfies it
    total_energy = base_energy + penalty_weight * (budget_penalty)

    ## 
    return total_energy, {
        "mean_payoff": payoff.mean().item(),
        "mean_base_energy": base_energy.mean().item(),
        "mean_budget_penalty": budget_penalty.mean().item(),
        "mean_total_energy": total_energy.mean().item()}
