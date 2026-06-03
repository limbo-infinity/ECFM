import torch
import torch.nn.functional as F


def transform_x(x, low, up):
    """Transform an unconstrained raw value into a bounded bid price."""
    return low + (up - low) * torch.sigmoid(x)


def transform_q(raw_q, q_max):
    return q_max * torch.tanh(raw_q)


def transform_q_bid_decision(raw_decision, q_max, low, up):
    if raw_decision.ndim != 4 or raw_decision.shape[-1] != 2:
        raise ValueError("raw_decision must have shape [batch, T, K, 2].")

    raw_q = raw_decision[..., 0]
    raw_bid = raw_decision[..., 1]
    q = transform_q(raw_q, q_max=q_max)
    bid_price = transform_x(raw_bid, low=low, up=up)
    return q, bid_price


def fixed_q_from_spread(spread, q_max):
    positive_q = torch.full_like(spread, q_max)
    negative_q = torch.full_like(spread, -q_max)
    return torch.where(spread < 0, positive_q, negative_q)


def project_l1_budget(values, budget, eps=1e-8):
    daily_l1 = values.abs().sum(dim=-1, keepdim=True)
    budget_tensor = torch.as_tensor(
        budget,
        dtype=values.dtype,
        device=values.device,
    )
    if budget_tensor.ndim == 0:
        budget_tensor = budget_tensor.reshape(1, 1, 1)
    elif budget_tensor.ndim == 1:
        budget_tensor = budget_tensor.reshape(-1, 1, 1)
    elif budget_tensor.ndim == 2:
        budget_tensor = budget_tensor.unsqueeze(-1)
    else:
        raise ValueError("budget must be a scalar, [batch], or [batch, T] tensor.")

    scale = torch.minimum(torch.ones_like(daily_l1), budget_tensor / (daily_l1 + eps))
    return values * scale


def _budget_like_daily_l1(budget, daily_l1):
    budget_tensor = torch.as_tensor(
        budget,
        dtype=daily_l1.dtype,
        device=daily_l1.device,
    )
    if budget_tensor.ndim == 0:
        return budget_tensor
    if budget_tensor.ndim == 1:
        return budget_tensor.reshape(-1, 1)
    if budget_tensor.ndim == 2:
        return budget_tensor
    raise ValueError("budget must be a scalar, [batch], or [batch, T] tensor.")


def get_model_device(model):
    return next(model.parameters()).device


def get_training_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def trading_energy(
    x,
    da_price,
    rt_price,
    low,
    up,
    budget,
    alpha=20.0,
    penalty_weight=100.0,
    return_decisions=False,
):
    """
    x:        [batch, T, K] generated bid decisions
    da_price: [batch, T, K] lambda prices
    rt_price: [batch, T, K] pi prices
    budget: scalar or [batch, T]
    """

    bid_price = transform_x(x, low, up)
    cleared_prob = torch.sigmoid(alpha * (bid_price - da_price))
    payoff = ((rt_price - da_price) * cleared_prob).sum(dim=(1, 2))
    base_energy = -payoff


    daily_l1 = bid_price.abs().sum(dim=-1)  # [batch, T]
    budget_penalty = F.relu(
        daily_l1 - _budget_like_daily_l1(budget, daily_l1)
    ).sum(dim=1)

    total_energy = base_energy + penalty_weight * (budget_penalty)

    logs = {
        "mean_payoff": payoff.mean().item(),
        "mean_base_energy": base_energy.mean().item(),
        "mean_budget_penalty": budget_penalty.mean().item(),
        "mean_total_energy": total_energy.mean().item(),
    }
    if return_decisions:
        logs["bid_price"] = bid_price
        logs["cleared_prob"] = cleared_prob

    return total_energy, logs


def q_bid_trading_energy(
    x,
    da_price,
    rt_price,
    low,
    up,
    budget,
    q_max=1.0,
    alpha=1.0,
    penalty_weight=100.0,
    return_decisions=False,
):
    """
    Energy for option C: the model generates both signed quantity q and bid price.

    x: [batch, T, K, 2], where x[..., 0] is raw q and x[..., 1] is raw bid price.
    q: transformed to [-q_max, q_max]
    bid_price: transformed to [low, up]
    """
    q, bid_price = transform_q_bid_decision(
        raw_decision=x,
        q_max=q_max,
        low=low,
        up=up,
    )
    cleared_prob = torch.sigmoid(alpha * q * (bid_price - da_price))
    payoff = (q * (rt_price - da_price) * cleared_prob).sum(dim=(1, 2))
    base_energy = -payoff

    daily_l1 = q.abs().sum(dim=-1)
    budget_penalty = F.relu(
        daily_l1 - _budget_like_daily_l1(budget, daily_l1)
    ).sum(dim=1)
    total_energy = base_energy + penalty_weight * budget_penalty

    logs = {
        "mean_payoff": payoff.mean().item(),
        "mean_base_energy": base_energy.mean().item(),
        "mean_budget_penalty": budget_penalty.mean().item(),
        "mean_total_energy": total_energy.mean().item(),
        "mean_l1_usage": daily_l1.mean().item(),
    }
    if return_decisions:
        logs["q"] = q
        logs["bid_price"] = bid_price
        logs["cleared_prob"] = cleared_prob

    return total_energy, logs


def fixed_q_bid_trading_energy(
    x,
    da_price,
    rt_price,
    low,
    up,
    budget,
    q,
    alpha=1.0,
    lam=0.1,
    penalty_weight=100.0,
    return_decisions=False,
):
    """
    Energy for the simpler fixed-q policy.

    x: [batch, T, K], raw bid price generated by the flow.
    q: [batch, T, K], fixed signed direction/quantity from predicted spread.
    """
    raw_bid_price = transform_x(x, low, up)
    bid_price = project_l1_budget(raw_bid_price, budget=budget)
    cleared_prob = torch.sigmoid(alpha * q * (bid_price - da_price))
    payoff = (q * (rt_price - da_price) * cleared_prob).sum(dim=(1, 2))
    base_energy = -payoff

    daily_l1 = bid_price.abs().sum(dim=-1)
    budget_penalty = torch.zeros_like(base_energy)
    total_energy = base_energy + penalty_weight * budget_penalty 

    logs = {
        "mean_payoff": payoff.mean().item(),
        "mean_base_energy": base_energy.mean().item(),
        "mean_budget_penalty": budget_penalty.mean().item(),
        "mean_total_energy": total_energy.mean().item(),
        "mean_l1_usage": daily_l1.mean().item(),
    }
    if return_decisions:
        logs["q"] = q
        logs["raw_bid_price"] = raw_bid_price
        logs["bid_price"] = bid_price
        logs["cleared_prob"] = cleared_prob

    return total_energy, logs
