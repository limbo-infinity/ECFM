import time

import numpy as np
import torch
import torch.nn.functional as F


def _as_float(value):
    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)


def predict_spread(model, xi):
    output = model(xi)
    if isinstance(output, tuple):
        da_pred, rt_pred = output
        return da_pred - rt_pred
    return output


def project_l1_budget(positions, budget, eps=1e-8):
    l1 = positions.abs().sum(dim=-1, keepdim=True)
    budget_tensor = torch.as_tensor(
        budget,
        dtype=positions.dtype,
        device=positions.device,
    )
    if budget_tensor.ndim == 0:
        budget_tensor = budget_tensor.reshape(1, 1, 1)
    elif budget_tensor.ndim == 1:
        budget_tensor = budget_tensor.reshape(-1, 1, 1)
    elif budget_tensor.ndim == 2:
        budget_tensor = budget_tensor.unsqueeze(-1)

    scale = torch.minimum(torch.ones_like(l1), budget_tensor / (l1 + eps))
    return positions * scale


def positions_from_spread(spread, q_max=1.0, budget=50.0, temperature=25.0):
    raw_positions = q_max * torch.tanh(spread / temperature)
    return project_l1_budget(raw_positions, budget=budget)


def simulate_positions(positions, da_true, rt_true, budget):
    true_spread = da_true - rt_true
    profit = (positions * true_spread).sum(dim=(1, 2))
    l1_usage = positions.abs().sum(dim=-1)
    budget_violation = F.relu(l1_usage - budget)
    return {
        "profit": profit,
        "true_spread": true_spread,
        "l1_usage": l1_usage,
        "budget_violation": budget_violation,
    }


def decision_metrics(profit, budget_violation, l1_usage, annualization=252):
    profit_np = profit.detach().cpu().numpy()
    mean_profit = float(profit_np.mean())
    std_profit = float(profit_np.std(ddof=1)) if profit_np.size > 1 else 0.0
    sharpe = 0.0
    if std_profit > 0:
        sharpe = mean_profit / std_profit * annualization ** 0.5

    return {
        "num_days": int(profit_np.size),
        "total_profit": float(profit_np.sum()),
        "mean_daily_profit": mean_profit,
        "std_daily_profit": std_profit,
        "annualized_sharpe": float(sharpe),
        "min_daily_profit": float(profit_np.min()),
        "max_daily_profit": float(profit_np.max()),
        "mean_budget_violation": _as_float(budget_violation.mean()),
        "max_budget_violation": _as_float(budget_violation.max()),
        "mean_l1_usage": _as_float(l1_usage.mean()),
        "max_l1_usage": _as_float(l1_usage.max()),
    }


def evaluate_spread_policy(
    model,
    data_loader,
    device,
    q_max=1.0,
    budget=50.0,
    temperature=25.0,
):
    model.eval()

    predicted_spreads = []
    true_spreads = []
    decisions = []
    profits = []
    l1_usages = []
    violations = []
    oracle_decisions = []
    oracle_profits = []
    zero_profits = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for xi, da_true, rt_true in data_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            spread_pred = predict_spread(model, xi)
            decision = positions_from_spread(
                spread_pred,
                q_max=q_max,
                budget=budget,
                temperature=temperature,
            )
            sim = simulate_positions(decision, da_true, rt_true, budget=budget)

            oracle_decision = positions_from_spread(
                sim["true_spread"],
                q_max=q_max,
                budget=budget,
                temperature=temperature,
            )
            oracle_sim = simulate_positions(
                oracle_decision,
                da_true,
                rt_true,
                budget=budget,
            )

            predicted_spreads.append(spread_pred.detach().cpu())
            true_spreads.append(sim["true_spread"].detach().cpu())
            decisions.append(decision.detach().cpu())
            profits.append(sim["profit"].detach().cpu())
            l1_usages.append(sim["l1_usage"].detach().cpu())
            violations.append(sim["budget_violation"].detach().cpu())
            oracle_decisions.append(oracle_decision.detach().cpu())
            oracle_profits.append(oracle_sim["profit"].detach().cpu())
            zero_profits.append(torch.zeros_like(sim["profit"]).detach().cpu())

    elapsed_seconds = time.perf_counter() - start_time

    predicted_spread = torch.cat(predicted_spreads, dim=0)
    true_spread = torch.cat(true_spreads, dim=0)
    decision = torch.cat(decisions, dim=0)
    profit = torch.cat(profits, dim=0)
    l1_usage = torch.cat(l1_usages, dim=0)
    violation = torch.cat(violations, dim=0)
    oracle_decision = torch.cat(oracle_decisions, dim=0)
    oracle_profit = torch.cat(oracle_profits, dim=0)
    zero_profit = torch.cat(zero_profits, dim=0)

    spread_error = predicted_spread - true_spread
    metrics = {
        "spread_mse": _as_float((spread_error**2).mean()),
        "spread_mae": _as_float(spread_error.abs().mean()),
        "latency_seconds": float(elapsed_seconds),
        "latency_ms_per_day": float(elapsed_seconds * 1000.0 / max(len(profit), 1)),
    }
    metrics.update(
        {
            f"policy_{key}": value
            for key, value in decision_metrics(profit, violation, l1_usage).items()
        }
    )
    metrics.update(
        {
            f"oracle_{key}": value
            for key, value in decision_metrics(
                oracle_profit,
                torch.zeros_like(violation),
                oracle_decision.abs().sum(dim=-1),
            ).items()
        }
    )
    metrics.update(
        {
            f"zero_{key}": value
            for key, value in decision_metrics(
                zero_profit,
                torch.zeros_like(violation),
                torch.zeros_like(l1_usage),
            ).items()
        }
    )

    arrays = {
        "predicted_spread": predicted_spread.numpy(),
        "true_spread": true_spread.numpy(),
        "decision": decision.numpy(),
        "profit": profit.numpy(),
        "l1_usage": l1_usage.numpy(),
        "budget_violation": violation.numpy(),
        "oracle_decision": oracle_decision.numpy(),
        "oracle_profit": oracle_profit.numpy(),
    }
    return metrics, arrays
