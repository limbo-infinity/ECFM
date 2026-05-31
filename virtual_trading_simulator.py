import time

import numpy as np
import torch
import torch.nn.functional as F

from ecfm_model import langevin_refine_trading
from helper import (
    fixed_q_bid_trading_energy,
    fixed_q_from_spread,
    project_l1_budget,
    q_bid_trading_energy,
    transform_q_bid_decision,
    transform_x,
)


def _as_float(value):
    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)


def predict_spread(model, xi):
    output = model(xi)
    if isinstance(output, tuple):
        da_pred, rt_pred = output
        return da_pred - rt_pred
    return output


def prediction_to_price_conditions(prediction_output):
    if isinstance(prediction_output, tuple):
        return prediction_output

    spread_pred = prediction_output
    da_pred = torch.zeros_like(spread_pred)
    rt_pred = -spread_pred
    return da_pred, rt_pred


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


def positions_from_spread(
    spread,
    q_max=1.0,
    budget=50.0,
    temperature=25.0,
    return_raw=False,
):
    raw_positions = q_max * torch.tanh(spread / temperature)
    projected_positions = project_l1_budget(raw_positions, budget=budget)
    if return_raw:
        return raw_positions, projected_positions
    return projected_positions


def binary_bids_from_spread(
    spread,
    q_max=1.0,
    budget=50.0,
    bid_low=-1000.0,
    bid_up=2000.0,
):
    direction = torch.where(spread < 0, 1.0, -1.0)
    raw_q = q_max * direction

    max_active = int(max(float(budget) // float(q_max), 0))
    max_active = min(max_active, spread.shape[-1])
    active = torch.zeros_like(spread, dtype=torch.bool)
    if max_active > 0:
        top_indices = spread.abs().topk(max_active, dim=-1).indices
        active.scatter_(-1, top_indices, True)

    actual_q = torch.where(active, raw_q, torch.zeros_like(raw_q))
    bid_price = torch.where(
        actual_q > 0,
        torch.full_like(actual_q, bid_up),
        torch.full_like(actual_q, bid_low),
    )
    return raw_q, actual_q, bid_price, active


def simulate_binary_virtual_bids(q, bid_price, da_true, rt_true, budget):
    true_spread = da_true - rt_true
    active = q != 0
    cleared_prob = torch.where(
        active,
        torch.sigmoid(q * (bid_price - da_true)),
        torch.zeros_like(q),
    )
    profit = (q * (rt_true - da_true) * cleared_prob).sum(dim=(1, 2))
    l1_usage = q.abs().sum(dim=-1)
    budget_violation = F.relu(l1_usage - budget)
    return {
        "profit": profit,
        "true_spread": true_spread,
        "cleared_prob": cleared_prob,
        "l1_usage": l1_usage,
        "budget_violation": budget_violation,
    }


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
    bid_low=-1000.0,
    bid_up=2000.0,
):
    model.eval()

    predicted_spreads = []
    true_spreads = []
    raw_decisions = []
    actual_decisions = []
    bid_prices = []
    cleared_probs = []
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
            raw_decision, actual_decision, bid_price, _ = binary_bids_from_spread(
                spread_pred,
                q_max=q_max,
                budget=budget,
                bid_low=bid_low,
                bid_up=bid_up,
            )
            sim = simulate_binary_virtual_bids(
                actual_decision,
                bid_price,
                da_true,
                rt_true,
                budget=budget,
            )

            _, oracle_actual_decision, oracle_bid_price, _ = binary_bids_from_spread(
                sim["true_spread"],
                q_max=q_max,
                budget=budget,
                bid_low=bid_low,
                bid_up=bid_up,
            )
            oracle_sim = simulate_binary_virtual_bids(
                oracle_actual_decision,
                oracle_bid_price,
                da_true,
                rt_true,
                budget=budget,
            )

            predicted_spreads.append(spread_pred.detach().cpu())
            true_spreads.append(sim["true_spread"].detach().cpu())
            raw_decisions.append(raw_decision.detach().cpu())
            actual_decisions.append(actual_decision.detach().cpu())
            bid_prices.append(bid_price.detach().cpu())
            cleared_probs.append(sim["cleared_prob"].detach().cpu())
            profits.append(sim["profit"].detach().cpu())
            l1_usages.append(sim["l1_usage"].detach().cpu())
            violations.append(sim["budget_violation"].detach().cpu())
            oracle_decisions.append(oracle_actual_decision.detach().cpu())
            oracle_profits.append(oracle_sim["profit"].detach().cpu())
            zero_profits.append(torch.zeros_like(sim["profit"]).detach().cpu())

    elapsed_seconds = time.perf_counter() - start_time

    predicted_spread = torch.cat(predicted_spreads, dim=0)
    true_spread = torch.cat(true_spreads, dim=0)
    raw_decision = torch.cat(raw_decisions, dim=0)
    actual_decision = torch.cat(actual_decisions, dim=0)
    bid_price = torch.cat(bid_prices, dim=0)
    cleared_prob = torch.cat(cleared_probs, dim=0)
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
        "mean_decision_adjustment": _as_float(
            (actual_decision - raw_decision).abs().mean()
        ),
        "max_decision_adjustment": _as_float(
            (actual_decision - raw_decision).abs().max()
        ),
        "mean_cleared_probability": _as_float(cleared_prob.mean()),
        "mean_active_cleared_probability": _as_float(
            cleared_prob[actual_decision != 0].mean()
            if (actual_decision != 0).any()
            else torch.tensor(0.0)
        ),
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
        "raw_decision": raw_decision.numpy(),
        "decision": actual_decision.numpy(),
        "actual_decision": actual_decision.numpy(),
        "bid_price": bid_price.numpy(),
        "cleared_prob": cleared_prob.numpy(),
        "profit": profit.numpy(),
        "l1_usage": l1_usage.numpy(),
        "budget_violation": violation.numpy(),
        "oracle_decision": oracle_decision.numpy(),
        "oracle_profit": oracle_profit.numpy(),
    }
    return metrics, arrays


def evaluate_q_bid_policy(
    prediction_model,
    ecfm_model,
    data_loader,
    device,
    low,
    up,
    budget,
    q_max=1.0,
    alpha=5.0,
    penalty_weight=100.0,
    num_samples=8,
    num_flow_steps=30,
    use_langevin=True,
    langevin_steps=30,
    langevin_step_size=1e-5,
):
    prediction_model.eval()
    ecfm_model.eval()

    predicted_das = []
    predicted_rts = []
    true_das = []
    true_rts = []
    raw_decisions = []
    q_decisions = []
    bid_prices = []
    cleared_probs = []
    selected_indices = []
    selected_surrogate_energies = []
    langevin_energy_traces = []
    profits = []
    l1_usages = []
    violations = []
    oracle_decisions = []
    oracle_profits = []
    zero_profits = []

    energy_kwargs = {
        "low": low,
        "up": up,
        "q_max": q_max,
        "alpha": alpha,
        "penalty_weight": penalty_weight,
    }

    start_time = time.perf_counter()
    for xi, da_true, rt_true in data_loader:
        xi = xi.to(device)
        da_true = da_true.to(device)
        rt_true = rt_true.to(device)

        with torch.no_grad():
            prediction_output = prediction_model(xi)
            da_pred, rt_pred = prediction_to_price_conditions(prediction_output)
            raw_samples = ecfm_model.sample(
                da_cond=da_pred,
                rt_cond=rt_pred,
                num_samples=num_samples,
                num_steps=num_flow_steps,
            )

            batch_size, sample_count, T, K, decision_dim = raw_samples.shape
            raw_flat = raw_samples.reshape(batch_size * sample_count, T, K, decision_dim)
            da_pred_rep = da_pred.unsqueeze(1).repeat(1, sample_count, 1, 1)
            rt_pred_rep = rt_pred.unsqueeze(1).repeat(1, sample_count, 1, 1)
            da_pred_rep = da_pred_rep.reshape(batch_size * sample_count, T, K)
            rt_pred_rep = rt_pred_rep.reshape(batch_size * sample_count, T, K)

            surrogate_energy, _ = q_bid_trading_energy(
                x=raw_flat,
                da_price=da_pred_rep,
                rt_price=rt_pred_rep,
                budget=budget,
                **energy_kwargs,
            )
            surrogate_energy = surrogate_energy.reshape(batch_size, sample_count)
            best_indices = surrogate_energy.argmin(dim=1)
            batch_indices = torch.arange(batch_size, device=device)
            raw_decision = raw_samples[batch_indices, best_indices]
            selected_energy = surrogate_energy[batch_indices, best_indices]

        if use_langevin and langevin_steps > 0:
            raw_decision, langevin_trace = langevin_refine_trading(
                raw_decision,
                da_price=da_pred.detach(),
                rt_price=rt_pred.detach(),
                budget=budget,
                energy_fn=q_bid_trading_energy,
                step_size=langevin_step_size,
                n_steps=langevin_steps,
                energy_kwargs=energy_kwargs,
                return_trace=True,
            )
        else:
            langevin_trace = torch.empty(
                raw_decision.shape[0],
                0,
                device=raw_decision.device,
            )

        with torch.no_grad():
            q, bid_price = transform_q_bid_decision(
                raw_decision,
                q_max=q_max,
                low=low,
                up=up,
            )
            cleared_prob = torch.sigmoid(alpha * q * (bid_price - da_true))
            profit = (q * (rt_true - da_true) * cleared_prob).sum(dim=(1, 2))
            l1_usage = bid_price.abs().sum(dim=-1)
            budget_violation = F.relu(l1_usage - budget)

            _, oracle_actual_decision, oracle_bid_price, _ = binary_bids_from_spread(
                da_true - rt_true,
                q_max=q_max,
                budget=budget,
                bid_low=low,
                bid_up=up,
            )
            oracle_sim = simulate_binary_virtual_bids(
                oracle_actual_decision,
                oracle_bid_price,
                da_true,
                rt_true,
                budget=budget,
            )

            predicted_das.append(da_pred.detach().cpu())
            predicted_rts.append(rt_pred.detach().cpu())
            true_das.append(da_true.detach().cpu())
            true_rts.append(rt_true.detach().cpu())
            raw_decisions.append(raw_decision.detach().cpu())
            q_decisions.append(q.detach().cpu())
            bid_prices.append(bid_price.detach().cpu())
            cleared_probs.append(cleared_prob.detach().cpu())
            selected_indices.append(best_indices.detach().cpu())
            selected_surrogate_energies.append(selected_energy.detach().cpu())
            langevin_energy_traces.append(langevin_trace.detach().cpu())
            profits.append(profit.detach().cpu())
            l1_usages.append(l1_usage.detach().cpu())
            violations.append(budget_violation.detach().cpu())
            oracle_decisions.append(oracle_actual_decision.detach().cpu())
            oracle_profits.append(oracle_sim["profit"].detach().cpu())
            zero_profits.append(torch.zeros_like(profit).detach().cpu())

    elapsed_seconds = time.perf_counter() - start_time

    predicted_da = torch.cat(predicted_das, dim=0)
    predicted_rt = torch.cat(predicted_rts, dim=0)
    true_da = torch.cat(true_das, dim=0)
    true_rt = torch.cat(true_rts, dim=0)
    raw_decision = torch.cat(raw_decisions, dim=0)
    q_decision = torch.cat(q_decisions, dim=0)
    bid_price = torch.cat(bid_prices, dim=0)
    cleared_prob = torch.cat(cleared_probs, dim=0)
    selected_index = torch.cat(selected_indices, dim=0)
    selected_surrogate_energy = torch.cat(selected_surrogate_energies, dim=0)
    langevin_energy_trace = torch.cat(langevin_energy_traces, dim=0)
    profit = torch.cat(profits, dim=0)
    l1_usage = torch.cat(l1_usages, dim=0)
    violation = torch.cat(violations, dim=0)
    oracle_decision = torch.cat(oracle_decisions, dim=0)
    oracle_profit = torch.cat(oracle_profits, dim=0)
    zero_profit = torch.cat(zero_profits, dim=0)

    spread_error = (predicted_da - predicted_rt) - (true_da - true_rt)
    price_mse = ((predicted_da - true_da) ** 2 + (predicted_rt - true_rt) ** 2).mean()
    metrics = {
        "price_mse": _as_float(price_mse),
        "spread_mse": _as_float((spread_error**2).mean()),
        "spread_mae": _as_float(spread_error.abs().mean()),
        "latency_seconds": float(elapsed_seconds),
        "latency_ms_per_day": float(elapsed_seconds * 1000.0 / max(len(profit), 1)),
        "mean_selected_surrogate_energy": _as_float(selected_surrogate_energy.mean()),
        "mean_langevin_start_energy": _as_float(
            langevin_energy_trace[:, 0].mean()
            if langevin_energy_trace.numel() > 0
            else torch.tensor(0.0)
        ),
        "mean_langevin_final_energy": _as_float(
            langevin_energy_trace[:, -1].mean()
            if langevin_energy_trace.numel() > 0
            else torch.tensor(0.0)
        ),
        "mean_q": _as_float(q_decision.mean()),
        "mean_abs_q": _as_float(q_decision.abs().mean()),
        "mean_bid_price": _as_float(bid_price.mean()),
        "mean_cleared_probability": _as_float(cleared_prob.mean()),
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
        "predicted_da": predicted_da.numpy(),
        "predicted_rt": predicted_rt.numpy(),
        "true_da": true_da.numpy(),
        "true_rt": true_rt.numpy(),
        "raw_decision": raw_decision.numpy(),
        "q_decision": q_decision.numpy(),
        "bid_price": bid_price.numpy(),
        "cleared_prob": cleared_prob.numpy(),
        "selected_sample_index": selected_index.numpy(),
        "selected_surrogate_energy": selected_surrogate_energy.numpy(),
        "langevin_energy_trace": langevin_energy_trace.numpy(),
        "profit": profit.numpy(),
        "l1_usage": l1_usage.numpy(),
        "budget_violation": violation.numpy(),
        "oracle_decision": oracle_decision.numpy(),
        "oracle_profit": oracle_profit.numpy(),
    }
    return metrics, arrays


def evaluate_fixed_q_bid_policy(
    prediction_model,
    ecfm_model,
    data_loader,
    device,
    low,
    up,
    budget,
    q_max=1.0,
    alpha=5.0,
    penalty_weight=100.0,
    num_samples=8,
    num_flow_steps=30,
    use_langevin=True,
    langevin_steps=30,
    langevin_step_size=1e-5,
):
    prediction_model.eval()
    ecfm_model.eval()

    predicted_das = []
    predicted_rts = []
    true_das = []
    true_rts = []
    raw_decisions = []
    q_decisions = []
    bid_prices = []
    cleared_probs = []
    selected_indices = []
    selected_surrogate_energies = []
    langevin_energy_traces = []
    profits = []
    l1_usages = []
    violations = []
    oracle_decisions = []
    oracle_profits = []
    zero_profits = []

    start_time = time.perf_counter()
    for xi, da_true, rt_true in data_loader:
        xi = xi.to(device)
        da_true = da_true.to(device)
        rt_true = rt_true.to(device)

        with torch.no_grad():
            prediction_output = prediction_model(xi)
            da_pred, rt_pred = prediction_to_price_conditions(prediction_output)
            spread_pred = da_pred - rt_pred
            q = fixed_q_from_spread(spread_pred, q_max=q_max)
            raw_samples = ecfm_model.sample(
                da_cond=da_pred,
                rt_cond=rt_pred,
                num_samples=num_samples,
                num_steps=num_flow_steps,
            )

            batch_size, sample_count, T, K = raw_samples.shape
            raw_flat = raw_samples.reshape(batch_size * sample_count, T, K)
            da_pred_rep = da_pred.unsqueeze(1).repeat(1, sample_count, 1, 1)
            rt_pred_rep = rt_pred.unsqueeze(1).repeat(1, sample_count, 1, 1)
            q_rep = q.unsqueeze(1).repeat(1, sample_count, 1, 1)
            da_pred_rep = da_pred_rep.reshape(batch_size * sample_count, T, K)
            rt_pred_rep = rt_pred_rep.reshape(batch_size * sample_count, T, K)
            q_rep = q_rep.reshape(batch_size * sample_count, T, K)

            surrogate_energy, _ = fixed_q_bid_trading_energy(
                x=raw_flat,
                da_price=da_pred_rep,
                rt_price=rt_pred_rep,
                low=low,
                up=up,
                budget=budget,
                q=q_rep,
                alpha=alpha,
                penalty_weight=penalty_weight,
            )
            surrogate_energy = surrogate_energy.reshape(batch_size, sample_count)
            best_indices = surrogate_energy.argmin(dim=1)
            batch_indices = torch.arange(batch_size, device=device)
            raw_decision = raw_samples[batch_indices, best_indices]
            selected_energy = surrogate_energy[batch_indices, best_indices]

        if use_langevin and langevin_steps > 0:
            raw_decision, langevin_trace = langevin_refine_trading(
                raw_decision,
                da_price=da_pred.detach(),
                rt_price=rt_pred.detach(),
                budget=budget,
                energy_fn=fixed_q_bid_trading_energy,
                step_size=langevin_step_size,
                n_steps=langevin_steps,
                energy_kwargs={
                    "low": low,
                    "up": up,
                    "q": q.detach(),
                    "alpha": alpha,
                    "penalty_weight": penalty_weight,
                },
                return_trace=True,
            )
        else:
            langevin_trace = torch.empty(
                raw_decision.shape[0],
                0,
                device=raw_decision.device,
            )

        with torch.no_grad():
            raw_bid_price = transform_x(raw_decision, low=low, up=up)
            bid_price = project_l1_budget(raw_bid_price, budget=budget)
            cleared_prob = torch.sigmoid(alpha * q * (bid_price - da_true))
            profit = (q * (rt_true - da_true) * cleared_prob).sum(dim=(1, 2))
            l1_usage = bid_price.abs().sum(dim=-1)
            budget_violation = F.relu(l1_usage - budget)

            _, oracle_actual_decision, oracle_bid_price, _ = binary_bids_from_spread(
                da_true - rt_true,
                q_max=q_max,
                budget=budget,
                bid_low=low,
                bid_up=up,
            )
            oracle_sim = simulate_binary_virtual_bids(
                oracle_actual_decision,
                oracle_bid_price,
                da_true,
                rt_true,
                budget=budget,
            )

            predicted_das.append(da_pred.detach().cpu())
            predicted_rts.append(rt_pred.detach().cpu())
            true_das.append(da_true.detach().cpu())
            true_rts.append(rt_true.detach().cpu())
            raw_decisions.append(raw_decision.detach().cpu())
            q_decisions.append(q.detach().cpu())
            bid_prices.append(bid_price.detach().cpu())
            cleared_probs.append(cleared_prob.detach().cpu())
            selected_indices.append(best_indices.detach().cpu())
            selected_surrogate_energies.append(selected_energy.detach().cpu())
            langevin_energy_traces.append(langevin_trace.detach().cpu())
            profits.append(profit.detach().cpu())
            l1_usages.append(l1_usage.detach().cpu())
            violations.append(budget_violation.detach().cpu())
            oracle_decisions.append(oracle_actual_decision.detach().cpu())
            oracle_profits.append(oracle_sim["profit"].detach().cpu())
            zero_profits.append(torch.zeros_like(profit).detach().cpu())

    elapsed_seconds = time.perf_counter() - start_time

    predicted_da = torch.cat(predicted_das, dim=0)
    predicted_rt = torch.cat(predicted_rts, dim=0)
    true_da = torch.cat(true_das, dim=0)
    true_rt = torch.cat(true_rts, dim=0)
    raw_decision = torch.cat(raw_decisions, dim=0)
    q_decision = torch.cat(q_decisions, dim=0)
    bid_price = torch.cat(bid_prices, dim=0)
    cleared_prob = torch.cat(cleared_probs, dim=0)
    selected_index = torch.cat(selected_indices, dim=0)
    selected_surrogate_energy = torch.cat(selected_surrogate_energies, dim=0)
    langevin_energy_trace = torch.cat(langevin_energy_traces, dim=0)
    profit = torch.cat(profits, dim=0)
    l1_usage = torch.cat(l1_usages, dim=0)
    violation = torch.cat(violations, dim=0)
    oracle_decision = torch.cat(oracle_decisions, dim=0)
    oracle_profit = torch.cat(oracle_profits, dim=0)
    zero_profit = torch.cat(zero_profits, dim=0)

    spread_error = (predicted_da - predicted_rt) - (true_da - true_rt)
    price_mse = ((predicted_da - true_da) ** 2 + (predicted_rt - true_rt) ** 2).mean()
    metrics = {
        "price_mse": _as_float(price_mse),
        "spread_mse": _as_float((spread_error**2).mean()),
        "spread_mae": _as_float(spread_error.abs().mean()),
        "latency_seconds": float(elapsed_seconds),
        "latency_ms_per_day": float(elapsed_seconds * 1000.0 / max(len(profit), 1)),
        "mean_selected_surrogate_energy": _as_float(selected_surrogate_energy.mean()),
        "mean_langevin_start_energy": _as_float(
            langevin_energy_trace[:, 0].mean()
            if langevin_energy_trace.numel() > 0
            else torch.tensor(0.0)
        ),
        "mean_langevin_final_energy": _as_float(
            langevin_energy_trace[:, -1].mean()
            if langevin_energy_trace.numel() > 0
            else torch.tensor(0.0)
        ),
        "mean_q": _as_float(q_decision.mean()),
        "mean_abs_q": _as_float(q_decision.abs().mean()),
        "mean_bid_price": _as_float(bid_price.mean()),
        "mean_cleared_probability": _as_float(cleared_prob.mean()),
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
        "predicted_da": predicted_da.numpy(),
        "predicted_rt": predicted_rt.numpy(),
        "true_da": true_da.numpy(),
        "true_rt": true_rt.numpy(),
        "raw_decision": raw_decision.numpy(),
        "q_decision": q_decision.numpy(),
        "bid_price": bid_price.numpy(),
        "cleared_prob": cleared_prob.numpy(),
        "selected_sample_index": selected_index.numpy(),
        "selected_surrogate_energy": selected_surrogate_energy.numpy(),
        "langevin_energy_trace": langevin_energy_trace.numpy(),
        "profit": profit.numpy(),
        "l1_usage": l1_usage.numpy(),
        "budget_violation": violation.numpy(),
        "oracle_decision": oracle_decision.numpy(),
        "oracle_profit": oracle_profit.numpy(),
    }
    return metrics, arrays


def evaluate_direct_bid_policy(
    prediction_model,
    bid_model,
    data_loader,
    device,
    low,
    up,
    budget,
    q_max=1.0,
    alpha=5.0,
    penalty_weight=100.0,
):
    prediction_model.eval()
    bid_model.eval()

    predicted_das = []
    predicted_rts = []
    true_das = []
    true_rts = []
    raw_decisions = []
    q_decisions = []
    bid_prices = []
    cleared_probs = []
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

            prediction_output = prediction_model(xi)
            da_pred, rt_pred = prediction_to_price_conditions(prediction_output)
            spread_pred = da_pred - rt_pred
            q = fixed_q_from_spread(spread_pred, q_max=q_max)
            raw_decision = bid_model(da_pred, rt_pred)
            raw_bid_price = transform_x(raw_decision, low=low, up=up)
            bid_price = project_l1_budget(raw_bid_price, budget=budget)
            cleared_prob = torch.sigmoid(alpha * q * (bid_price - da_true))
            profit = (q * (rt_true - da_true) * cleared_prob).sum(dim=(1, 2))
            l1_usage = bid_price.abs().sum(dim=-1)
            budget_violation = F.relu(l1_usage - budget)

            _, oracle_actual_decision, oracle_bid_price, _ = binary_bids_from_spread(
                da_true - rt_true,
                q_max=q_max,
                budget=budget,
                bid_low=low,
                bid_up=up,
            )
            oracle_sim = simulate_binary_virtual_bids(
                oracle_actual_decision,
                oracle_bid_price,
                da_true,
                rt_true,
                budget=budget,
            )

            predicted_das.append(da_pred.detach().cpu())
            predicted_rts.append(rt_pred.detach().cpu())
            true_das.append(da_true.detach().cpu())
            true_rts.append(rt_true.detach().cpu())
            raw_decisions.append(raw_decision.detach().cpu())
            q_decisions.append(q.detach().cpu())
            bid_prices.append(bid_price.detach().cpu())
            cleared_probs.append(cleared_prob.detach().cpu())
            profits.append(profit.detach().cpu())
            l1_usages.append(l1_usage.detach().cpu())
            violations.append(budget_violation.detach().cpu())
            oracle_decisions.append(oracle_actual_decision.detach().cpu())
            oracle_profits.append(oracle_sim["profit"].detach().cpu())
            zero_profits.append(torch.zeros_like(profit).detach().cpu())

    elapsed_seconds = time.perf_counter() - start_time

    predicted_da = torch.cat(predicted_das, dim=0)
    predicted_rt = torch.cat(predicted_rts, dim=0)
    true_da = torch.cat(true_das, dim=0)
    true_rt = torch.cat(true_rts, dim=0)
    raw_decision = torch.cat(raw_decisions, dim=0)
    q_decision = torch.cat(q_decisions, dim=0)
    bid_price = torch.cat(bid_prices, dim=0)
    cleared_prob = torch.cat(cleared_probs, dim=0)
    profit = torch.cat(profits, dim=0)
    l1_usage = torch.cat(l1_usages, dim=0)
    violation = torch.cat(violations, dim=0)
    oracle_decision = torch.cat(oracle_decisions, dim=0)
    oracle_profit = torch.cat(oracle_profits, dim=0)
    zero_profit = torch.cat(zero_profits, dim=0)

    spread_error = (predicted_da - predicted_rt) - (true_da - true_rt)
    price_mse = ((predicted_da - true_da) ** 2 + (predicted_rt - true_rt) ** 2).mean()
    metrics = {
        "price_mse": _as_float(price_mse),
        "spread_mse": _as_float((spread_error**2).mean()),
        "spread_mae": _as_float(spread_error.abs().mean()),
        "latency_seconds": float(elapsed_seconds),
        "latency_ms_per_day": float(elapsed_seconds * 1000.0 / max(len(profit), 1)),
        "mean_q": _as_float(q_decision.mean()),
        "mean_abs_q": _as_float(q_decision.abs().mean()),
        "mean_bid_price": _as_float(bid_price.mean()),
        "mean_cleared_probability": _as_float(cleared_prob.mean()),
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
        "predicted_da": predicted_da.numpy(),
        "predicted_rt": predicted_rt.numpy(),
        "true_da": true_da.numpy(),
        "true_rt": true_rt.numpy(),
        "raw_decision": raw_decision.numpy(),
        "q_decision": q_decision.numpy(),
        "bid_price": bid_price.numpy(),
        "cleared_prob": cleared_prob.numpy(),
        "profit": profit.numpy(),
        "l1_usage": l1_usage.numpy(),
        "budget_violation": violation.numpy(),
        "oracle_decision": oracle_decision.numpy(),
        "oracle_profit": oracle_profit.numpy(),
    }
    return metrics, arrays
