from helper import (
    fixed_q_bid_trading_energy,
    fixed_q_from_spread,
    get_model_device,
    q_bid_trading_energy,
    soft_q_from_spread,
)
from contextlib import nullcontext
import torch
import torch.nn.functional as F



def _flow_condition_from_prediction(prediction_output):
    if isinstance(prediction_output, tuple):
        return prediction_output

    spread_cond = prediction_output
    zero_cond = torch.zeros_like(spread_cond)
    return spread_cond, zero_cond


def _spread_mse_from_prediction(prediction_output, da_true, rt_true):
    if isinstance(prediction_output, tuple):
        da_pred, rt_pred = prediction_output
        spread_pred = da_pred - rt_pred
    else:
        spread_pred = prediction_output

    spread_true = da_true - rt_true
    return F.mse_loss(spread_pred, spread_true)


def _spread_from_prediction(prediction_output):
    if isinstance(prediction_output, tuple):
        da_pred, rt_pred = prediction_output
        return da_pred - rt_pred
    return prediction_output


def _batch_risk_loss(
    profit_per_day,
    risk_loss_mode="none",
    sharpe_loss_weight=0.0,
    variance_loss_weight=0.0,
    eps=1e-6,
):
    mode = (risk_loss_mode or "none").lower()
    profit_variance = profit_per_day.var(unbiased=False)
    profit_std = torch.sqrt(profit_variance + eps)
    batch_sharpe = profit_per_day.mean() / profit_std

    if mode in ("none", "off"):
        risk_loss = torch.zeros(
            (),
            dtype=profit_per_day.dtype,
            device=profit_per_day.device,
        )
    elif mode == "sharpe":
        risk_loss = -sharpe_loss_weight * batch_sharpe
    elif mode in ("variance", "var", "mean_variance", "mean-variance"):
        risk_loss = variance_loss_weight * profit_variance
    else:
        raise ValueError(
            "risk_loss_mode must be one of: 'none', 'sharpe', 'mean_variance'."
        )

    return risk_loss, batch_sharpe, profit_variance


def train_end_to_end_fixed_q_bid(
    prediction_model,
    ecfm_model,
    train_loader,
    optimizer,
    low,
    up,
    budget,
    q_max=1.0,
    device=None,
    num_epochs=20,
    num_samples=4,
    num_flow_steps=20,
    alpha=1.0,
    penalty_weight=100.0,
    prediction_regularizer_weight=0.0,
    risk_loss_mode="none",
    sharpe_loss_weight=0.0,
    variance_loss_weight=0.0,
    risk_eps=1e-6,
    train_soft_q=True,
    q_temperature=10.0,
    freeze_prediction_model=False,
):
    """
    End-to-end trainer where the flow only generates bid price.

    q is fixed to +/- q_max from the predicted spread, so the ECFM model should
    be created with decision_dim=1.
    """
    if device is None:
        device = get_model_device(prediction_model)
    if getattr(ecfm_model, "decision_dim", None) != 1:
        raise ValueError("ecfm_model must be created with decision_dim=1.")

    losses = {
        "total": [],
        "energy": [],
        "spread_regularizer": [],
        "risk": [],
        "batch_sharpe": [],
        "profit_variance": [],
    }
    if freeze_prediction_model:
        prediction_model.eval()
        for parameter in prediction_model.parameters():
            parameter.requires_grad_(False)
    else:
        prediction_model.train()
    ecfm_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_energy_loss = 0.0
        total_spread_regularizer = 0.0
        total_risk_loss = 0.0
        total_batch_sharpe = 0.0
        total_profit_variance = 0.0
        num_batches = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            optimizer.zero_grad()

            pred_context = torch.no_grad() if freeze_prediction_model else nullcontext()
            with pred_context:
                prediction_output = prediction_model(xi)
            da_cond, rt_cond = _flow_condition_from_prediction(prediction_output)
            spread_pred = _spread_from_prediction(prediction_output)
            if train_soft_q:
                q = soft_q_from_spread(
                    spread_pred,
                    q_max=q_max,
                    temperature=q_temperature,
                )
            else:
                q = fixed_q_from_spread(spread_pred, q_max=q_max)

            raw_candidates = ecfm_model.sample(
                da_cond=da_cond,
                rt_cond=rt_cond,
                num_samples=num_samples,
                num_steps=num_flow_steps,
                differentiable=True,
            )

            batch_size, sample_count, T, K = raw_candidates.shape
            raw_candidates = raw_candidates.reshape(batch_size * sample_count, T, K)
            da_rep = da_true.unsqueeze(1).repeat(1, sample_count, 1, 1)
            rt_rep = rt_true.unsqueeze(1).repeat(1, sample_count, 1, 1)
            q_rep = q.unsqueeze(1).repeat(1, sample_count, 1, 1)
            da_rep = da_rep.reshape(batch_size * sample_count, T, K)
            rt_rep = rt_rep.reshape(batch_size * sample_count, T, K)
            q_rep = q_rep.reshape(batch_size * sample_count, T, K)

            energy, _ = fixed_q_bid_trading_energy(
                x=raw_candidates,
                da_price=da_rep,
                rt_price=rt_rep,
                low=low,
                up=up,
                budget=budget,
                q=q_rep,
                alpha=alpha,
                penalty_weight=penalty_weight,
            )
            daily_energy = energy.reshape(batch_size, sample_count).mean(dim=1)
            energy_loss = daily_energy.mean()
            profit_per_day = -daily_energy
            risk_loss, batch_sharpe, profit_variance = _batch_risk_loss(
                profit_per_day,
                risk_loss_mode=risk_loss_mode,
                sharpe_loss_weight=sharpe_loss_weight,
                variance_loss_weight=variance_loss_weight,
                eps=risk_eps,
            )
            spread_regularizer = _spread_mse_from_prediction(
                prediction_output,
                da_true,
                rt_true,
            )
            loss = (
                energy_loss
                + risk_loss
                + prediction_regularizer_weight * spread_regularizer
            )
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_spread_regularizer += spread_regularizer.item()
            total_risk_loss += risk_loss.item()
            total_batch_sharpe += batch_sharpe.item()
            total_profit_variance += profit_variance.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_energy_loss = total_energy_loss / max(num_batches, 1)
        avg_spread_regularizer = total_spread_regularizer / max(num_batches, 1)
        avg_risk_loss = total_risk_loss / max(num_batches, 1)
        avg_batch_sharpe = total_batch_sharpe / max(num_batches, 1)
        avg_profit_variance = total_profit_variance / max(num_batches, 1)
        losses["total"].append(avg_loss)
        losses["energy"].append(avg_energy_loss)
        losses["spread_regularizer"].append(avg_spread_regularizer)
        losses["risk"].append(avg_risk_loss)
        losses["batch_sharpe"].append(avg_batch_sharpe)
        losses["profit_variance"].append(avg_profit_variance)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"total loss: {avg_loss:.4f} | "
            f"energy: {avg_energy_loss:.4f} | "
            f"risk: {avg_risk_loss:.4f} | "
            f"batch sharpe: {avg_batch_sharpe:.4f} | "
            f"spread reg: {avg_spread_regularizer:.4f}"
        )

    return losses


def train_end_to_end_direct_bid(
    prediction_model,
    bid_model,
    train_loader,
    optimizer,
    low,
    up,
    budget,
    q_max=1.0,
    device=None,
    num_epochs=20,
    alpha=1.0,
    penalty_weight=100.0,
    prediction_regularizer_weight=0.0,
    risk_loss_mode="none",
    sharpe_loss_weight=0.0,
    variance_loss_weight=0.0,
    risk_eps=1e-6,
    train_soft_q=True,
    q_temperature=10.0,
    freeze_prediction_model=False,
):
    """
    Direct MLP baseline: predicted prices -> one raw bid-price vector.
    """
    if device is None:
        device = get_model_device(prediction_model)

    losses = {
        "total": [],
        "energy": [],
        "spread_regularizer": [],
        "risk": [],
        "batch_sharpe": [],
        "profit_variance": [],
    }
    if freeze_prediction_model:
        prediction_model.eval()
        for parameter in prediction_model.parameters():
            parameter.requires_grad_(False)
    else:
        prediction_model.train()
    bid_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_energy_loss = 0.0
        total_spread_regularizer = 0.0
        total_risk_loss = 0.0
        total_batch_sharpe = 0.0
        total_profit_variance = 0.0
        num_batches = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            optimizer.zero_grad()

            pred_context = torch.no_grad() if freeze_prediction_model else nullcontext()
            with pred_context:
                prediction_output = prediction_model(xi)
            da_cond, rt_cond = _flow_condition_from_prediction(prediction_output)
            spread_pred = _spread_from_prediction(prediction_output)
            if train_soft_q:
                q = soft_q_from_spread(
                    spread_pred,
                    q_max=q_max,
                    temperature=q_temperature,
                )
            else:
                q = fixed_q_from_spread(spread_pred, q_max=q_max)
            raw_bid = bid_model(da_cond, rt_cond)

            energy, _ = fixed_q_bid_trading_energy(
                x=raw_bid,
                da_price=da_true,
                rt_price=rt_true,
                low=low,
                up=up,
                budget=budget,
                q=q,
                alpha=alpha,
                penalty_weight=penalty_weight,
            )
            energy_loss = energy.mean()
            profit_per_day = -energy
            risk_loss, batch_sharpe, profit_variance = _batch_risk_loss(
                profit_per_day,
                risk_loss_mode=risk_loss_mode,
                sharpe_loss_weight=sharpe_loss_weight,
                variance_loss_weight=variance_loss_weight,
                eps=risk_eps,
            )
            spread_regularizer = _spread_mse_from_prediction(
                prediction_output,
                da_true,
                rt_true,
            )
            loss = (
                energy_loss
                + risk_loss
                + prediction_regularizer_weight * spread_regularizer
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_spread_regularizer += spread_regularizer.item()
            total_risk_loss += risk_loss.item()
            total_batch_sharpe += batch_sharpe.item()
            total_profit_variance += profit_variance.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_energy_loss = total_energy_loss / max(num_batches, 1)
        avg_spread_regularizer = total_spread_regularizer / max(num_batches, 1)
        avg_risk_loss = total_risk_loss / max(num_batches, 1)
        avg_batch_sharpe = total_batch_sharpe / max(num_batches, 1)
        avg_profit_variance = total_profit_variance / max(num_batches, 1)
        losses["total"].append(avg_loss)
        losses["energy"].append(avg_energy_loss)
        losses["spread_regularizer"].append(avg_spread_regularizer)
        losses["risk"].append(avg_risk_loss)
        losses["batch_sharpe"].append(avg_batch_sharpe)
        losses["profit_variance"].append(avg_profit_variance)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"direct total loss: {avg_loss:.4f} | "
            f"energy: {avg_energy_loss:.4f} | "
            f"risk: {avg_risk_loss:.4f} | "
            f"batch sharpe: {avg_batch_sharpe:.4f} | "
            f"spread reg: {avg_spread_regularizer:.4f}"
        )

    return losses
