from helper import (
    fixed_q_bid_trading_energy,
    fixed_q_from_spread,
    get_model_device,
    q_bid_trading_energy,
)
import torch
import torch.nn.functional as F



def train_end_to_end(
    *args,
    **kwargs,
):
    raise NotImplementedError(
        "Use train_end_to_end_q_bid for the option-C virtual bidding pipeline."
    )


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


def train_end_to_end_q_bid(
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
):
    """
    First-pass end-to-end trainer for option C.

    The flow model should be ECFMModel(..., decision_dim=2). It generates raw
    [q, bid_price] decisions, which q_bid_trading_energy transforms into
    budgeted signed quantities and bounded bid prices.
    """
    if device is None:
        device = get_model_device(prediction_model)
    if getattr(ecfm_model, "decision_dim", None) != 2:
        raise ValueError("ecfm_model must be created with decision_dim=2 for option C.")

    losses = {
        "total": [],
        "energy": [],
        "spread_regularizer": [],
    }
    prediction_model.train()
    ecfm_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_energy_loss = 0.0
        total_spread_regularizer = 0.0
        num_batches = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            optimizer.zero_grad()

            prediction_output = prediction_model(xi)
            da_cond, rt_cond = _flow_condition_from_prediction(prediction_output)
            raw_candidates = ecfm_model.sample(
                da_cond=da_cond,
                rt_cond=rt_cond,
                num_samples=num_samples,
                num_steps=num_flow_steps,
                differentiable=True,
            )

            batch_size, sample_count, T, K, decision_dim = raw_candidates.shape
            raw_candidates = raw_candidates.reshape(
                batch_size * sample_count,
                T,
                K,
                decision_dim,
            )
            da_rep = da_true.unsqueeze(1).repeat(1, sample_count, 1, 1)
            rt_rep = rt_true.unsqueeze(1).repeat(1, sample_count, 1, 1)
            da_rep = da_rep.reshape(batch_size * sample_count, T, K)
            rt_rep = rt_rep.reshape(batch_size * sample_count, T, K)

            energy, _ = q_bid_trading_energy(
                x=raw_candidates,
                da_price=da_rep,
                rt_price=rt_rep,
                low=low,
                up=up,
                budget=budget,
                q_max=q_max,
                alpha=alpha,
                penalty_weight=penalty_weight,
            )
            energy_loss = energy.mean()
            spread_regularizer = _spread_mse_from_prediction(
                prediction_output,
                da_true,
                rt_true,
            )
            loss = energy_loss + prediction_regularizer_weight * spread_regularizer
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_spread_regularizer += spread_regularizer.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_energy_loss = total_energy_loss / max(num_batches, 1)
        avg_spread_regularizer = total_spread_regularizer / max(num_batches, 1)
        losses["total"].append(avg_loss)
        losses["energy"].append(avg_energy_loss)
        losses["spread_regularizer"].append(avg_spread_regularizer)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"total loss: {avg_loss:.4f} | "
            f"energy: {avg_energy_loss:.4f} | "
            f"spread reg: {avg_spread_regularizer:.4f}"
        )

    return losses


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
    }
    prediction_model.train()
    ecfm_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_energy_loss = 0.0
        total_spread_regularizer = 0.0
        num_batches = 0

        for xi, da_true, rt_true in train_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)

            optimizer.zero_grad()

            prediction_output = prediction_model(xi)
            da_cond, rt_cond = _flow_condition_from_prediction(prediction_output)
            spread_pred = _spread_from_prediction(prediction_output)
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
            energy_loss = energy.mean()
            spread_regularizer = _spread_mse_from_prediction(
                prediction_output,
                da_true,
                rt_true,
            )
            loss = energy_loss + prediction_regularizer_weight * spread_regularizer
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_spread_regularizer += spread_regularizer.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_energy_loss = total_energy_loss / max(num_batches, 1)
        avg_spread_regularizer = total_spread_regularizer / max(num_batches, 1)
        losses["total"].append(avg_loss)
        losses["energy"].append(avg_energy_loss)
        losses["spread_regularizer"].append(avg_spread_regularizer)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"total loss: {avg_loss:.4f} | "
            f"energy: {avg_energy_loss:.4f} | "
            f"spread reg: {avg_spread_regularizer:.4f}"
        )

    return losses
