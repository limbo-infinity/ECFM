from helper import get_model_device, q_bid_trading_energy
import torch



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

    losses = []
    prediction_model.train()
    ecfm_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
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
            loss = energy.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} | end-to-end q/bid loss: {avg_loss:.4f}")

    return losses
