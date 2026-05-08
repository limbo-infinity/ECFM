from helper import get_model_device
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


    
    
    

## General skeleton written by Chat -> modify to fit our case
def train_end_to_end(
    prediction_model,
    ecfm_model,
    train_loader,
    optimizer,
    device=None,
    num_epochs=100,
    lambda_flow=1.0,
    lambda_task=1.0,
):
    if device is None:
        device = get_model_device(prediction_model)

    prediction_model.train()
    ecfm_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in train_loader: 
            features = batch["features"].to(device)
            true_prices = batch["true_prices"].to(device)
            da_price = batch["da_price"].to(device)
            rt_price = batch["rt_price"].to(device)

            optimizer.zero_grad()
            da_cond, rt_cond = prediction_model(features)

            # 2. ECFM learns conditional distribution of prices/scenarios
            flow_loss = ecfm_model.rectified_flow_loss(
                z_1=true_prices,
                da_cond=da_cond,
                rt_cond=rt_cond,
            )

            # 3. ECFM generates scenarios conditioned on prediction_model output
            generated_scenarios = ecfm_model.sample(
                da_cond=da_cond,
                rt_cond=rt_cond,
                num_samples=32,
            )

            # 4. Use generated scenarios to make trading decision
            cleared_prob = make_trading_decision(generated_scenarios)

            # 5. Compute actual payoff using realized prices
            payoff = ((rt_price - da_price) * cleared_prob).sum(dim=(1, 2))

            # Since PyTorch minimizes losses, negative payoff = maximize payoff
            task_loss = -payoff.mean()

            # 6. Combine distribution-learning loss and task loss
            loss = lambda_flow * flow_loss + lambda_task * task_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss / len(train_loader):.4f}")
