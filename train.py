from helper import get_model_device
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np



## General skeleton written by Chat -> modify to fit our case
def train_end_to_end(
    prediction_model,
    ecfm_model,
    train_loader,
    energy_fn,
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

            flow_loss = ecfm_model.rectified_flow_loss(z_1=true_prices
                                    , da_cond=da_cond, rt_cond=rt_cond)

            #
            x_cand = ecfm_model.sample(da_cond=da_cond,rt_cond=rt_cond,num_samples=32)

            ## Do not set these vars as global since we will be implementing 
            ## everything in main but right now they are undefined
            loss = energy_fn(x,da_price,rt_price,low,
                            up,budget,alpha=20.0,penalty_weight=100.0,)

            loss.backward()
            
            ## UPdates both model params so no need to worry about updating both
            ## model params 
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss / len(train_loader):.4f}")
