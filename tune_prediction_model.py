import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from data import load_prices_from_zone_data, make_train_val_loaders
from helper import get_training_device
from pred_model import PricePredictor, evaluate_prediction_model, train_prediction_model


SEARCH_SPACE = {
    "history_len": (7, 45),
    "hidden_dim": [64, 128, 256, 512],
    "num_hidden_layers": (2, 10),
    "batch_size": [16, 32, 64, 128],
    "learning_rate": (1e-5, 5e-3),
    "weight_decay": (1e-6, 1),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_random_params():
    return {
        "history_len": random.randint(*SEARCH_SPACE["history_len"]),
        "hidden_dim": random.choice(SEARCH_SPACE["hidden_dim"]),
        "num_hidden_layers": random.choice(SEARCH_SPACE["num_hidden_layers"]),
        "batch_size": random.choice(SEARCH_SPACE["batch_size"]),
        "learning_rate": 10 ** random.uniform(
            math.log10(SEARCH_SPACE["learning_rate"][0]),
            math.log10(SEARCH_SPACE["learning_rate"][1]),
        ),
        "weight_decay": 10 ** random.uniform(
            math.log10(SEARCH_SPACE["weight_decay"][0]),
            math.log10(SEARCH_SPACE["weight_decay"][1]),
        ),
    }


def encode_params(params):
    return np.array(
        [
            params["history_len"],
            SEARCH_SPACE["hidden_dim"].index(params["hidden_dim"]),
            SEARCH_SPACE["num_hidden_layers"].index(params["num_hidden_layers"]),
            SEARCH_SPACE["batch_size"].index(params["batch_size"]),
            math.log10(params["learning_rate"]),
            math.log10(params["weight_decay"]),
        ],
        dtype=np.float64,
    )


def decode_params(encoded):
    history_low, history_high = SEARCH_SPACE["history_len"]
    hidden_values = SEARCH_SPACE["hidden_dim"]
    layer_values = SEARCH_SPACE["num_hidden_layers"]
    batch_values = SEARCH_SPACE["batch_size"]
    lr_low, lr_high = SEARCH_SPACE["learning_rate"]
    wd_low, wd_high = SEARCH_SPACE["weight_decay"]

    return {
        "history_len": int(round(np.clip(encoded[0], history_low, history_high))),
        "hidden_dim": hidden_values[
            int(round(np.clip(encoded[1], 0, len(hidden_values) - 1)))
        ],
        "num_hidden_layers": layer_values[
            int(round(np.clip(encoded[2], 0, len(layer_values) - 1)))
        ],
        "batch_size": batch_values[
            int(round(np.clip(encoded[3], 0, len(batch_values) - 1)))
        ],
        "learning_rate": 10 ** float(
            np.clip(encoded[4], math.log10(lr_low), math.log10(lr_high))
        ),
        "weight_decay": 10 ** float(
            np.clip(encoded[5], math.log10(wd_low), math.log10(wd_high))
        ),
    }


def expected_improvement(model, candidates, best_loss):
    mean, std = model.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)
    improvement = best_loss - mean
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def propose_params(results, num_candidates=512):
    if len(results) < 2:
        return sample_random_params()

    x_train = np.vstack([encode_params(result["params"]) for result in results])
    y_train = np.array([result["val_loss"] for result in results], dtype=np.float64)
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=0,
        n_restarts_optimizer=3,
    )
    model.fit(x_train, y_train)

    candidates = np.vstack(
        [encode_params(sample_random_params()) for _ in range(num_candidates)]
    )
    scores = expected_improvement(model, candidates, best_loss=float(y_train.min()))
    best_candidate = candidates[int(np.argmax(scores))]
    return decode_params(best_candidate)


def run_trial(trial_id, params, da_prices, rt_prices, device, num_epochs, gap, seed):
    set_seed(seed + trial_id)
    train_loader, val_loader = make_train_val_loaders(
        da_prices=da_prices,
        rt_prices=rt_prices,
        history_len=params["history_len"],
        gap=gap,
        batch_size=params["batch_size"],
        train_frac=0.8,
    )

    k = da_prices.shape[1]
    obs_dim = 2 * params["history_len"] * k
    model = PricePredictor(
        obs_dim=obs_dim,
        hidden_dim=params["hidden_dim"],
        T=1,
        K=k,
        num_hidden_layers=params["num_hidden_layers"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    losses = train_prediction_model(
        g_phi=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        verbose=False,
    )
    final_train_loss = evaluate_prediction_model(model, train_loader, device=device)
    final_val_loss = evaluate_prediction_model(model, val_loader, device=device)
    return {
        "trial": trial_id,
        "params": params,
        "train_epoch_loss": losses["train"][-1],
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
    }


def save_results(results, output_path):
    fieldnames = [
        "trial",
        "val_loss",
        "train_loss",
        "train_epoch_loss",
        "history_len",
        "hidden_dim",
        "num_hidden_layers",
        "batch_size",
        "learning_rate",
        "weight_decay",
    ]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "trial": result["trial"],
                "val_loss": result["val_loss"],
                "train_loss": result["train_loss"],
                "train_epoch_loss": result["train_epoch_loss"],
                **result["params"],
            }
            writer.writerow(row)


def main():
    num_trials = 50
    num_initial_random = 5
    num_epochs = 30
    gap = 1
    seed = 42
    output_path = Path("bayesian_tuning_results.csv")

    set_seed(seed)
    device = get_training_device()
    da_prices, rt_prices, metadata = load_prices_from_zone_data()
    print(f"Using device: {device}")
    print(f"Loaded {metadata['num_days']} days with K={metadata['K']}")

    results = []
    for trial_id in range(1, num_trials + 1):
        if trial_id <= num_initial_random:
            params = sample_random_params()
        else:
            params = propose_params(results)

        result = run_trial(
            trial_id=trial_id,
            params=params,
            da_prices=da_prices,
            rt_prices=rt_prices,
            device=device,
            num_epochs=num_epochs,
            gap=gap,
            seed=seed,
        )
        results.append(result)
        save_results(results, output_path)

        print(
            f"Trial {trial_id:02d}/{num_trials} | "
            f"val={result['val_loss']:.4f} | "
            f"train_eval={result['train_loss']:.4f} | "
            f"train_epoch={result['train_epoch_loss']:.4f} | "
            f"params={result['params']}"
        )

    best = min(results, key=lambda result: result["val_loss"])
    print("\nBest trial")
    print(f"val loss: {best['val_loss']:.4f}")
    print(f"train eval loss: {best['train_loss']:.4f}")
    print(f"train epoch loss: {best['train_epoch_loss']:.4f}")
    print(f"params: {best['params']}")
    print(f"Saved all results to {output_path}")


if __name__ == "__main__":
    main()
