import os
import random
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import load_train_val_from_zone_data
from experiment_logger import ExperimentLogger
from helper import get_training_device
from pred_model import evaluate_prediction_model, train_prediction_model, PricePredictor
from virtual_trading_simulator import evaluate_spread_policy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_summary(loader, metadata):
    target_indices = list(loader.dataset.target_indices)
    if not target_indices:
        return {"count": 0, "start_date": None, "end_date": None}

    dates = metadata["dates"]
    return {
        "count": len(target_indices),
        "start_index": int(target_indices[0]),
        "end_index": int(target_indices[-1]),
        "start_date": dates[target_indices[0]],
        "end_date": dates[target_indices[-1]],
    }


def prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def plot_losses(losses, output_path, hyperparams, final_train_loss, final_val_loss):
    epochs = range(1, len(losses["train"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses["train"], marker="o", label="Training loss")
    if losses["val"]:
        plt.plot(epochs, losses["val"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Prediction model loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    annotation = "\n".join(
        [
            f"history_len: {hyperparams['history_len']}",
            f"gap: {hyperparams['gap']}",
            f"hidden_dim: {hyperparams['hidden_dim']}",
            f"hidden_layers: {hyperparams['num_hidden_layers']}",
            f"prediction_target: {hyperparams['prediction_target']}",
            f"learning_rate: {hyperparams['learning_rate']:.2e}",
            f"weight_decay: {hyperparams['weight_decay']:.2e}",
            f"batch_size: {hyperparams['batch_size']}",
            f"epochs: {hyperparams['num_epochs']}",
            f"final train eval: {final_train_loss:.4f}",
            f"final val eval: {final_val_loss:.4f}",
        ]
    )
    plt.gca().text(
        0.985,
        0.86,
        annotation,
        transform=plt.gca().transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    history_len = 30
    gap = 1
    batch_size = 32
    hidden_dim = 512
    num_hidden_layers = 5
    prediction_target = "spread"
    learning_rate = 1e-4
    weight_decay = 1e-3
    num_epochs = 10
    feature_set = "historical_da_rt_prices"
    model_type = "spread_mlp"
    q_max = 1.0
    budget = 50.0
    decision_temperature = 25.0

    train_loader, val_loader, metadata = load_train_val_from_zone_data(
        history_len=history_len,
        gap=gap,
        batch_size=batch_size,
    )
    T = 1
    K = metadata["K"]
    obs_dim = 2 * history_len * K

    device = get_training_device()
    logger = ExperimentLogger(run_name=f"{model_type}_{prediction_target}_seed{seed}")
    print(f"Experiment directory: {logger.run_dir}")
    print(f"Using device: {device}")
    print(f"Training days: {len(train_loader.dataset)}")
    print(f"Validation days: {len(val_loader.dataset)}")
    print(f"obs_dim: {obs_dim}, T: {T}, K: {K}")

    hyperparams = {
        "history_len": history_len,
        "gap": gap,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "prediction_target": prediction_target,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
    }
    benchmark_config = {
        "q_max": q_max,
        "budget": budget,
        "decision_temperature": decision_temperature,
        "policy": "q = q_max * tanh(predicted_spread / decision_temperature), projected to L1 budget",
        "profit": "sum(q * (DA - RT))",
    }
    config = {
        "model_type": model_type,
        "feature_set": feature_set,
        "seed": seed,
        "device": str(device),
        "data": {
            "source": "zone_data",
            "num_days": metadata["num_days"],
            "num_zones": metadata["num_zones"],
            "hours_per_day": metadata["hours_per_day"],
            "K": metadata["K"],
            "start_date": metadata["dates"][0],
            "end_date": metadata["dates"][-1],
        },
        "splits": {
            "train": split_summary(train_loader, metadata),
            "val": split_summary(val_loader, metadata),
            "test": None,
        },
        "hyperparams": hyperparams,
        "benchmark": benchmark_config,
    }
    logger.save_config(config)

    g_phi = PricePredictor(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        T=T,
        K=K,
        num_hidden_layers=num_hidden_layers,
        output_mode=prediction_target,
    )
    g_phi.to(device)
    print(g_phi)

    optimizer = torch.optim.AdamW(
        g_phi.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    losses = train_prediction_model(
        g_phi=g_phi,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        loss_mode=prediction_target,
    )
    final_train_loss = evaluate_prediction_model(
        g_phi,
        train_loader,
        device=device,
        loss_mode=prediction_target,
    )
    final_val_loss = evaluate_prediction_model(
        g_phi,
        val_loader,
        device=device,
        loss_mode=prediction_target,
    )

    train_benchmark_metrics, train_benchmark_arrays = evaluate_spread_policy(
        g_phi,
        train_loader,
        device=device,
        q_max=q_max,
        budget=budget,
        temperature=decision_temperature,
    )
    val_benchmark_metrics, val_benchmark_arrays = evaluate_spread_policy(
        g_phi,
        val_loader,
        device=device,
        q_max=q_max,
        budget=budget,
        temperature=decision_temperature,
    )

    plot_path = logger.path("prediction_loss.png")
    plot_losses(losses, plot_path, hyperparams, final_train_loss, final_val_loss)
    logger.save_npz(
        "losses.npz",
        train_loss=np.asarray(losses["train"], dtype=np.float32),
        val_loss=np.asarray(losses["val"], dtype=np.float32),
    )
    logger.save_npz("train_benchmark_arrays.npz", **train_benchmark_arrays)
    logger.save_npz("val_benchmark_arrays.npz", **val_benchmark_arrays)

    metrics = {
        "final_train_prediction_loss": final_train_loss,
        "final_val_prediction_loss": final_val_loss,
        **prefix_metrics("train_benchmark", train_benchmark_metrics),
        **prefix_metrics("val_benchmark", val_benchmark_metrics),
    }
    logger.save_metrics_csv("metrics.csv", metrics)
    logger.save_summary(
        {
            "config_file": "config.json",
            "metrics_file": "metrics.csv",
            "model_file": "model.pt",
            "loss_plot": "prediction_loss.png",
            "loss_arrays": "losses.npz",
            "train_benchmark_arrays": "train_benchmark_arrays.npz",
            "val_benchmark_arrays": "val_benchmark_arrays.npz",
            "metrics": metrics,
        }
    )
    logger.save_model(
        "model.pt",
        g_phi,
        extra={
            "model_type": model_type,
            "obs_dim": obs_dim,
            "T": T,
            "K": K,
            "hyperparams": hyperparams,
            "final_train_prediction_loss": final_train_loss,
            "final_val_prediction_loss": final_val_loss,
        },
    )

    print(f"Final train eval loss: {final_train_loss:.4f}")
    print(f"Final val eval loss: {final_val_loss:.4f}")
    print(
        "Validation benchmark profit: "
        f"{val_benchmark_metrics['policy_total_profit']:.4f}"
    )
    print(
        "Validation benchmark Sharpe: "
        f"{val_benchmark_metrics['policy_annualized_sharpe']:.4f}"
    )
    print(f"Saved loss plot to {plot_path}")
    print(f"Saved experiment artifacts to {logger.run_dir}")
