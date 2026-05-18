import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from data import load_train_val_from_zone_data
from helper import get_training_device
from pred_model import evaluate_prediction_model, train_prediction_model, PricePredictor


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
    history_len = 30
    gap = 1
    batch_size = 32
    hidden_dim = 512
    num_hidden_layers = 5
    learning_rate = 1e-4
    weight_decay = 1e-3
    num_epochs = 500

    train_loader, val_loader, metadata = load_train_val_from_zone_data(
        history_len=history_len,
        gap=gap,
        batch_size=batch_size,
    )
    T = 1
    K = metadata["K"]
    obs_dim = 2 * history_len * K

    device = get_training_device()
    print(f"Using device: {device}")
    print(f"Training days: {len(train_loader.dataset)}")
    print(f"Validation days: {len(val_loader.dataset)}")
    print(f"obs_dim: {obs_dim}, T: {T}, K: {K}")

    g_phi = PricePredictor(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        T=T,
        K=K,
        num_hidden_layers=num_hidden_layers,
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
    )
    final_train_loss = evaluate_prediction_model(g_phi, train_loader, device=device)
    final_val_loss = evaluate_prediction_model(g_phi, val_loader, device=device)

    plot_path = Path("prediction_loss.png")
    hyperparams = {
        "history_len": history_len,
        "gap": gap,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
    }
    plot_losses(losses, plot_path, hyperparams, final_train_loss, final_val_loss)
    print(f"Final train eval loss: {final_train_loss:.4f}")
    print(f"Final val eval loss: {final_val_loss:.4f}")
    print(f"Saved loss plot to {plot_path}")
