import os
import random
import copy

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_train_val_from_zone_data
from ecfm_model import ECFMModel
from experiment_logger import ExperimentLogger
from helper import get_training_device
from pred_model import (
    DirectBidMLP,
    PricePredictor,
    evaluate_prediction_model,
    train_prediction_model,
)
from train import train_end_to_end_direct_bid, train_end_to_end_fixed_q_bid
from virtual_trading_simulator import (
    evaluate_direct_bid_policy,
    evaluate_fixed_q_bid_policy,
)


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


def plot_curve(values, output_path, title, ylabel, label, hyperparams):
    epochs = range(1, len(values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    annotation = "\n".join(
        [
            f"history_len: {hyperparams['history_len']}",
            f"gap: {hyperparams['gap']}",
            f"prediction_hidden_dim: {hyperparams['prediction_hidden_dim']}",
            f"flow_hidden_dim: {hyperparams['flow_hidden_dim']}",
            f"q_max: {hyperparams['q_max']}",
            f"budget: {hyperparams['budget']}",
            f"train_samples: {hyperparams['train_num_samples']}",
            f"flow_steps: {hyperparams['train_num_flow_steps']}",
            f"langevin_steps: {hyperparams['langevin_steps']}",
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


def plot_end_to_end_losses(losses, output_path, hyperparams):
    epochs = range(1, len(losses["total"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses["total"], marker="o", label="Total loss")
    plt.plot(epochs, losses["energy"], marker="o", label="Trading energy")
    plt.plot(
        epochs,
        losses["spread_regularizer"],
        marker="o",
        label="Spread MSE regularizer",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("End-to-end decision training loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    annotation = "\n".join(
        [
            f"regularizer_weight: {hyperparams['prediction_regularizer_weight']:.2e}",
            f"flow_hidden_dim: {hyperparams['flow_hidden_dim']}",
            f"q_max: {hyperparams['q_max']}",
            f"budget: {hyperparams['budget']}",
            f"train_samples: {hyperparams['train_num_samples']}",
            f"flow_steps: {hyperparams['train_num_flow_steps']}",
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


def plot_langevin_losses(train_trace, val_trace, output_path, hyperparams):
    plt.figure(figsize=(10, 6))

    if train_trace.size > 0:
        steps = range(train_trace.shape[1])
        plt.plot(
            steps,
            train_trace.mean(axis=0),
            marker="o",
            label="Training refinement loss",
        )
    if val_trace.size > 0:
        steps = range(val_trace.shape[1])
        plt.plot(
            steps,
            val_trace.mean(axis=0),
            marker="o",
            label="Validation refinement loss",
        )

    plt.xlabel("Langevin step")
    plt.ylabel("Predicted energy")
    plt.title("Langevin refinement loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    annotation = "\n".join(
        [
            f"langevin_steps: {hyperparams['langevin_steps']}",
            f"langevin_step_size: {hyperparams['langevin_step_size']:.2e}",
            f"alpha: {hyperparams['alpha']}",
            f"penalty_weight: {hyperparams['penalty_weight']}",
            f"q_max: {hyperparams['q_max']}",
            f"budget: {hyperparams['budget']}",
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


def plot_prediction_losses(losses, output_path, hyperparams):
    epochs = range(1, len(losses["train"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses["train"], marker="o", label="Training loss")
    if losses["val"]:
        plt.plot(epochs, losses["val"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Prediction pretraining loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    annotation = "\n".join(
        [
            f"history_len: {hyperparams['history_len']}",
            f"gap: {hyperparams['gap']}",
            f"hidden_dim: {hyperparams['prediction_hidden_dim']}",
            f"hidden_layers: {hyperparams['prediction_num_hidden_layers']}",
            f"prediction_output: prices",
            f"prediction_loss: spread",
            f"learning_rate: {hyperparams['prediction_learning_rate']:.2e}",
            f"weight_decay: {hyperparams['prediction_weight_decay']:.2e}",
            f"batch_size: {hyperparams['batch_size']}",
            f"epochs: {hyperparams['prediction_pretrain_epochs']}",
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
    gap = 2
    batch_size = 32
    prediction_hidden_dim = 512
    prediction_num_hidden_layers = 5
    direct_bid_hidden_dim = 512
    direct_bid_hidden_layers = 3
    flow_hidden_dim = 512
    prediction_pretrain_epochs = 30
    end_to_end_epochs = 100
    prediction_learning_rate = 1e-4
    end_to_end_learning_rate = 5e-5
    prediction_weight_decay = 1e-3
    end_to_end_weight_decay = 1e-4
    prediction_regularizer_weight = 0.03
    train_num_samples = 4
    train_num_flow_steps = 15
    eval_num_samples = 8
    eval_num_flow_steps = 20
    langevin_steps = 20
    langevin_step_size = 1e-4
    alpha = 5.0
    penalty_weight = 1.0
    q_max = 1.0
    budget = 350.0
    bid_low = -1000.0
    bid_up = 2000.0
    feature_set = "historical_da_rt_prices"
    model_type = "ecfm_fixed_q_bid"

    train_loader, val_loader, metadata = load_train_val_from_zone_data(
        history_len=history_len,
        gap=gap,
        batch_size=batch_size,
    )
    T = 1
    K = metadata["K"]
    obs_dim = 2 * history_len * K

    device = get_training_device()
    logger = ExperimentLogger(run_name=f"{model_type}_seed{seed}")
    print(f"Experiment directory: {logger.run_dir}")
    print(f"Using device: {device}")
    print(f"Training days: {len(train_loader.dataset)}")
    print(f"Validation days: {len(val_loader.dataset)}")
    print(f"obs_dim: {obs_dim}, T: {T}, K: {K}")

    hyperparams = {
        "history_len": history_len,
        "gap": gap,
        "batch_size": batch_size,
        "prediction_hidden_dim": prediction_hidden_dim,
        "prediction_num_hidden_layers": prediction_num_hidden_layers,
        "direct_bid_hidden_dim": direct_bid_hidden_dim,
        "direct_bid_hidden_layers": direct_bid_hidden_layers,
        "flow_hidden_dim": flow_hidden_dim,
        "prediction_pretrain_epochs": prediction_pretrain_epochs,
        "end_to_end_epochs": end_to_end_epochs,
        "prediction_learning_rate": prediction_learning_rate,
        "end_to_end_learning_rate": end_to_end_learning_rate,
        "prediction_weight_decay": prediction_weight_decay,
        "end_to_end_weight_decay": end_to_end_weight_decay,
        "prediction_regularizer_weight": prediction_regularizer_weight,
        "train_num_samples": train_num_samples,
        "train_num_flow_steps": train_num_flow_steps,
        "eval_num_samples": eval_num_samples,
        "eval_num_flow_steps": eval_num_flow_steps,
        "langevin_steps": langevin_steps,
        "langevin_step_size": langevin_step_size,
        "alpha": alpha,
        "penalty_weight": penalty_weight,
        "q_max": q_max,
        "budget": budget,
        "bid_low": bid_low,
        "bid_up": bid_up,
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
        "decision_model": {
            "decision": "fixed q from predicted spread, raw bid price generated by ECFM",
            "q_rule": "q = +q_max if predicted DA-RT < 0 else -q_max",
            "bid_transform": "low + (up - low) * sigmoid(raw_bid)",
            "budget_constraint": (
                "hard projection onto sum(abs(bid_price)) <= budget for each day"
            ),
            "clearing": "sigmoid(alpha * q * (bid_price - DA))", ## ideally we do have a large 
                                                                ## value of alpha to simulate 
                                                                # the indicator function    
            "profit": "sum(q * (RT - DA) * clearing_probability)",
            "candidate_selection": (
                "sample candidates from the flow, pick the lowest predicted-energy "
                "candidate, then optionally run Langevin refinement using predicted prices"
            ),
        },
    }
    logger.save_config(config)

    prediction_model = PricePredictor(
        obs_dim=obs_dim,
        hidden_dim=prediction_hidden_dim,
        T=T,
        K=K,
        num_hidden_layers=prediction_num_hidden_layers,
        output_mode="prices",
    ).to(device)
    ecfm_model = ECFMModel(
        T=T,
        K=K,
        hidden_dim=flow_hidden_dim,
        decision_dim=1,
    ).to(device)
    direct_bid_model = DirectBidMLP(
        T=T,
        K=K,
        hidden_dim=direct_bid_hidden_dim,
        num_hidden_layers=direct_bid_hidden_layers,
    ).to(device)

    prediction_optimizer = torch.optim.AdamW(
        prediction_model.parameters(),
        lr=prediction_learning_rate,
        weight_decay=prediction_weight_decay,
    )
    prediction_losses = train_prediction_model(
        g_phi=prediction_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=prediction_optimizer,
        num_epochs=prediction_pretrain_epochs,
        loss_mode="spread", ## try with prices first and then consider using spread loss
    )
    pretrain_train_loss = evaluate_prediction_model(
        prediction_model,
        train_loader,
        device=device,
        loss_mode="spread",
    )
    pretrain_val_loss = evaluate_prediction_model(
        prediction_model,
        val_loader,
        device=device,
        loss_mode="spread",
    )

    direct_prediction_model = copy.deepcopy(prediction_model).to(device)
    ecfm_prediction_model = copy.deepcopy(prediction_model).to(device)

    direct_optimizer = torch.optim.AdamW(
        list(direct_prediction_model.parameters()) + list(direct_bid_model.parameters()),
        lr=end_to_end_learning_rate,
        weight_decay=end_to_end_weight_decay,
    )
    direct_losses = train_end_to_end_direct_bid(
        prediction_model=direct_prediction_model,
        bid_model=direct_bid_model,
        train_loader=train_loader,
        optimizer=direct_optimizer,
        low=bid_low,
        up=bid_up,
        budget=budget,
        q_max=q_max,
        device=device,
        num_epochs=end_to_end_epochs,
        alpha=alpha,
        penalty_weight=penalty_weight,
        prediction_regularizer_weight=prediction_regularizer_weight,
    )

    end_to_end_optimizer = torch.optim.AdamW(
        list(ecfm_prediction_model.parameters()) + list(ecfm_model.parameters()),
        lr=end_to_end_learning_rate,
        weight_decay=end_to_end_weight_decay,
    )
    end_to_end_losses = train_end_to_end_fixed_q_bid(
        prediction_model=ecfm_prediction_model,
        ecfm_model=ecfm_model,
        train_loader=train_loader,
        optimizer=end_to_end_optimizer,
        low=bid_low,
        up=bid_up,
        budget=budget,
        q_max=q_max,
        device=device,
        num_epochs=end_to_end_epochs,
        num_samples=train_num_samples,
        num_flow_steps=train_num_flow_steps,
        alpha=alpha,
        penalty_weight=penalty_weight,
        prediction_regularizer_weight=prediction_regularizer_weight,
    )

    final_train_prediction_loss = evaluate_prediction_model(
        ecfm_prediction_model,
        train_loader,
        device=device,
        loss_mode="spread",
    )
    final_val_prediction_loss = evaluate_prediction_model(
        ecfm_prediction_model,
        val_loader,
        device=device,
        loss_mode="spread",
    )
    direct_final_train_prediction_loss = evaluate_prediction_model(
        direct_prediction_model,
        train_loader,
        device=device,
        loss_mode="spread",
    )
    direct_final_val_prediction_loss = evaluate_prediction_model(
        direct_prediction_model,
        val_loader,
        device=device,
        loss_mode="spread",
    )

    direct_train_policy_metrics, direct_train_policy_arrays = evaluate_direct_bid_policy(
        prediction_model=direct_prediction_model,
        bid_model=direct_bid_model,
        data_loader=train_loader,
        device=device,
        low=bid_low,
        up=bid_up,
        budget=budget,
        q_max=q_max,
        alpha=alpha,
        penalty_weight=penalty_weight,
    )
    direct_val_policy_metrics, direct_val_policy_arrays = evaluate_direct_bid_policy(
        prediction_model=direct_prediction_model,
        bid_model=direct_bid_model,
        data_loader=val_loader,
        device=device,
        low=bid_low,
        up=bid_up,
        budget=budget,
        q_max=q_max,
        alpha=alpha,
        penalty_weight=penalty_weight,
    )

    train_policy_metrics, train_policy_arrays = evaluate_fixed_q_bid_policy(
        prediction_model=ecfm_prediction_model,
        ecfm_model=ecfm_model,
        data_loader=train_loader,
        device=device,
        low=bid_low,
        up=bid_up,
        budget=budget,
        q_max=q_max,
        alpha=alpha,
        penalty_weight=penalty_weight,
        num_samples=eval_num_samples,
        num_flow_steps=eval_num_flow_steps,
        use_langevin=langevin_steps > 0,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
    )
    val_policy_metrics, val_policy_arrays = evaluate_fixed_q_bid_policy(
        prediction_model=ecfm_prediction_model,
        ecfm_model=ecfm_model,
        data_loader=val_loader,
        device=device,
        low=bid_low,
        up=bid_up,
        budget=budget,
        q_max=q_max,
        alpha=alpha,
        penalty_weight=penalty_weight,
        num_samples=eval_num_samples,
        num_flow_steps=eval_num_flow_steps,
        use_langevin=langevin_steps > 0,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
    )

    prediction_plot_path = logger.path("prediction_pretrain_loss.png")
    direct_plot_path = logger.path("direct_mlp_loss.png")
    end_to_end_plot_path = logger.path("end_to_end_loss.png")
    langevin_plot_path = logger.path("langevin_refinement_loss.png")
    plot_prediction_losses(prediction_losses, prediction_plot_path, hyperparams)
    plot_end_to_end_losses(
        direct_losses,
        direct_plot_path,
        hyperparams=hyperparams,
    )
    plot_end_to_end_losses(
        end_to_end_losses,
        end_to_end_plot_path,
        hyperparams=hyperparams,
    )
    plot_langevin_losses(
        train_policy_arrays["langevin_energy_trace"],
        val_policy_arrays["langevin_energy_trace"],
        langevin_plot_path,
        hyperparams,
    )
    logger.save_npz(
        "losses.npz",
        prediction_train_loss=np.asarray(prediction_losses["train"], dtype=np.float32),
        prediction_val_loss=np.asarray(prediction_losses["val"], dtype=np.float32),
        direct_total_loss=np.asarray(direct_losses["total"], dtype=np.float32),
        direct_energy_loss=np.asarray(direct_losses["energy"], dtype=np.float32),
        direct_spread_regularizer=np.asarray(
            direct_losses["spread_regularizer"],
            dtype=np.float32,
        ),
        end_to_end_total_loss=np.asarray(end_to_end_losses["total"], dtype=np.float32),
        end_to_end_energy_loss=np.asarray(end_to_end_losses["energy"], dtype=np.float32),
        end_to_end_spread_regularizer=np.asarray(
            end_to_end_losses["spread_regularizer"],
            dtype=np.float32,
        ),
        train_langevin_energy_trace=train_policy_arrays["langevin_energy_trace"],
        val_langevin_energy_trace=val_policy_arrays["langevin_energy_trace"],
    )
    logger.save_npz("direct_train_policy_arrays.npz", **direct_train_policy_arrays)
    logger.save_npz("direct_val_policy_arrays.npz", **direct_val_policy_arrays)
    logger.save_npz("train_policy_arrays.npz", **train_policy_arrays)
    logger.save_npz("val_policy_arrays.npz", **val_policy_arrays)

    metrics = {
        "pretrain_train_prediction_loss": pretrain_train_loss,
        "pretrain_val_prediction_loss": pretrain_val_loss,
        "direct_final_train_prediction_loss": direct_final_train_prediction_loss,
        "direct_final_val_prediction_loss": direct_final_val_prediction_loss,
        "direct_final_total_loss": direct_losses["total"][-1],
        "direct_final_energy_loss": direct_losses["energy"][-1],
        "direct_final_spread_regularizer": direct_losses["spread_regularizer"][-1],
        "final_train_prediction_loss": final_train_prediction_loss,
        "final_val_prediction_loss": final_val_prediction_loss,
        "final_end_to_end_total_loss": end_to_end_losses["total"][-1],
        "final_end_to_end_energy_loss": end_to_end_losses["energy"][-1],
        "final_end_to_end_spread_regularizer": end_to_end_losses[
            "spread_regularizer"
        ][-1],
        **prefix_metrics("train_policy", train_policy_metrics),
        **prefix_metrics("val_policy", val_policy_metrics),
        **prefix_metrics("direct_train_policy", direct_train_policy_metrics),
        **prefix_metrics("direct_val_policy", direct_val_policy_metrics),
    }
    logger.save_metrics_csv("metrics.csv", metrics)
    logger.save_summary(
        {
            "config_file": "config.json",
            "metrics_file": "metrics.csv",
            "prediction_model_file": "prediction_model.pt",
            "direct_prediction_model_file": "direct_prediction_model.pt",
            "direct_bid_model_file": "direct_bid_model.pt",
            "ecfm_prediction_model_file": "ecfm_prediction_model.pt",
            "ecfm_model_file": "ecfm_model.pt",
            "prediction_loss_plot": "prediction_pretrain_loss.png",
            "direct_loss_plot": "direct_mlp_loss.png",
            "end_to_end_loss_plot": "end_to_end_loss.png",
            "langevin_loss_plot": "langevin_refinement_loss.png",
            "loss_arrays": "losses.npz",
            "direct_train_policy_arrays": "direct_train_policy_arrays.npz",
            "direct_val_policy_arrays": "direct_val_policy_arrays.npz",
            "train_policy_arrays": "train_policy_arrays.npz",
            "val_policy_arrays": "val_policy_arrays.npz",
            "metrics": metrics,
        }
    )
    logger.save_model(
        "prediction_model.pt",
        prediction_model,
        extra={
            "model_type": "price_predictor",
            "obs_dim": obs_dim,
            "T": T,
            "K": K,
            "hyperparams": hyperparams,
            "final_train_prediction_loss": pretrain_train_loss,
            "final_val_prediction_loss": pretrain_val_loss,
        },
    )
    logger.save_model(
        "direct_prediction_model.pt",
        direct_prediction_model,
        extra={
            "model_type": "direct_baseline_price_predictor",
            "obs_dim": obs_dim,
            "T": T,
            "K": K,
            "hyperparams": hyperparams,
            "final_train_prediction_loss": direct_final_train_prediction_loss,
            "final_val_prediction_loss": direct_final_val_prediction_loss,
        },
    )
    logger.save_model(
        "direct_bid_model.pt",
        direct_bid_model,
        extra={
            "model_type": "direct_bid_mlp",
            "T": T,
            "K": K,
            "hyperparams": hyperparams,
            "final_total_loss": direct_losses["total"][-1],
            "final_energy_loss": direct_losses["energy"][-1],
            "final_spread_regularizer": direct_losses["spread_regularizer"][-1],
        },
    )
    logger.save_model(
        "ecfm_prediction_model.pt",
        ecfm_prediction_model,
        extra={
            "model_type": "ecfm_branch_price_predictor",
            "obs_dim": obs_dim,
            "T": T,
            "K": K,
            "hyperparams": hyperparams,
            "final_train_prediction_loss": final_train_prediction_loss,
            "final_val_prediction_loss": final_val_prediction_loss,
        },
    )
    logger.save_model(
        "ecfm_model.pt",
        ecfm_model,
        extra={
            "model_type": "ecfm_fixed_q_bid",
            "T": T,
            "K": K,
            "decision_dim": 1,
            "hyperparams": hyperparams,
            "final_end_to_end_total_loss": end_to_end_losses["total"][-1],
            "final_end_to_end_energy_loss": end_to_end_losses["energy"][-1],
            "final_end_to_end_spread_regularizer": end_to_end_losses[
                "spread_regularizer"
            ][-1],
        },
    )

    print(f"Pretrain val prediction loss: {pretrain_val_loss:.4f}")
    print(
        "Direct MLP validation policy profit: "
        f"{direct_val_policy_metrics['policy_total_profit']:.4f}"
    )
    print(
        "Direct MLP validation policy Sharpe: "
        f"{direct_val_policy_metrics['policy_annualized_sharpe']:.4f}"
    )
    print(f"Final val prediction loss: {final_val_prediction_loss:.4f}")
    print(f"Final end-to-end total loss: {end_to_end_losses['total'][-1]:.4f}")
    print(f"Final end-to-end energy loss: {end_to_end_losses['energy'][-1]:.4f}")
    print(
        "Final end-to-end spread regularizer: "
        f"{end_to_end_losses['spread_regularizer'][-1]:.4f}"
    )
    print(f"Validation policy profit: {val_policy_metrics['policy_total_profit']:.4f}")
    print(f"Validation policy Sharpe: {val_policy_metrics['policy_annualized_sharpe']:.4f}")
    print(f"Saved experiment artifacts to {logger.run_dir}")
