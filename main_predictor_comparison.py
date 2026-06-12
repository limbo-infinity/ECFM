import copy
import os
import random

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
    FMTSPricePredictor,
    PricePredictor,
    evaluate_prediction_model,
    train_fmts_prediction_model,
    train_prediction_model,
)
from train import train_end_to_end_direct_bid, train_end_to_end_fixed_q_bid
from virtual_trading_simulator import (
    evaluate_direct_bid_policy,
    evaluate_dpds_policy,
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


def plot_predictor_losses(mlp_losses, fmts_losses, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    mlp_epochs = range(1, len(mlp_losses["train"]) + 1)
    fmts_epochs = range(1, len(fmts_losses["train"]) + 1)

    axes[0].plot(mlp_epochs, mlp_losses["train"], marker="o", label="MLP train MSE")
    if mlp_losses["val"]:
        axes[0].plot(mlp_epochs, mlp_losses["val"], marker="o", label="MLP val MSE")
    axes[0].set_ylabel("Spread MSE")
    axes[0].set_title("MLP predictor")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(fmts_epochs, fmts_losses["train"], marker="o", label="FM-TS train flow loss")
    if fmts_losses["val"]:
        axes[1].plot(fmts_epochs, fmts_losses["val"], marker="o", label="FM-TS val spread MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("FM-TS predictor")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy_profits(policy_profit_series, target_dates, output_path):
    min_len = min(len(values) for values in policy_profit_series.values())
    x = np.arange(min_len)
    dates = list(target_dates)[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, values in policy_profit_series.items():
        profits = np.asarray(values, dtype=np.float32).reshape(-1)[:min_len]
        line = axes[0].plot(
            x,
            profits,
            linewidth=0.45,
            alpha=0.12,
            label="_nolegend_",
        )[0]
        if profits.size >= 30:
            rolling_mean = np.convolve(profits, np.ones(30) / 30.0, mode="valid")
            rolling_x = x[29:]
            axes[0].plot(
                rolling_x,
                rolling_mean,
                linewidth=2.0,
                color=line.get_color(),
                label=f"{label} 30-day mean",
            )
        axes[1].plot(x, np.cumsum(profits), linewidth=1.8, marker=None, label=label)

    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Daily profit")
    axes[1].set_ylabel("Cumulative profit")
    axes[1].set_xlabel("Validation day")
    axes[0].set_title("Validation daily profit, faint raw series with 30-day means")
    axes[1].set_title("Validation cumulative profit")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    if dates:
        tick_count = min(8, len(dates))
        tick_positions = np.linspace(0, len(dates) - 1, tick_count, dtype=int)
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels([dates[i] for i in tick_positions], rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def prediction_output_to_spread(prediction_output):
    if isinstance(prediction_output, tuple):
        da_pred, rt_pred = prediction_output
        return da_pred - rt_pred
    return prediction_output


def collect_spread_predictions(model, data_loader, device):
    predicted_spreads = []
    true_spreads = []
    model.eval()
    with torch.no_grad():
        for xi, da_true, rt_true in data_loader:
            xi = xi.to(device)
            da_true = da_true.to(device)
            rt_true = rt_true.to(device)
            predicted_spread = prediction_output_to_spread(model(xi))
            true_spread = da_true - rt_true
            predicted_spreads.append(predicted_spread.detach().cpu())
            true_spreads.append(true_spread.detach().cpu())

    return {
        "predicted": torch.cat(predicted_spreads, dim=0).numpy(),
        "true": torch.cat(true_spreads, dim=0).numpy(),
    }


def plot_spread_scatter(spread_predictions, output_path, max_points=30000):
    fig, axes = plt.subplots(1, len(spread_predictions), figsize=(12, 5), sharex=True, sharey=True)
    if len(spread_predictions) == 1:
        axes = [axes]

    for ax, (label, values) in zip(axes, spread_predictions.items()):
        true_spread = np.asarray(values["true"], dtype=np.float32).reshape(-1)
        predicted_spread = np.asarray(values["predicted"], dtype=np.float32).reshape(-1)
        if true_spread.size > max_points:
            rng = np.random.default_rng(0)
            indices = rng.choice(true_spread.size, size=max_points, replace=False)
            true_spread = true_spread[indices]
            predicted_spread = predicted_spread[indices]

        ax.scatter(true_spread, predicted_spread, s=5, alpha=0.18)
        min_val = float(min(true_spread.min(), predicted_spread.min()))
        max_val = float(max(true_spread.max(), predicted_spread.max()))
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1.0)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("True spread")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Predicted spread")

    fig.suptitle("Validation predicted spread vs true spread")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def direction_accuracy_diagnostics(spread_predictions, thresholds=(5.0, 10.0, 25.0)):
    diagnostics = {}
    for label, values in spread_predictions.items():
        true_spread = np.asarray(values["true"], dtype=np.float32).reshape(-1)
        predicted_spread = np.asarray(values["predicted"], dtype=np.float32).reshape(-1)
        true_nonzero = true_spread != 0.0
        correct_all = np.sign(predicted_spread) == np.sign(true_spread)
        correct = correct_all[true_nonzero]
        weights = np.abs(true_spread[true_nonzero])
        weight_sum = weights.sum()

        model_diagnostics = {
            "raw": float(correct.mean()) if correct.size else 0.0,
            "weighted": float((correct * weights).sum() / weight_sum)
            if weight_sum > 0
            else 0.0,
            "zero_rate": float((np.abs(predicted_spread) < 1e-6).mean()),
        }
        for threshold in thresholds:
            mask = np.abs(true_spread) > threshold
            threshold_correct = correct_all[mask]
            model_diagnostics[f"abs_gt_{threshold:g}"] = (
                float(threshold_correct.mean()) if threshold_correct.size else 0.0
            )
            model_diagnostics[f"abs_gt_{threshold:g}_count"] = int(mask.sum())
        diagnostics[label] = model_diagnostics

    return diagnostics


def plot_direction_accuracy(spread_predictions, output_path, thresholds=(5.0, 10.0, 25.0)):
    diagnostics = direction_accuracy_diagnostics(
        spread_predictions,
        thresholds=thresholds,
    )
    labels = list(diagnostics)
    metric_names = ["raw", "weighted"] + [
        f"abs_gt_{threshold:g}" for threshold in thresholds
    ]
    metric_labels = ["Raw", "|spread|-weighted"] + [
        f"|spread|>{threshold:g}" for threshold in thresholds
    ]
    values = np.asarray(
        [
            [diagnostics[label][metric_name] for metric_name in metric_names]
            for label in labels
        ],
        dtype=np.float32,
    )

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label_idx, label in enumerate(labels):
        offsets = x - 0.4 + width / 2 + label_idx * width
        bars = ax.bar(offsets, values[label_idx], width=width, label=label)
        for bar, value in zip(bars, values[label_idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Direction accuracy")
    ax.set_title("Validation spread direction accuracy diagnostics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    zero_note = " | ".join(
        [
            f"{label} zero-rate {diagnostics[label]['zero_rate']:.3f}"
            for label in labels
        ]
    )
    ax.text(
        0.99,
        0.02,
        zero_note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return diagnostics


def plot_profit_distribution_and_drawdown(policy_profit_series, output_path):
    labels = list(policy_profit_series)
    profits = [
        np.asarray(policy_profit_series[label], dtype=np.float32).reshape(-1)
        for label in labels
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].boxplot(profits, labels=labels, showfliers=False)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Daily profit")
    axes[0].set_title("Validation daily profit distribution")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=25)

    for label, values in zip(labels, profits):
        cumulative = np.cumsum(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        axes[1].plot(drawdown, linewidth=1.4, label=label)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Validation day")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Validation cumulative-profit drawdown")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_fmts_inference_ablation(model, val_loader, device, samples_grid, steps_grid):
    original_samples = model.num_prediction_samples
    original_steps = model.num_prediction_steps
    heatmap = np.zeros((len(samples_grid), len(steps_grid)), dtype=np.float32)
    try:
        for row_idx, sample_count in enumerate(samples_grid):
            for col_idx, step_count in enumerate(steps_grid):
                model.num_prediction_samples = int(sample_count)
                model.num_prediction_steps = int(step_count)
                val_mse = evaluate_prediction_model(
                    model,
                    val_loader,
                    device=device,
                    loss_mode="spread",
                )
                heatmap[row_idx, col_idx] = val_mse
                print(
                    "[fmts_ablation] "
                    f"samples={sample_count}, steps={step_count}, "
                    f"val spread MSE={val_mse:.4f}",
                    flush=True,
                )
    finally:
        model.num_prediction_samples = original_samples
        model.num_prediction_steps = original_steps

    return heatmap


def plot_fmts_ablation_heatmap(heatmap, samples_grid, steps_grid, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(steps_grid)))
    ax.set_xticklabels([str(value) for value in steps_grid])
    ax.set_yticks(np.arange(len(samples_grid)))
    ax.set_yticklabels([str(value) for value in samples_grid])
    ax.set_xlabel("FM-TS prediction flow steps")
    ax.set_ylabel("FM-TS prediction samples")
    ax.set_title("FM-TS inference ablation: validation spread MSE")
    fig.colorbar(im, ax=ax, label="Validation spread MSE")

    for row_idx in range(heatmap.shape[0]):
        for col_idx in range(heatmap.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{heatmap[row_idx, col_idx]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_fmts_decision_sample_ablation(combo_results, val_loader, device, hyperparams):
    sample_grid = hyperparams["fmts_decision_ablation_samples"]
    rows = {}

    for combo_key in ["fmts_direct", "fmts_ecfm"]:
        result = combo_results.get(combo_key)
        if result is None:
            continue

        predictor = result["predictor"]
        original_samples = predictor.num_prediction_samples
        original_steps = predictor.num_prediction_steps
        rows[combo_key] = {
            "samples": [],
            "total_profit": [],
            "annualized_sharpe": [],
            "spread_mse": [],
            "latency_ms_per_day": [],
        }

        try:
            for sample_count in sample_grid:
                predictor.num_prediction_samples = int(sample_count)
                predictor.num_prediction_steps = hyperparams["fmts_prediction_steps"]

                if combo_key.endswith("_direct"):
                    metrics, _ = evaluate_direct_bid_policy(
                        prediction_model=predictor,
                        bid_model=result["decision_model"],
                        data_loader=val_loader,
                        device=device,
                        low=hyperparams["bid_low"],
                        up=hyperparams["bid_up"],
                        budget=hyperparams["budget"],
                        q_max=hyperparams["q_max"],
                        alpha=hyperparams["alpha"],
                        penalty_weight=hyperparams["penalty_weight"],
                        use_soft_q=hyperparams["eval_soft_q"],
                        q_temperature=hyperparams["q_temperature"],
                    )
                else:
                    metrics, _ = evaluate_fixed_q_bid_policy(
                        prediction_model=predictor,
                        ecfm_model=result["decision_model"],
                        data_loader=val_loader,
                        device=device,
                        low=hyperparams["bid_low"],
                        up=hyperparams["bid_up"],
                        budget=hyperparams["budget"],
                        q_max=hyperparams["q_max"],
                        alpha=hyperparams["alpha"],
                        penalty_weight=hyperparams["penalty_weight"],
                        num_samples=hyperparams["eval_num_samples"],
                        num_flow_steps=hyperparams["eval_num_flow_steps"],
                        use_langevin=hyperparams["langevin_steps"] > 0,
                        langevin_steps=hyperparams["langevin_steps"],
                        langevin_step_size=hyperparams["langevin_step_size"],
                        use_soft_q=hyperparams["eval_soft_q"],
                        q_temperature=hyperparams["q_temperature"],
                    )

                rows[combo_key]["samples"].append(int(sample_count))
                rows[combo_key]["total_profit"].append(metrics["policy_total_profit"])
                rows[combo_key]["annualized_sharpe"].append(
                    metrics["policy_annualized_sharpe"]
                )
                rows[combo_key]["spread_mse"].append(metrics["spread_mse"])
                rows[combo_key]["latency_ms_per_day"].append(
                    metrics["latency_ms_per_day"]
                )
                print(
                    "[fmts_decision_ablation] "
                    f"{combo_key}, predictor_samples={sample_count}, "
                    f"profit={metrics['policy_total_profit']:.4f}, "
                    f"sharpe={metrics['policy_annualized_sharpe']:.4f}, "
                    f"spread MSE={metrics['spread_mse']:.4f}",
                    flush=True,
                )
        finally:
            predictor.num_prediction_samples = original_samples
            predictor.num_prediction_steps = original_steps

    return {
        combo_key: {
            metric_name: np.asarray(metric_values, dtype=np.float32)
            for metric_name, metric_values in combo_rows.items()
        }
        for combo_key, combo_rows in rows.items()
    }


def plot_fmts_decision_sample_ablation(ablation, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_specs = [
        ("total_profit", "Validation total profit"),
        ("annualized_sharpe", "Validation annualized Sharpe"),
        ("spread_mse", "Validation spread MSE"),
        ("latency_ms_per_day", "Latency ms/day"),
    ]

    for ax, (metric_name, ylabel) in zip(axes.reshape(-1), metric_specs):
        for combo_key, values in ablation.items():
            ax.plot(
                values["samples"],
                values[metric_name],
                marker="o",
                linewidth=1.8,
                label=combo_key,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[1, 0].set_xlabel("FM-TS prediction samples")
    axes[1, 1].set_xlabel("FM-TS prediction samples")
    fig.suptitle("FM-TS prediction sample ablation: decision quality")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_alpha_ablation(
    combo_results,
    val_loader,
    device,
    hyperparams,
    da_price_matrix=None,
    rt_price_matrix=None,
):
    alpha_grid = hyperparams["alpha_ablation_values"]
    rows = {}

    for combo_key, result in combo_results.items():
        rows[combo_key] = {
            "alpha": [],
            "total_profit": [],
            "annualized_sharpe": [],
            "latency_ms_per_day": [],
        }
        for alpha_value in alpha_grid:
            if combo_key.endswith("_direct"):
                metrics, _ = evaluate_direct_bid_policy(
                    prediction_model=result["predictor"],
                    bid_model=result["decision_model"],
                    data_loader=val_loader,
                    device=device,
                    low=hyperparams["bid_low"],
                    up=hyperparams["bid_up"],
                    budget=hyperparams["budget"],
                    q_max=hyperparams["q_max"],
                    alpha=float(alpha_value),
                    penalty_weight=hyperparams["penalty_weight"],
                    use_soft_q=hyperparams["eval_soft_q"],
                    q_temperature=hyperparams["q_temperature"],
                )
            else:
                metrics, _ = evaluate_fixed_q_bid_policy(
                    prediction_model=result["predictor"],
                    ecfm_model=result["decision_model"],
                    data_loader=val_loader,
                    device=device,
                    low=hyperparams["bid_low"],
                    up=hyperparams["bid_up"],
                    budget=hyperparams["budget"],
                    q_max=hyperparams["q_max"],
                    alpha=float(alpha_value),
                    penalty_weight=hyperparams["penalty_weight"],
                    num_samples=hyperparams["eval_num_samples"],
                    num_flow_steps=hyperparams["eval_num_flow_steps"],
                    use_langevin=hyperparams["langevin_steps"] > 0,
                    langevin_steps=hyperparams["langevin_steps"],
                    langevin_step_size=hyperparams["langevin_step_size"],
                    use_soft_q=hyperparams["eval_soft_q"],
                    q_temperature=hyperparams["q_temperature"],
                )

            rows[combo_key]["alpha"].append(float(alpha_value))
            rows[combo_key]["total_profit"].append(metrics["policy_total_profit"])
            rows[combo_key]["annualized_sharpe"].append(
                metrics["policy_annualized_sharpe"]
            )
            rows[combo_key]["latency_ms_per_day"].append(metrics["latency_ms_per_day"])
            print(
                "[alpha_ablation] "
                f"{combo_key}, alpha={alpha_value:g}, "
                f"profit={metrics['policy_total_profit']:.4f}, "
                f"sharpe={metrics['policy_annualized_sharpe']:.4f}",
                flush=True,
            )

    if da_price_matrix is not None and rt_price_matrix is not None:
        rows["dpds"] = {
            "alpha": [],
            "total_profit": [],
            "annualized_sharpe": [],
            "latency_ms_per_day": [],
        }
        for alpha_value in alpha_grid:
            metrics, _ = evaluate_dpds_policy(
                da_prices=da_price_matrix,
                rt_prices=rt_price_matrix,
                target_indices=val_loader.dataset.target_indices,
                gap=hyperparams["gap"],
                budget=hyperparams["budget"],
                bid_low=hyperparams["bid_low"],
                bid_up=hyperparams["bid_up"],
                q_max=hyperparams["q_max"],
                alpha=float(alpha_value),
                grid_size=hyperparams["dpds_grid_size"],
                rho=hyperparams["dpds_rho"],
            )
            rows["dpds"]["alpha"].append(float(alpha_value))
            rows["dpds"]["total_profit"].append(metrics["policy_total_profit"])
            rows["dpds"]["annualized_sharpe"].append(
                metrics["policy_annualized_sharpe"]
            )
            rows["dpds"]["latency_ms_per_day"].append(metrics["latency_ms_per_day"])
            print(
                "[alpha_ablation] "
                f"dpds, alpha={alpha_value:g}, "
                f"profit={metrics['policy_total_profit']:.4f}, "
                f"sharpe={metrics['policy_annualized_sharpe']:.4f}",
                flush=True,
            )

    return {
        policy_name: {
            metric_name: np.asarray(metric_values, dtype=np.float32)
            for metric_name, metric_values in policy_rows.items()
        }
        for policy_name, policy_rows in rows.items()
    }


def plot_alpha_ablation(alpha_ablation, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for policy_name, values in alpha_ablation.items():
        axes[0].plot(
            values["alpha"],
            values["total_profit"],
            marker="o",
            linewidth=1.8,
            label=policy_name,
        )
        axes[1].plot(
            values["alpha"],
            values["annualized_sharpe"],
            marker="o",
            linewidth=1.8,
            label=policy_name,
        )

    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Validation total profit")
    axes[1].set_ylabel("Validation annualized Sharpe")
    axes[1].set_xlabel("Clearing sigmoid alpha")
    axes[0].set_title("Alpha ablation: clearing hardness sensitivity")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def e2e_ablation_rows(hyperparams):
    spread_weight = hyperparams["e2e_ablation_spread_regularizer_weight"]
    sharpe_weight = hyperparams["e2e_ablation_sharpe_loss_weight"]
    return [
        {
            "name": "frozen_energy",
            "freeze_prediction_during_e2e": True,
            "prediction_regularizer_weight": 0.0,
            "risk_loss_mode": "none",
            "sharpe_loss_weight": 0.0,
        },
        {
            "name": "unfrozen_energy",
            "freeze_prediction_during_e2e": False,
            "prediction_regularizer_weight": 0.0,
            "risk_loss_mode": "none",
            "sharpe_loss_weight": 0.0,
        },
        {
            "name": "unfrozen_spread",
            "freeze_prediction_during_e2e": False,
            "prediction_regularizer_weight": spread_weight,
            "risk_loss_mode": "none",
            "sharpe_loss_weight": 0.0,
        },
        {
            "name": "unfrozen_sharpe",
            "freeze_prediction_during_e2e": False,
            "prediction_regularizer_weight": 0.0,
            "risk_loss_mode": "sharpe",
            "sharpe_loss_weight": sharpe_weight,
        },
        {
            "name": "unfrozen_spread_sharpe",
            "freeze_prediction_during_e2e": False,
            "prediction_regularizer_weight": spread_weight,
            "risk_loss_mode": "sharpe",
            "sharpe_loss_weight": sharpe_weight,
        },
    ]


def run_mlp_ecfm_e2e_regularizer_ablation(
    base_predictor,
    train_loader,
    val_loader,
    device,
    hyperparams,
):
    results = {}
    for row in e2e_ablation_rows(hyperparams):
        ablation_hyperparams = copy.deepcopy(hyperparams)
        ablation_hyperparams.update(row)
        ablation_hyperparams["end_to_end_epochs"] = hyperparams["e2e_ablation_epochs"]
        predictor = copy.deepcopy(base_predictor).to(device)
        combo_name = f"mlp_ecfm_ablation_{row['name']}"
        print(
            "[e2e_ablation] "
            f"{row['name']} | freeze={row['freeze_prediction_during_e2e']} | "
            f"spread_weight={row['prediction_regularizer_weight']} | "
            f"risk={row['risk_loss_mode']} | "
            f"sharpe_weight={row['sharpe_loss_weight']}",
            flush=True,
        )
        decision_model, losses, metrics, arrays = train_ecfm_decision(
            combo_name,
            predictor,
            train_loader,
            val_loader,
            device,
            ablation_hyperparams,
        )
        final_val_spread_mse = evaluate_prediction_model(
            predictor,
            val_loader,
            device=device,
            loss_mode="spread",
        )
        results[row["name"]] = {
            "predictor": predictor,
            "decision_model": decision_model,
            "losses": losses,
            "metrics": metrics,
            "arrays": arrays,
            "config": row,
            "final_val_spread_mse": final_val_spread_mse,
        }
        print(
            "[e2e_ablation] "
            f"{row['name']} finished | "
            f"profit={metrics['policy_total_profit']:.4f} | "
            f"sharpe={metrics['policy_annualized_sharpe']:.4f} | "
            f"val spread MSE={final_val_spread_mse:.4f}",
            flush=True,
        )

    return results


def plot_e2e_regularizer_ablation(ablation_results, output_path):
    labels = [
        label
        for label in ablation_results
        if label != "unfrozen_spread"
    ]
    x = np.arange(len(labels))
    total_profit = np.asarray(
        [
            ablation_results[label]["metrics"]["policy_total_profit"]
            for label in labels
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(x, total_profit)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Validation total profit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, total_profit):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    fig.suptitle("MLP + ECFM end-to-end ablation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_direct_decision(
    combo_name,
    predictor,
    train_loader,
    val_loader,
    device,
    hyperparams,
):
    print(f"[{combo_name}] Training Direct MLP decision model...", flush=True)
    bid_model = DirectBidMLP(
        T=hyperparams["T"],
        K=hyperparams["K"],
        hidden_dim=hyperparams["direct_bid_hidden_dim"],
        num_hidden_layers=hyperparams["direct_bid_hidden_layers"],
    ).to(device)
    optimizer_parameters = list(bid_model.parameters())
    if not hyperparams["freeze_prediction_during_e2e"]:
        optimizer_parameters += list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=hyperparams["end_to_end_learning_rate"],
        weight_decay=hyperparams["end_to_end_weight_decay"],
    )
    losses = train_end_to_end_direct_bid(
        prediction_model=predictor,
        bid_model=bid_model,
        train_loader=train_loader,
        optimizer=optimizer,
        low=hyperparams["bid_low"],
        up=hyperparams["bid_up"],
        budget=hyperparams["budget"],
        q_max=hyperparams["q_max"],
        device=device,
        num_epochs=hyperparams["end_to_end_epochs"],
        alpha=hyperparams["alpha"],
        penalty_weight=hyperparams["penalty_weight"],
        prediction_regularizer_weight=hyperparams["prediction_regularizer_weight"],
        risk_loss_mode=hyperparams["risk_loss_mode"],
        sharpe_loss_weight=hyperparams["sharpe_loss_weight"],
        variance_loss_weight=hyperparams["variance_loss_weight"],
        train_soft_q=hyperparams["train_soft_q"],
        q_temperature=hyperparams["q_temperature"],
        freeze_prediction_model=hyperparams["freeze_prediction_during_e2e"],
    )
    print(f"[{combo_name}] Evaluating Direct MLP decision model...", flush=True)
    metrics, arrays = evaluate_direct_bid_policy(
        prediction_model=predictor,
        bid_model=bid_model,
        data_loader=val_loader,
        device=device,
        low=hyperparams["bid_low"],
        up=hyperparams["bid_up"],
        budget=hyperparams["budget"],
        q_max=hyperparams["q_max"],
        alpha=hyperparams["alpha"],
        penalty_weight=hyperparams["penalty_weight"],
        use_soft_q=hyperparams["eval_soft_q"],
        q_temperature=hyperparams["q_temperature"],
    )
    print(f"[{combo_name}] Finished evaluation.", flush=True)
    return bid_model, losses, metrics, arrays


def train_ecfm_decision(
    combo_name,
    predictor,
    train_loader,
    val_loader,
    device,
    hyperparams,
):
    print(f"[{combo_name}] Training ECFM decision model...", flush=True)
    ecfm_model = ECFMModel(
        T=hyperparams["T"],
        K=hyperparams["K"],
        hidden_dim=hyperparams["flow_hidden_dim"],
        decision_dim=1,
    ).to(device)
    optimizer_parameters = list(ecfm_model.parameters())
    if not hyperparams["freeze_prediction_during_e2e"]:
        optimizer_parameters += list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=hyperparams["end_to_end_learning_rate"],
        weight_decay=hyperparams["end_to_end_weight_decay"],
    )
    losses = train_end_to_end_fixed_q_bid(
        prediction_model=predictor,
        ecfm_model=ecfm_model,
        train_loader=train_loader,
        optimizer=optimizer,
        low=hyperparams["bid_low"],
        up=hyperparams["bid_up"],
        budget=hyperparams["budget"],
        q_max=hyperparams["q_max"],
        device=device,
        num_epochs=hyperparams["end_to_end_epochs"],
        num_samples=hyperparams["train_num_samples"],
        num_flow_steps=hyperparams["train_num_flow_steps"],
        alpha=hyperparams["alpha"],
        penalty_weight=hyperparams["penalty_weight"],
        prediction_regularizer_weight=hyperparams["prediction_regularizer_weight"],
        risk_loss_mode=hyperparams["risk_loss_mode"],
        sharpe_loss_weight=hyperparams["sharpe_loss_weight"],
        variance_loss_weight=hyperparams["variance_loss_weight"],
        train_soft_q=hyperparams["train_soft_q"],
        q_temperature=hyperparams["q_temperature"],
        freeze_prediction_model=hyperparams["freeze_prediction_during_e2e"],
    )
    print(
        f"[{combo_name}] Evaluating ECFM decision model "
        f"(samples={hyperparams['eval_num_samples']}, "
        f"flow_steps={hyperparams['eval_num_flow_steps']}, "
        f"langevin_steps={hyperparams['langevin_steps']})...",
        flush=True,
    )
    metrics, arrays = evaluate_fixed_q_bid_policy(
        prediction_model=predictor,
        ecfm_model=ecfm_model,
        data_loader=val_loader,
        device=device,
        low=hyperparams["bid_low"],
        up=hyperparams["bid_up"],
        budget=hyperparams["budget"],
        q_max=hyperparams["q_max"],
        alpha=hyperparams["alpha"],
        penalty_weight=hyperparams["penalty_weight"],
        num_samples=hyperparams["eval_num_samples"],
        num_flow_steps=hyperparams["eval_num_flow_steps"],
        use_langevin=hyperparams["langevin_steps"] > 0,
        langevin_steps=hyperparams["langevin_steps"],
        langevin_step_size=hyperparams["langevin_step_size"],
        use_soft_q=hyperparams["eval_soft_q"],
        q_temperature=hyperparams["q_temperature"],
    )
    print(f"[{combo_name}] Finished evaluation.", flush=True)
    return ecfm_model, losses, metrics, arrays


if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    history_len = 30
    gap = 2
    batch_size = 32
    normalize_inputs = True
    include_time_features = True
    prediction_hidden_dim = 512
    prediction_num_hidden_layers = 5
    fmts_hidden_dim = 384
    fmts_num_hidden_layers = 3
    fmts_output_mode = "spread"
    fmts_prediction_samples = 20
    fmts_prediction_steps = 5
    fmts_ablation_samples = [8, 12, 16, 20]
    fmts_ablation_steps = [4, 8, 16, 32]
    fmts_decision_ablation_samples = [1, 2, 4, 8, 12, 16, 20]
    run_fmts_inference_ablation = False
    run_fmts_decision_ablation = False
    run_alpha_ablation_study = True
    alpha_ablation_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    direct_bid_hidden_dim = 512
    direct_bid_hidden_layers = 3
    flow_hidden_dim = 512
    prediction_pretrain_epochs = 30
    fmts_pretrain_epochs = 30
    end_to_end_epochs = 100
    prediction_learning_rate = 1e-4
    fmts_learning_rate = 1e-4
    end_to_end_learning_rate = 5e-5
    prediction_weight_decay = 1e-3
    fmts_weight_decay = 1e-4
    end_to_end_weight_decay = 1e-4
    freeze_prediction_during_e2e = False
    prediction_regularizer_weight = 1e-2
    risk_loss_mode = "none"
    sharpe_loss_weight = 5
    variance_loss_weight = 5
    run_e2e_regularizer_ablation = False
    e2e_ablation_epochs = 20
    e2e_ablation_spread_regularizer_weight = 1e-2
    e2e_ablation_sharpe_loss_weight = 10.0
    train_soft_q = False
    eval_soft_q = False
    q_temperature = 10.0
    train_num_samples = 20
    train_num_flow_steps = 10
    eval_num_samples = 20
    eval_num_flow_steps = 8
    langevin_steps = 0
    langevin_step_size = 1e-4
    dpds_grid_size = 25
    dpds_rho = 0.002
    alpha = 5.0
    penalty_weight = 1.0
    q_max = 1.0
    budget = 250000.0
    bid_low = -1000.0
    bid_up = 2000.0

    train_loader, val_loader, metadata = load_train_val_from_zone_data(
        history_len=history_len,
        gap=gap,
        batch_size=batch_size,
        normalize_inputs=normalize_inputs,
        include_time_features=include_time_features,
    )
    T = 1
    K = metadata["K"]
    obs_dim = train_loader.dataset.obs_dim
    device = get_training_device()

    hyperparams = {
        "history_len": history_len,
        "gap": gap,
        "batch_size": batch_size,
        "normalize_inputs": normalize_inputs,
        "include_time_features": include_time_features,
        "time_feature_dim": metadata["time_feature_dim"],
        "prediction_hidden_dim": prediction_hidden_dim,
        "prediction_num_hidden_layers": prediction_num_hidden_layers,
        "fmts_hidden_dim": fmts_hidden_dim,
        "fmts_num_hidden_layers": fmts_num_hidden_layers,
        "fmts_output_mode": fmts_output_mode,
        "fmts_prediction_samples": fmts_prediction_samples,
        "fmts_prediction_steps": fmts_prediction_steps,
        "fmts_ablation_samples": fmts_ablation_samples,
        "fmts_ablation_steps": fmts_ablation_steps,
        "fmts_decision_ablation_samples": fmts_decision_ablation_samples,
        "run_fmts_inference_ablation": run_fmts_inference_ablation,
        "run_fmts_decision_ablation": run_fmts_decision_ablation,
        "run_alpha_ablation_study": run_alpha_ablation_study,
        "alpha_ablation_values": alpha_ablation_values,
        "direct_bid_hidden_dim": direct_bid_hidden_dim,
        "direct_bid_hidden_layers": direct_bid_hidden_layers,
        "flow_hidden_dim": flow_hidden_dim,
        "prediction_pretrain_epochs": prediction_pretrain_epochs,
        "fmts_pretrain_epochs": fmts_pretrain_epochs,
        "end_to_end_epochs": end_to_end_epochs,
        "prediction_learning_rate": prediction_learning_rate,
        "fmts_learning_rate": fmts_learning_rate,
        "end_to_end_learning_rate": end_to_end_learning_rate,
        "prediction_weight_decay": prediction_weight_decay,
        "fmts_weight_decay": fmts_weight_decay,
        "end_to_end_weight_decay": end_to_end_weight_decay,
        "freeze_prediction_during_e2e": freeze_prediction_during_e2e,
        "prediction_regularizer_weight": prediction_regularizer_weight,
        "risk_loss_mode": risk_loss_mode,
        "sharpe_loss_weight": sharpe_loss_weight,
        "variance_loss_weight": variance_loss_weight,
        "run_e2e_regularizer_ablation": run_e2e_regularizer_ablation,
        "e2e_ablation_epochs": e2e_ablation_epochs,
        "e2e_ablation_spread_regularizer_weight": (
            e2e_ablation_spread_regularizer_weight
        ),
        "e2e_ablation_sharpe_loss_weight": e2e_ablation_sharpe_loss_weight,
        "train_soft_q": train_soft_q,
        "eval_soft_q": eval_soft_q,
        "q_temperature": q_temperature,
        "train_num_samples": train_num_samples,
        "train_num_flow_steps": train_num_flow_steps,
        "eval_num_samples": eval_num_samples,
        "eval_num_flow_steps": eval_num_flow_steps,
        "langevin_steps": langevin_steps,
        "langevin_step_size": langevin_step_size,
        "dpds_grid_size": dpds_grid_size,
        "dpds_rho": dpds_rho,
        "alpha": alpha,
        "penalty_weight": penalty_weight,
        "q_max": q_max,
        "budget": budget,
        "bid_low": bid_low,
        "bid_up": bid_up,
        "T": T,
        "K": K,
    }

    logger = ExperimentLogger(run_name=f"predictor_decision_comparison_seed{seed}")
    logger.save_config(
        {
            "model_type": "predictor_decision_comparison",
            "seed": seed,
            "device": str(device),
            "data": {
                "source": "zone_data",
                "num_days": metadata["num_days"],
                "num_zones": metadata["num_zones"],
                "hours_per_day": metadata["hours_per_day"],
                "K": metadata["K"],
                "obs_dim": metadata["obs_dim"],
                "normalize_inputs": metadata["normalize_inputs"],
                "include_time_features": metadata["include_time_features"],
                "time_feature_dim": metadata["time_feature_dim"],
                "start_date": metadata["dates"][0],
                "end_date": metadata["dates"][-1],
            },
            "splits": {
                "train": split_summary(train_loader, metadata),
                "val": split_summary(val_loader, metadata),
                "test": None,
            },
            "hyperparams": hyperparams,
            "comparison_grid": {
                "rows": ["MLP predictor", "FM-TS predictor"],
                "columns": ["Direct MLP decision", "ECFM decision"],
                "baseline": "DPDS",
            },
        }
    )

    print(f"Experiment directory: {logger.run_dir}")
    print(f"Using device: {device}")
    print(f"Training days: {len(train_loader.dataset)}")
    print(f"Validation days: {len(val_loader.dataset)}")
    print(f"obs_dim: {obs_dim}, T: {T}, K: {K}")

    mlp_predictor = PricePredictor(
        obs_dim=obs_dim,
        hidden_dim=prediction_hidden_dim,
        T=T,
        K=K,
        num_hidden_layers=prediction_num_hidden_layers,
        output_mode="prices",
    ).to(device)
    mlp_optimizer = torch.optim.AdamW(
        mlp_predictor.parameters(),
        lr=prediction_learning_rate,
        weight_decay=prediction_weight_decay,
    )
    mlp_losses = train_prediction_model(
        g_phi=mlp_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=mlp_optimizer,
        num_epochs=prediction_pretrain_epochs,
        loss_mode="spread",
    )
    mlp_pretrain_train_loss = evaluate_prediction_model(
        mlp_predictor,
        train_loader,
        device=device,
        loss_mode="spread",
    )
    mlp_pretrain_val_loss = evaluate_prediction_model(
        mlp_predictor,
        val_loader,
        device=device,
        loss_mode="spread",
    )

    fmts_predictor = FMTSPricePredictor(
        obs_dim=obs_dim,
        hidden_dim=fmts_hidden_dim,
        T=T,
        K=K,
        num_hidden_layers=fmts_num_hidden_layers,
        output_mode=fmts_output_mode,
        num_prediction_samples=fmts_prediction_samples,
        num_prediction_steps=fmts_prediction_steps,
    ).to(device)
    fmts_optimizer = torch.optim.AdamW(
        fmts_predictor.parameters(),
        lr=fmts_learning_rate,
        weight_decay=fmts_weight_decay,
    )
    fmts_losses = train_fmts_prediction_model(
        g_phi=fmts_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=fmts_optimizer,
        num_epochs=fmts_pretrain_epochs,
        loss_mode="spread",
    )
    fmts_pretrain_train_loss = evaluate_prediction_model(
        fmts_predictor,
        train_loader,
        device=device,
        loss_mode="spread",
    )
    fmts_pretrain_val_loss = evaluate_prediction_model(
        fmts_predictor,
        val_loader,
        device=device,
        loss_mode="spread",
    )

    predictors = {
        "mlp": mlp_predictor,
        "fmts": fmts_predictor,
    }
    combo_results = {}
    policy_profit_series = {}

    for predictor_name, base_predictor in predictors.items():
        direct_predictor = copy.deepcopy(base_predictor).to(device)
        direct_key = f"{predictor_name}_direct"
        direct_model, direct_losses, direct_metrics, direct_arrays = train_direct_decision(
            direct_key,
            direct_predictor,
            train_loader,
            val_loader,
            device,
            hyperparams,
        )
        combo_results[direct_key] = {
            "predictor": direct_predictor,
            "decision_model": direct_model,
            "losses": direct_losses,
            "metrics": direct_metrics,
            "arrays": direct_arrays,
        }
        policy_profit_series[direct_key] = direct_arrays["profit"]

        ecfm_predictor = copy.deepcopy(base_predictor).to(device)
        ecfm_key = f"{predictor_name}_ecfm"
        ecfm_model, ecfm_losses, ecfm_metrics, ecfm_arrays = train_ecfm_decision(
            ecfm_key,
            ecfm_predictor,
            train_loader,
            val_loader,
            device,
            hyperparams,
        )
        combo_results[ecfm_key] = {
            "predictor": ecfm_predictor,
            "decision_model": ecfm_model,
            "losses": ecfm_losses,
            "metrics": ecfm_metrics,
            "arrays": ecfm_arrays,
        }
        policy_profit_series[ecfm_key] = ecfm_arrays["profit"]

    da_price_matrix = train_loader.dataset.da_prices.detach().cpu().numpy()
    rt_price_matrix = train_loader.dataset.rt_prices.detach().cpu().numpy()
    print(
        f"[dpds] Running DPDS baseline (grid_size={dpds_grid_size}, rho={dpds_rho})...",
        flush=True,
    )
    dpds_metrics, dpds_arrays = evaluate_dpds_policy(
        da_prices=da_price_matrix,
        rt_prices=rt_price_matrix,
        target_indices=val_loader.dataset.target_indices,
        gap=gap,
        budget=budget,
        bid_low=bid_low,
        bid_up=bid_up,
        q_max=q_max,
        alpha=alpha,
        grid_size=dpds_grid_size,
        rho=dpds_rho,
    )
    print("[dpds] Finished DPDS baseline.", flush=True)
    policy_profit_series["dpds"] = dpds_arrays["profit"]

    print("[save] Plotting and saving experiment artifacts...", flush=True)
    spread_predictions = {
        "MLP": collect_spread_predictions(mlp_predictor, val_loader, device),
        "FM-TS": collect_spread_predictions(fmts_predictor, val_loader, device),
    }
    plot_spread_scatter(
        spread_predictions,
        logger.path("spread_prediction_scatter.png"),
    )
    direction_diagnostics = plot_direction_accuracy(
        spread_predictions,
        logger.path("spread_direction_accuracy.png"),
    )
    plot_predictor_losses(
        mlp_losses,
        fmts_losses,
        logger.path("predictor_losses.png"),
    )
    validation_target_dates = [
        metadata["dates"][int(target_index)]
        for target_index in val_loader.dataset.target_indices
    ]
    plot_policy_profits(
        policy_profit_series,
        validation_target_dates,
        logger.path("validation_policy_profit_comparison.png"),
    )
    plot_profit_distribution_and_drawdown(
        policy_profit_series,
        logger.path("profit_distribution_drawdown.png"),
    )
    fmts_ablation_mse = None
    if run_fmts_inference_ablation:
        print("[fmts_ablation] Running inference ablation heatmap...", flush=True)
        fmts_ablation_mse = run_fmts_inference_ablation(
            fmts_predictor,
            val_loader,
            device,
            samples_grid=fmts_ablation_samples,
            steps_grid=fmts_ablation_steps,
        )
        plot_fmts_ablation_heatmap(
            fmts_ablation_mse,
            fmts_ablation_samples,
            fmts_ablation_steps,
            logger.path("fmts_samples_steps_ablation.png"),
        )
    fmts_decision_ablation = {}
    if run_fmts_decision_ablation:
        print(
            "[fmts_decision_ablation] Running decision-quality sample ablation...",
            flush=True,
        )
        fmts_decision_ablation = run_fmts_decision_sample_ablation(
            combo_results,
            val_loader,
            device,
            hyperparams,
        )
        plot_fmts_decision_sample_ablation(
            fmts_decision_ablation,
            logger.path("fmts_decision_sample_ablation.png"),
        )
    e2e_regularizer_ablation = {}
    if run_e2e_regularizer_ablation:
        print("[e2e_ablation] Running freeze/regularizer ablation...", flush=True)
        e2e_regularizer_ablation = run_mlp_ecfm_e2e_regularizer_ablation(
            mlp_predictor,
            train_loader,
            val_loader,
            device,
            hyperparams,
        )
        plot_e2e_regularizer_ablation(
            e2e_regularizer_ablation,
            logger.path("e2e_regularizer_ablation.png"),
        )
    alpha_ablation = {}
    if run_alpha_ablation_study:
        print("[alpha_ablation] Running alpha sensitivity study...", flush=True)
        alpha_ablation = run_alpha_ablation(
            combo_results,
            val_loader,
            device,
            hyperparams,
            da_price_matrix=da_price_matrix,
            rt_price_matrix=rt_price_matrix,
        )
        plot_alpha_ablation(
            alpha_ablation,
            logger.path("alpha_ablation.png"),
        )

    logger.save_npz(
        "prediction_losses.npz",
        mlp_train=np.asarray(mlp_losses["train"], dtype=np.float32),
        mlp_val=np.asarray(mlp_losses["val"], dtype=np.float32),
        fmts_train=np.asarray(fmts_losses["train"], dtype=np.float32),
        fmts_val=np.asarray(fmts_losses["val"], dtype=np.float32),
    )
    if train_loader.dataset.input_stats is not None:
        logger.save_npz(
            "input_normalization_stats.npz",
            da_mean=train_loader.dataset.input_stats["da_mean"].numpy(),
            da_std=train_loader.dataset.input_stats["da_std"].numpy(),
            rt_mean=train_loader.dataset.input_stats["rt_mean"].numpy(),
            rt_std=train_loader.dataset.input_stats["rt_std"].numpy(),
        )
    logger.save_npz(
        "spread_prediction_diagnostics.npz",
        mlp_predicted_spread=spread_predictions["MLP"]["predicted"],
        mlp_true_spread=spread_predictions["MLP"]["true"],
        fmts_predicted_spread=spread_predictions["FM-TS"]["predicted"],
        fmts_true_spread=spread_predictions["FM-TS"]["true"],
    )
    if fmts_ablation_mse is not None:
        logger.save_npz(
            "fmts_samples_steps_ablation.npz",
            samples=np.asarray(fmts_ablation_samples, dtype=np.int64),
            steps=np.asarray(fmts_ablation_steps, dtype=np.int64),
            val_spread_mse=fmts_ablation_mse,
        )
    if fmts_decision_ablation:
        logger.save_npz(
            "fmts_decision_sample_ablation.npz",
            **{
                f"{combo_key}_{metric_name}": metric_values
                for combo_key, combo_values in fmts_decision_ablation.items()
                for metric_name, metric_values in combo_values.items()
            },
        )
    if e2e_regularizer_ablation:
        logger.save_npz(
            "e2e_regularizer_ablation_metrics.npz",
            names=np.asarray(list(e2e_regularizer_ablation), dtype=str),
            total_profit=np.asarray(
                [
                    result["metrics"]["policy_total_profit"]
                    for result in e2e_regularizer_ablation.values()
                ],
                dtype=np.float32,
            ),
            annualized_sharpe=np.asarray(
                [
                    result["metrics"]["policy_annualized_sharpe"]
                    for result in e2e_regularizer_ablation.values()
                ],
                dtype=np.float32,
            ),
            final_val_spread_mse=np.asarray(
                [
                    result["final_val_spread_mse"]
                    for result in e2e_regularizer_ablation.values()
                ],
                dtype=np.float32,
            ),
        )
    if alpha_ablation:
        logger.save_npz(
            "alpha_ablation.npz",
            **{
                f"{policy_name}_{metric_name}": metric_values
                for policy_name, policy_values in alpha_ablation.items()
                for metric_name, metric_values in policy_values.items()
            },
        )
    logger.save_npz("dpds_val_policy_arrays.npz", **dpds_arrays)
    for key, result in combo_results.items():
        losses = result["losses"]
        logger.save_npz(
            f"{key}_losses.npz",
            total=np.asarray(losses["total"], dtype=np.float32),
            energy=np.asarray(losses["energy"], dtype=np.float32),
            risk=np.asarray(losses["risk"], dtype=np.float32),
            batch_sharpe=np.asarray(losses["batch_sharpe"], dtype=np.float32),
            profit_variance=np.asarray(losses["profit_variance"], dtype=np.float32),
        )
        logger.save_npz(f"{key}_val_policy_arrays.npz", **result["arrays"])
        logger.save_model(
            f"{key}_predictor.pt",
            result["predictor"],
            extra={"combo": key, "hyperparams": hyperparams},
        )
        logger.save_model(
            f"{key}_decision_model.pt",
            result["decision_model"],
            extra={"combo": key, "hyperparams": hyperparams},
        )
    for key, result in e2e_regularizer_ablation.items():
        losses = result["losses"]
        logger.save_npz(
            f"mlp_ecfm_ablation_{key}_losses.npz",
            total=np.asarray(losses["total"], dtype=np.float32),
            energy=np.asarray(losses["energy"], dtype=np.float32),
            spread_regularizer=np.asarray(
                losses["spread_regularizer"],
                dtype=np.float32,
            ),
            risk=np.asarray(losses["risk"], dtype=np.float32),
            batch_sharpe=np.asarray(losses["batch_sharpe"], dtype=np.float32),
            profit_variance=np.asarray(losses["profit_variance"], dtype=np.float32),
        )
        logger.save_npz(
            f"mlp_ecfm_ablation_{key}_val_policy_arrays.npz",
            **result["arrays"],
        )
        logger.save_model(
            f"mlp_ecfm_ablation_{key}_predictor.pt",
            result["predictor"],
            extra={
                "combo": f"mlp_ecfm_ablation_{key}",
                "hyperparams": hyperparams,
                "ablation_config": result["config"],
            },
        )
        logger.save_model(
            f"mlp_ecfm_ablation_{key}_decision_model.pt",
            result["decision_model"],
            extra={
                "combo": f"mlp_ecfm_ablation_{key}",
                "hyperparams": hyperparams,
                "ablation_config": result["config"],
            },
        )

    logger.save_model(
        "mlp_pretrained_predictor.pt",
        mlp_predictor,
        extra={
            "model_type": "mlp_price_predictor",
            "final_train_spread_mse": mlp_pretrain_train_loss,
            "final_val_spread_mse": mlp_pretrain_val_loss,
            "hyperparams": hyperparams,
        },
    )
    logger.save_model(
        "fmts_pretrained_predictor.pt",
        fmts_predictor,
        extra={
            "model_type": "fmts_price_predictor",
            "final_train_spread_mse": fmts_pretrain_train_loss,
            "final_val_spread_mse": fmts_pretrain_val_loss,
            "hyperparams": hyperparams,
        },
    )

    metrics = {
        "mlp_pretrain_train_spread_mse": mlp_pretrain_train_loss,
        "mlp_pretrain_val_spread_mse": mlp_pretrain_val_loss,
        "fmts_pretrain_train_spread_mse": fmts_pretrain_train_loss,
        "fmts_pretrain_val_spread_mse": fmts_pretrain_val_loss,
        "dpds_val_total_profit": dpds_metrics["policy_total_profit"],
        "dpds_val_annualized_sharpe": dpds_metrics["policy_annualized_sharpe"],
        **prefix_metrics("dpds_val_policy", dpds_metrics),
    }
    for predictor_name, predictor_diagnostics in direction_diagnostics.items():
        safe_name = predictor_name.lower().replace("-", "_")
        for metric_name, metric_value in predictor_diagnostics.items():
            metrics[f"{safe_name}_direction_{metric_name}"] = metric_value
    for combo_key, combo_ablation in fmts_decision_ablation.items():
        if combo_ablation["samples"].size == 0:
            continue
        best_profit_index = int(np.argmax(combo_ablation["total_profit"]))
        best_sharpe_index = int(np.argmax(combo_ablation["annualized_sharpe"]))
        metrics.update(
            {
                f"{combo_key}_sample_ablation_best_profit_sample": int(
                    combo_ablation["samples"][best_profit_index]
                ),
                f"{combo_key}_sample_ablation_best_profit": float(
                    combo_ablation["total_profit"][best_profit_index]
                ),
                f"{combo_key}_sample_ablation_best_sharpe_sample": int(
                    combo_ablation["samples"][best_sharpe_index]
                ),
                f"{combo_key}_sample_ablation_best_sharpe": float(
                    combo_ablation["annualized_sharpe"][best_sharpe_index]
                ),
            }
        )
    for ablation_name, result in e2e_regularizer_ablation.items():
        losses = result["losses"]
        combo_metrics = result["metrics"]
        metrics.update(
            {
                f"mlp_ecfm_ablation_{ablation_name}_final_total_loss": (
                    losses["total"][-1]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_final_energy_loss": (
                    losses["energy"][-1]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_final_risk_loss": (
                    losses["risk"][-1]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_final_spread_regularizer": (
                    losses["spread_regularizer"][-1]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_final_batch_sharpe": (
                    losses["batch_sharpe"][-1]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_val_spread_mse": (
                    result["final_val_spread_mse"]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_val_total_profit": (
                    combo_metrics["policy_total_profit"]
                ),
                f"mlp_ecfm_ablation_{ablation_name}_val_annualized_sharpe": (
                    combo_metrics["policy_annualized_sharpe"]
                ),
            }
        )
    for policy_name, policy_ablation in alpha_ablation.items():
        if policy_ablation["alpha"].size == 0:
            continue
        best_profit_index = int(np.argmax(policy_ablation["total_profit"]))
        best_sharpe_index = int(np.argmax(policy_ablation["annualized_sharpe"]))
        metrics.update(
            {
                f"{policy_name}_alpha_ablation_best_profit_alpha": float(
                    policy_ablation["alpha"][best_profit_index]
                ),
                f"{policy_name}_alpha_ablation_best_profit": float(
                    policy_ablation["total_profit"][best_profit_index]
                ),
                f"{policy_name}_alpha_ablation_best_sharpe_alpha": float(
                    policy_ablation["alpha"][best_sharpe_index]
                ),
                f"{policy_name}_alpha_ablation_best_sharpe": float(
                    policy_ablation["annualized_sharpe"][best_sharpe_index]
                ),
            }
        )
    for key, result in combo_results.items():
        combo_metrics = result["metrics"]
        losses = result["losses"]
        metrics.update(
            {
                f"{key}_final_total_loss": losses["total"][-1],
                f"{key}_final_energy_loss": losses["energy"][-1],
                f"{key}_final_batch_sharpe": losses["batch_sharpe"][-1],
                f"{key}_val_total_profit": combo_metrics["policy_total_profit"],
                f"{key}_val_annualized_sharpe": combo_metrics[
                    "policy_annualized_sharpe"
                ],
                **prefix_metrics(f"{key}_val_policy", combo_metrics),
            }
        )
    logger.save_metrics_csv("metrics.csv", metrics)

    summary = {
        "config_file": "config.json",
        "metrics_file": "metrics.csv",
        "predictor_loss_plot": "predictor_losses.png",
        "validation_profit_plot": "validation_policy_profit_comparison.png",
        "spread_scatter_plot": "spread_prediction_scatter.png",
        "spread_direction_accuracy_plot": "spread_direction_accuracy.png",
        "profit_distribution_drawdown_plot": "profit_distribution_drawdown.png",
        "fmts_samples_steps_ablation_plot": (
            "fmts_samples_steps_ablation.png"
            if fmts_ablation_mse is not None
            else None
        ),
        "fmts_decision_sample_ablation_plot": (
            "fmts_decision_sample_ablation.png"
            if fmts_decision_ablation
            else None
        ),
        "e2e_regularizer_ablation_plot": (
            "e2e_regularizer_ablation.png"
            if e2e_regularizer_ablation
            else None
        ),
        "alpha_ablation_plot": "alpha_ablation.png" if alpha_ablation else None,
        "prediction_loss_arrays": "prediction_losses.npz",
        "input_normalization_stats": (
            "input_normalization_stats.npz"
            if train_loader.dataset.input_stats is not None
            else None
        ),
        "spread_prediction_diagnostics": "spread_prediction_diagnostics.npz",
        "fmts_samples_steps_ablation": (
            "fmts_samples_steps_ablation.npz"
            if fmts_ablation_mse is not None
            else None
        ),
        "fmts_decision_sample_ablation": (
            "fmts_decision_sample_ablation.npz"
            if fmts_decision_ablation
            else None
        ),
        "e2e_regularizer_ablation_metrics": (
            "e2e_regularizer_ablation_metrics.npz"
            if e2e_regularizer_ablation
            else None
        ),
        "alpha_ablation": "alpha_ablation.npz" if alpha_ablation else None,
        "dpds_val_policy_arrays": "dpds_val_policy_arrays.npz",
        "combos": {
            key: {
                "loss_arrays": f"{key}_losses.npz",
                "val_policy_arrays": f"{key}_val_policy_arrays.npz",
                "predictor_model": f"{key}_predictor.pt",
                "decision_model": f"{key}_decision_model.pt",
                "val_total_profit": result["metrics"]["policy_total_profit"],
                "val_annualized_sharpe": result["metrics"][
                    "policy_annualized_sharpe"
                ],
            }
            for key, result in combo_results.items()
        },
        "e2e_regularizer_ablation": {
            key: {
                "config": result["config"],
                "loss_arrays": f"mlp_ecfm_ablation_{key}_losses.npz",
                "val_policy_arrays": (
                    f"mlp_ecfm_ablation_{key}_val_policy_arrays.npz"
                ),
                "predictor_model": f"mlp_ecfm_ablation_{key}_predictor.pt",
                "decision_model": (
                    f"mlp_ecfm_ablation_{key}_decision_model.pt"
                ),
                "val_total_profit": result["metrics"]["policy_total_profit"],
                "val_annualized_sharpe": result["metrics"][
                    "policy_annualized_sharpe"
                ],
                "val_spread_mse": result["final_val_spread_mse"],
            }
            for key, result in e2e_regularizer_ablation.items()
        },
        "metrics": metrics,
    }
    logger.save_summary(summary)
    print("[save] Finished saving experiment artifacts.", flush=True)

    print(f"MLP pretrain val spread MSE: {mlp_pretrain_val_loss:.4f}")
    print(f"FM-TS pretrain val spread MSE: {fmts_pretrain_val_loss:.4f}")
    print(
        "DPDS validation policy profit: "
        f"{dpds_metrics['policy_total_profit']:.4f}"
    )
    print(
        "DPDS validation policy Sharpe: "
        f"{dpds_metrics['policy_annualized_sharpe']:.4f}"
    )
    for key, result in combo_results.items():
        print(
            f"{key} validation policy profit: "
            f"{result['metrics']['policy_total_profit']:.4f}"
        )
        print(
            f"{key} validation policy Sharpe: "
            f"{result['metrics']['policy_annualized_sharpe']:.4f}"
        )
    print(f"Saved comparison artifacts to {logger.run_dir}")
