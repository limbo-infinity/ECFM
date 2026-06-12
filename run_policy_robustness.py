import argparse
import copy
import csv
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
from main_predictor_comparison import split_summary
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


SEEDS = list(range(42, 52))
MATCHED_DECISION_CONFIGS = [
    {
        "name": "unfrozen_spread_reg_0p1",
        "risk_loss_mode": "none",
        "sharpe_loss_weight": 0.0,
        "prediction_regularizer_weight": 0.1,
        "freeze_prediction_during_e2e": False,
    },
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_hyperparams():
    return {
        "history_len": 30,
        "gap": 2,
        "batch_size": 32,
        "normalize_inputs": True,
        "include_time_features": True,
        "prediction_hidden_dim": 512,
        "prediction_num_hidden_layers": 5,
        "fmts_hidden_dim": 384,
        "fmts_num_hidden_layers": 3,
        "fmts_output_mode": "spread",
        "fmts_prediction_samples": 20,
        "fmts_prediction_steps": 5,
        "direct_bid_hidden_dim": 512,
        "direct_bid_hidden_layers": 3,
        "flow_hidden_dim": 512,
        "prediction_pretrain_epochs": 30,
        "fmts_pretrain_epochs": 30,
        "end_to_end_epochs": 100,
        "prediction_learning_rate": 1e-4,
        "fmts_learning_rate": 1e-4,
        "end_to_end_learning_rate": 5e-5,
        "prediction_weight_decay": 1e-3,
        "fmts_weight_decay": 1e-4,
        "end_to_end_weight_decay": 1e-4,
        "freeze_prediction_during_e2e": False,
        "prediction_regularizer_weight": 0.1,
        "risk_loss_mode": "none",
        "sharpe_loss_weight": 0.0,
        "variance_loss_weight": 5.0,
        "train_soft_q": False,
        "eval_soft_q": False,
        "q_temperature": 10.0,
        "train_num_samples": 20,
        "train_num_flow_steps": 10,
        "eval_num_samples": 20,
        "eval_num_flow_steps": 8,
        "langevin_steps": 0,
        "langevin_step_size": 1e-4,
        "dpds_grid_size": 25,
        "dpds_rho": 0.002,
        "alpha": 0.1,
        "penalty_weight": 1.0,
        "q_max": 1.0,
        "budget": 250000.0,
        "bid_low": -1000.0,
        "bid_up": 2000.0,
    }


def train_predictors(train_loader, val_loader, obs_dim, T, K, device, hyperparams):
    mlp_predictor = PricePredictor(
        obs_dim=obs_dim,
        hidden_dim=hyperparams["prediction_hidden_dim"],
        T=T,
        K=K,
        num_hidden_layers=hyperparams["prediction_num_hidden_layers"],
        output_mode="prices",
    ).to(device)
    mlp_optimizer = torch.optim.AdamW(
        mlp_predictor.parameters(),
        lr=hyperparams["prediction_learning_rate"],
        weight_decay=hyperparams["prediction_weight_decay"],
    )
    train_prediction_model(
        g_phi=mlp_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=mlp_optimizer,
        num_epochs=hyperparams["prediction_pretrain_epochs"],
        loss_mode="spread",
    )

    fmts_predictor = FMTSPricePredictor(
        obs_dim=obs_dim,
        hidden_dim=hyperparams["fmts_hidden_dim"],
        T=T,
        K=K,
        num_hidden_layers=hyperparams["fmts_num_hidden_layers"],
        output_mode=hyperparams["fmts_output_mode"],
        num_prediction_samples=hyperparams["fmts_prediction_samples"],
        num_prediction_steps=hyperparams["fmts_prediction_steps"],
    ).to(device)
    fmts_optimizer = torch.optim.AdamW(
        fmts_predictor.parameters(),
        lr=hyperparams["fmts_learning_rate"],
        weight_decay=hyperparams["fmts_weight_decay"],
    )
    train_fmts_prediction_model(
        g_phi=fmts_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=fmts_optimizer,
        num_epochs=hyperparams["fmts_pretrain_epochs"],
        loss_mode="spread",
    )

    return {
        "mlp": mlp_predictor,
        "fmts": fmts_predictor,
    }


def train_and_evaluate_direct(
    predictor,
    train_loader,
    val_loader,
    device,
    hyperparams,
):
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
    metrics, _ = evaluate_direct_bid_policy(
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
    return losses, metrics


def train_and_evaluate_ecfm(
    predictor,
    train_loader,
    val_loader,
    device,
    hyperparams,
):
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
    metrics, _ = evaluate_fixed_q_bid_policy(
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
    return losses, metrics


def append_row(rows, seed, config_name, policy_name, metrics, losses=None):
    row = {
        "seed": seed,
        "config": config_name,
        "policy": policy_name,
        "total_profit": metrics["policy_total_profit"],
        "annualized_sharpe": metrics["policy_annualized_sharpe"],
        "mean_daily_profit": metrics["policy_mean_daily_profit"],
        "std_daily_profit": metrics["policy_std_daily_profit"],
        "spread_mse": metrics.get("spread_mse", np.nan),
        "spread_mae": metrics.get("spread_mae", np.nan),
        "latency_ms_per_day": metrics.get("latency_ms_per_day", np.nan),
        "mean_budget_violation": metrics.get("policy_mean_budget_violation", np.nan),
        "max_budget_violation": metrics.get("policy_max_budget_violation", np.nan),
        "mean_l1_usage": metrics.get("policy_mean_l1_usage", np.nan),
        "max_l1_usage": metrics.get("policy_max_l1_usage", np.nan),
        "final_total_loss": np.nan,
        "final_energy_loss": np.nan,
        "final_batch_sharpe": np.nan,
    }
    if losses is not None:
        row["final_total_loss"] = losses["total"][-1]
        row["final_energy_loss"] = losses["energy"][-1]
        row["final_batch_sharpe"] = losses["batch_sharpe"][-1]
    rows.append(row)


def save_rows_csv(path, rows):
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    def finite_mean(values):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.nan
        return float(finite_values.mean())

    def finite_std(values):
        finite_values = values[np.isfinite(values)]
        if finite_values.size <= 1:
            return 0.0
        return float(finite_values.std(ddof=1))

    def finite_max(values):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.nan
        return float(finite_values.max())

    groups = {}
    for row in rows:
        key = (row["config"], row["policy"])
        groups.setdefault(key, []).append(row)

    aggregate = []
    for (config_name, policy_name), group_rows in sorted(groups.items()):
        profits = np.asarray([row["total_profit"] for row in group_rows], dtype=np.float32)
        sharpes = np.asarray(
            [row["annualized_sharpe"] for row in group_rows],
            dtype=np.float32,
        )
        latencies = np.asarray(
            [row.get("latency_ms_per_day", np.nan) for row in group_rows],
            dtype=np.float32,
        )
        max_budget_violations = np.asarray(
            [row.get("max_budget_violation", np.nan) for row in group_rows],
            dtype=np.float32,
        )
        mean_budget_violations = np.asarray(
            [row.get("mean_budget_violation", np.nan) for row in group_rows],
            dtype=np.float32,
        )
        spread_mses = np.asarray(
            [row.get("spread_mse", np.nan) for row in group_rows],
            dtype=np.float32,
        )
        aggregate.append(
            {
                "config": config_name,
                "policy": policy_name,
                "num_seeds": len(group_rows),
                "profit_mean": float(profits.mean()),
                "profit_std": float(profits.std(ddof=1)) if len(profits) > 1 else 0.0,
                "sharpe_mean": float(sharpes.mean()),
                "sharpe_std": float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0,
                "spread_mse_mean": finite_mean(spread_mses),
                "spread_mse_std": finite_std(spread_mses),
                "latency_ms_per_day_mean": finite_mean(latencies),
                "latency_ms_per_day_std": finite_std(latencies),
                "mean_budget_violation_mean": finite_mean(mean_budget_violations),
                "max_budget_violation_max": finite_max(max_budget_violations),
            }
        )
    return aggregate


def plot_aggregate(aggregate, output_path):
    policies = list(dict.fromkeys(row["policy"] for row in aggregate))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    x = np.arange(len(policies))
    profit_means = []
    profit_stds = []
    for policy_name in policies:
        matches = [row for row in aggregate if row["policy"] == policy_name]
        if not matches:
            continue
        profit_means.append(matches[0]["profit_mean"])
        profit_stds.append(matches[0]["profit_std"])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(
        x,
        profit_means,
        yerr=profit_stds,
        capsize=4,
        color=colors[: len(policies)],
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Validation total profit")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, profit_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    fig.suptitle("Matched policy robustness across seeds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched policy robustness experiments."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Random seeds to run. Defaults to 10 seeds: 42 through 51.",
    )
    parser.add_argument(
        "--run-name",
        default="matched_policy_robustness_sharpe10_10seeds",
        help="Experiment folder suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_hyperparams = default_hyperparams()
    seeds = list(args.seeds)
    logger = ExperimentLogger(run_name=args.run_name)
    logger.save_config(
        {
            "model_type": "matched_policy_robustness",
            "seeds": seeds,
            "matched_decision_configs": MATCHED_DECISION_CONFIGS,
            "base_hyperparams": base_hyperparams,
            "notes": (
                "Each decision config is applied to Direct MLP and ECFM policies "
                "with both MLP and FM-TS predictors. Results report mean/std "
                "across seeds."
            ),
        }
    )

    rows = []
    device = get_training_device()
    print(f"Robustness directory: {logger.run_dir}")
    print(f"Using device: {device}")

    for seed in seeds:
        print(f"[seed {seed}] Loading data and training predictors...", flush=True)
        set_seed(seed)
        train_loader, val_loader, metadata = load_train_val_from_zone_data(
            history_len=base_hyperparams["history_len"],
            gap=base_hyperparams["gap"],
            batch_size=base_hyperparams["batch_size"],
            normalize_inputs=base_hyperparams["normalize_inputs"],
            include_time_features=base_hyperparams["include_time_features"],
        )
        T = 1
        K = metadata["K"]
        obs_dim = train_loader.dataset.obs_dim
        seed_hyperparams = copy.deepcopy(base_hyperparams)
        seed_hyperparams.update({"T": T, "K": K, "obs_dim": obs_dim})

        if seed == seeds[0]:
            logger.save_json(
                "data_summary.json",
                {
                    "data": {
                        "source": "zone_data",
                        "num_days": metadata["num_days"],
                        "num_zones": metadata["num_zones"],
                        "hours_per_day": metadata["hours_per_day"],
                        "K": metadata["K"],
                        "obs_dim": obs_dim,
                        "start_date": metadata["dates"][0],
                        "end_date": metadata["dates"][-1],
                    },
                    "splits": {
                        "train": split_summary(train_loader, metadata),
                        "val": split_summary(val_loader, metadata),
                    },
                },
            )

        predictors = train_predictors(
            train_loader,
            val_loader,
            obs_dim,
            T,
            K,
            device,
            seed_hyperparams,
        )

        da_price_matrix = train_loader.dataset.da_prices.detach().cpu().numpy()
        rt_price_matrix = train_loader.dataset.rt_prices.detach().cpu().numpy()

        for config in MATCHED_DECISION_CONFIGS:
            config_name = config["name"]
            hyperparams = copy.deepcopy(seed_hyperparams)
            hyperparams.update(config)
            print(f"[seed {seed}] Matched config: {config_name}", flush=True)

            dpds_metrics, _ = evaluate_dpds_policy(
                da_prices=da_price_matrix,
                rt_prices=rt_price_matrix,
                target_indices=val_loader.dataset.target_indices,
                gap=hyperparams["gap"],
                budget=hyperparams["budget"],
                bid_low=hyperparams["bid_low"],
                bid_up=hyperparams["bid_up"],
                q_max=hyperparams["q_max"],
                alpha=hyperparams["alpha"],
                grid_size=hyperparams["dpds_grid_size"],
                rho=hyperparams["dpds_rho"],
            )
            append_row(rows, seed, config_name, "dpds", dpds_metrics)

            for predictor_name, base_predictor in predictors.items():
                direct_predictor = copy.deepcopy(base_predictor).to(device)
                direct_losses, direct_metrics = train_and_evaluate_direct(
                    direct_predictor,
                    train_loader,
                    val_loader,
                    device,
                    hyperparams,
                )
                append_row(
                    rows,
                    seed,
                    config_name,
                    f"{predictor_name}_direct",
                    direct_metrics,
                    direct_losses,
                )

                ecfm_predictor = copy.deepcopy(base_predictor).to(device)
                ecfm_losses, ecfm_metrics = train_and_evaluate_ecfm(
                    ecfm_predictor,
                    train_loader,
                    val_loader,
                    device,
                    hyperparams,
                )
                append_row(
                    rows,
                    seed,
                    config_name,
                    f"{predictor_name}_ecfm",
                    ecfm_metrics,
                    ecfm_losses,
                )

            save_rows_csv(logger.path("robustness_results_partial.csv"), rows)

    aggregate = aggregate_rows(rows)
    save_rows_csv(logger.path("robustness_results.csv"), rows)
    save_rows_csv(logger.path("robustness_aggregate.csv"), aggregate)
    plot_aggregate(aggregate, logger.path("robustness_aggregate.png"))
    logger.save_summary(
        {
            "results_csv": "robustness_results.csv",
            "partial_results_csv": "robustness_results_partial.csv",
            "aggregate_csv": "robustness_aggregate.csv",
            "aggregate_plot": "robustness_aggregate.png",
            "num_rows": len(rows),
            "num_aggregate_rows": len(aggregate),
        }
    )
    print(f"Saved robustness artifacts to {logger.run_dir}")

    for row in aggregate:
        print(
            f"{row['config']} | {row['policy']} | "
            f"profit {row['profit_mean']:.2f} +/- {row['profit_std']:.2f} | "
            f"Sharpe {row['sharpe_mean']:.3f} +/- {row['sharpe_std']:.3f}"
        )


if __name__ == "__main__":
    main()
